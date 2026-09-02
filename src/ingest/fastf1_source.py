"""FastF1 session ingestion (F002).

Pulls one session down once and lands it as immutable, typed parquet under
``data/raw/fastf1/<snapshot-date>/<session-slug>/``. Everything downstream in
slice 1 reads those files; nothing downstream talks to FastF1.

Guarantees:
  * the FastF1 cache is enabled before any session load, from one place;
  * every network call retries transient failures with exponential backoff;
  * one bad driver is logged and skipped, never aborting the session;
  * a completed snapshot is immutable and re-ingesting it is a no-op.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import fastf1
import pandas as pd

from src.config import FASTF1_CACHE_DIR, FASTF1_RAW_ROOT, session_slug
from src.ingest import schemas
from src.ingest.retry import PermanentIngestError, retry_with_backoff

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
SESSION_META_NAME = "session_meta.json"
MANIFEST_SCHEMA_VERSION = 1

#: Internal helper columns injected before coercion; see schemas.CAR_TELEMETRY.
_DRIVER_COL = "_driver"
_LAP_COL = "_lap_number"


@dataclass(frozen=True)
class SessionSnapshot:
    """Handle on one completed raw snapshot."""

    season: int
    event: str
    session: str
    snapshot_date: date
    root: Path
    drivers_ingested: list[str]
    drivers_skipped: dict[str, str]

    def path(self, artefact: str) -> Path:
        return self.root / f"{artefact}.parquet"


class SnapshotCorruptError(PermanentIngestError):
    """An existing snapshot exists but does not match its manifest."""


# --------------------------------------------------------------------------
# cache and session loading
# --------------------------------------------------------------------------


def enable_cache(cache_dir: Path | None = None) -> Path:
    """Enable the FastF1 cache. The only place that does so.

    Resolved at call time rather than bound as a default so that tests can
    redirect it without writing into the real project data directory.
    """
    cache_dir = cache_dir or FASTF1_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    logger.debug("fastf1_cache_enabled path=%s", cache_dir)
    return cache_dir


def _load_session(season: int, event: str, session: str) -> Any:
    """Enable the cache, then fetch and load the session with backoff."""
    enable_cache()

    def _fetch() -> Any:
        sess = fastf1.get_session(season, event, session)
        sess.load(laps=True, telemetry=True, weather=True, messages=False)
        return sess

    return retry_with_backoff(_fetch, description=f"load {season} {event} {session}")


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def _iter_driver_laps(driver_laps: Any) -> Iterator[Any]:
    """Yield Lap objects. Isolated so tests can supply a stand-in Laps."""
    for _, lap in driver_laps.iterlaps():
        yield lap


def _extract_driver_telemetry(
    driver_laps: Any, driver: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Car and position telemetry for one driver, tagged with lap number.

    Per-lap fault tolerance: a lap whose telemetry is missing or malformed is
    logged and skipped. A driver with no usable lap raises, and the caller
    skips that driver.
    """
    car_parts: list[pd.DataFrame] = []
    pos_parts: list[pd.DataFrame] = []
    skipped_laps = 0

    for lap in _iter_driver_laps(driver_laps):
        lap_number = lap["LapNumber"]
        try:
            car = lap.get_car_data()
            pos = lap.get_pos_data()
        except Exception as exc:  # noqa: BLE001 - one bad lap must not stop a driver
            skipped_laps += 1
            logger.warning(
                "lap_skipped driver=%s lap=%s reason=%s: %s",
                driver,
                lap_number,
                type(exc).__name__,
                exc,
            )
            continue

        if car is None or len(car) == 0 or pos is None or len(pos) == 0:
            skipped_laps += 1
            logger.warning("lap_skipped driver=%s lap=%s reason=empty", driver, lap_number)
            continue

        car_parts.append(_tag(car, driver, lap_number, default_source="car"))
        pos_parts.append(_tag(pos, driver, lap_number, default_source="pos"))

    if not car_parts:
        raise PermanentIngestError(
            f"driver {driver}: no lap yielded usable telemetry "
            f"({skipped_laps} lap(s) skipped)"
        )

    if skipped_laps:
        logger.info("driver_partial driver=%s laps_skipped=%d", driver, skipped_laps)

    return (
        pd.concat(car_parts, ignore_index=True),
        pd.concat(pos_parts, ignore_index=True),
    )


def _tag(frame: pd.DataFrame, driver: str, lap_number: Any, *, default_source: str) -> pd.DataFrame:
    """Attach driver/lap identity and guarantee the optional Source column."""
    # frame is a fastf1 Telemetry (a DataFrame subclass); .copy() rather than
    # pd.DataFrame(frame), which passes the block manager and is deprecated.
    out = frame.copy()
    out[_DRIVER_COL] = driver
    out[_LAP_COL] = lap_number
    if "Source" not in out.columns:
        out["Source"] = default_source
    if "Status" not in out.columns and default_source == "pos":
        out["Status"] = pd.NA
    return out


def _extract_corners(session: Any) -> pd.DataFrame:
    circuit_info = session.get_circuit_info()
    if circuit_info is None or not hasattr(circuit_info, "corners"):
        raise PermanentIngestError("session exposes no circuit info; F008 cannot align without it")
    corners = circuit_info.corners.copy()
    if "Letter" not in corners.columns:
        corners["Letter"] = ""
    corners["Letter"] = corners["Letter"].fillna("")
    return corners


def _session_meta(session: Any, season: int, event: str, session_id: str) -> dict[str, Any]:
    """Best-effort descriptive metadata. Never fails the ingest."""
    meta: dict[str, Any] = {
        "season": season,
        "event_requested": event,
        "session_requested": session_id,
    }
    for key, getter in (
        ("session_name", lambda: session.name),
        ("session_date", lambda: str(session.date)),
        ("event_name", lambda: session.event["EventName"]),
        ("country", lambda: session.event["Country"]),
        ("location", lambda: session.event["Location"]),
        ("round_number", lambda: int(session.event["RoundNumber"])),
    ):
        try:
            meta[key] = getter()
        except Exception:  # noqa: BLE001 - metadata is descriptive, not load-bearing
            meta[key] = None
    return meta


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_artefact(frame: pd.DataFrame, schema: schemas.ArtefactSchema, target_dir: Path) -> Path:
    """Coerce, validate uniqueness, write parquet."""
    typed = schemas.coerce(frame, schema)
    schemas.assert_unique(typed, schema)
    path = target_dir / f"{schema.name}.parquet"
    typed.to_parquet(path, index=False)
    logger.info("artefact_written name=%s rows=%d path=%s", schema.name, len(typed), path)
    return path


def _assert_referential_integrity(car: pd.DataFrame, laps: pd.DataFrame) -> None:
    """Every telemetry (driver, lap) must exist in laps (global CLAUDE.md 3.3)."""
    lap_keys = set(zip(laps["driver"].astype(str), laps["lap_number"].astype("Int64")))
    tel_keys = set(zip(car["driver"].astype(str), car["lap_number"].astype("Int64")))
    orphans = tel_keys - lap_keys
    if orphans:
        raise schemas.SchemaError(
            f"car_telemetry references {len(orphans)} (driver, lap) pair(s) absent "
            f"from laps; first: {sorted(orphans)[:5]}"
        )


def _load_existing(root: Path) -> SessionSnapshot | None:
    """Return the snapshot at `root` if it is present and intact."""
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for filename, recorded in manifest.get("files", {}).items():
        path = root / filename
        if not path.is_file():
            raise SnapshotCorruptError(f"{root}: manifest lists {filename}, which is missing")
        actual = _sha256(path)
        if actual != recorded["sha256"]:
            raise SnapshotCorruptError(
                f"{root}: {filename} checksum mismatch "
                f"(manifest {recorded['sha256'][:12]}..., on disk {actual[:12]}...). "
                "Raw snapshots are immutable; investigate rather than overwrite."
            )

    return SessionSnapshot(
        season=manifest["season"],
        event=manifest["event"],
        session=manifest["session"],
        snapshot_date=date.fromisoformat(manifest["snapshot_date"]),
        root=root,
        drivers_ingested=list(manifest["drivers_ingested"]),
        drivers_skipped=dict(manifest["drivers_skipped"]),
    )


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------


def ingest_session(
    season: int,
    event: str,
    session: str,
    snapshot_date: date | None = None,
    drivers: Iterable[str] | None = None,
) -> SessionSnapshot:
    """Ingest one FastF1 session into an immutable raw snapshot.

    Idempotent: if the snapshot already exists and matches its manifest, it is
    returned without touching the network.
    """
    snapshot_date = snapshot_date or datetime.now(timezone.utc).date()
    slug = session_slug(season, event, session)
    root = FASTF1_RAW_ROOT / snapshot_date.isoformat() / slug

    existing = _load_existing(root)
    if existing is not None:
        logger.info("snapshot_reused root=%s drivers=%d", root, len(existing.drivers_ingested))
        return existing

    started = datetime.now(timezone.utc)
    loaded = _load_session(season, event, session)

    all_laps = loaded.laps
    if all_laps is None or len(all_laps) == 0:
        raise PermanentIngestError(f"{slug}: session loaded but contains no laps")

    requested = list(drivers) if drivers is not None else sorted(
        str(d) for d in pd.Series(all_laps["Driver"]).dropna().unique()
    )

    car_parts: list[pd.DataFrame] = []
    pos_parts: list[pd.DataFrame] = []
    ingested: list[str] = []
    skipped: dict[str, str] = {}

    for driver in requested:
        driver_laps = all_laps[all_laps["Driver"] == driver]
        try:
            if len(driver_laps) == 0:
                raise PermanentIngestError(f"driver {driver}: no laps in session")
            car, pos = _extract_driver_telemetry(driver_laps, driver)
        except Exception as exc:  # noqa: BLE001 - per-driver fault tolerance
            reason = f"{type(exc).__name__}: {exc}"
            skipped[driver] = reason
            logger.warning("driver_skipped driver=%s reason=%s", driver, reason)
            continue

        car_parts.append(car)
        pos_parts.append(pos)
        ingested.append(driver)

    if not ingested:
        raise PermanentIngestError(
            f"{slug}: every requested driver failed ({len(skipped)} skipped)"
        )

    laps_frame = all_laps[all_laps["Driver"].isin(requested)]

    staging = root.parent / f"{root.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        typed_laps = schemas.coerce(laps_frame, schemas.LAPS)
        schemas.assert_unique(typed_laps, schemas.LAPS)

        car_frame = pd.concat(car_parts, ignore_index=True)
        typed_car = schemas.coerce(car_frame, schemas.CAR_TELEMETRY)
        schemas.assert_unique(typed_car, schemas.CAR_TELEMETRY)
        _assert_referential_integrity(typed_car, typed_laps)

        written: dict[str, Path] = {}
        typed_laps.to_parquet(staging / "laps.parquet", index=False)
        written["laps.parquet"] = staging / "laps.parquet"
        typed_car.to_parquet(staging / "car_telemetry.parquet", index=False)
        written["car_telemetry.parquet"] = staging / "car_telemetry.parquet"

        written["pos_data.parquet"] = _write_artefact(
            pd.concat(pos_parts, ignore_index=True), schemas.POS_DATA, staging
        )
        written["weather.parquet"] = _write_artefact(
            loaded.weather_data.copy(), schemas.WEATHER, staging
        )
        written["circuit_corners.parquet"] = _write_artefact(
            _extract_corners(loaded), schemas.CIRCUIT_CORNERS, staging
        )

        meta_path = staging / SESSION_META_NAME
        meta_path.write_text(
            json.dumps(_session_meta(loaded, season, event, session), indent=2),
            encoding="utf-8",
        )
        written[SESSION_META_NAME] = meta_path

        rows_per_driver = {
            driver: int((typed_car["driver"] == driver).sum()) for driver in ingested
        }

        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "season": season,
            "event": event,
            "session": session,
            "slug": slug,
            "snapshot_date": snapshot_date.isoformat(),
            "ingested_at_utc": started.isoformat(),
            "duration_s": round((datetime.now(timezone.utc) - started).total_seconds(), 3),
            "versions": {
                "fastf1": fastf1.__version__,
                "pandas": pd.__version__,
            },
            "drivers_requested": requested,
            "drivers_ingested": ingested,
            "drivers_skipped": skipped,
            "telemetry_rows_per_driver": rows_per_driver,
            "files": {
                name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
                for name, path in sorted(written.items())
            },
        }
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        root.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    logger.info(
        "snapshot_complete slug=%s root=%s drivers_ingested=%d drivers_skipped=%d duration_s=%.1f",
        slug,
        root,
        len(ingested),
        len(skipped),
        (datetime.now(timezone.utc) - started).total_seconds(),
    )

    return SessionSnapshot(
        season=season,
        event=event,
        session=session,
        snapshot_date=snapshot_date,
        root=root,
        drivers_ingested=ingested,
        drivers_skipped=skipped,
    )
