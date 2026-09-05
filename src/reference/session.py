"""Season-level reference ingest: API in, immutable snapshot and typed parquet out."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.config import INTERIM_ROOT
from src.reference.jolpica import JOLPICA_RAW_ROOT, JolpicaClient, JolpicaEmptyError
from src.reference import tables as build

logger = logging.getLogger(__name__)

REFERENCE_ROOT = INTERIM_ROOT / "reference"

TABLE_FILES = {
    "dim_event": "dim_event.parquet",
    "dim_session_schedule": "dim_session_schedule.parquet",
    "dim_driver": "dim_driver.parquet",
    "dim_constructor": "dim_constructor.parquet",
    "driver_entry": "driver_entry.parquet",
}


@dataclass(frozen=True)
class ReferenceResult:
    season: int
    root: Path
    snapshot_root: Path
    tables: dict[str, pd.DataFrame]
    rounds_ingested: tuple[int, ...]
    rounds_skipped: tuple[tuple[int, str], ...]
    requests_made: int
    cache_hits: int
    elapsed_s: float

    @property
    def counts(self) -> dict[str, int]:
        return {name: len(frame) for name, frame in self.tables.items()}


def ingest_reference(
    season: int,
    rounds: Sequence[int] | None = None,
    snapshot_date: str | None = None,
    out_root: Path | None = None,
    client: JolpicaClient | None = None,
) -> ReferenceResult:
    """Build every reference table for one season.

    Per-round entries are fetched one round at a time and a round that has no
    result yet -- a future race -- is skipped and reported rather than failing
    the season (global 3.1, per-record fault tolerance).
    """
    started = time.perf_counter()
    client = client or JolpicaClient()
    snapshot_date = snapshot_date or date.today().isoformat()

    races = client.races(season)
    drivers = client.drivers(season)
    constructors = client.constructors(season)

    event_table = build.build_event_table(races, season)
    schedule = build.build_session_schedule(races, season)
    driver_table = build.build_driver_table(drivers, season)
    constructor_table = build.build_constructor_table(constructors, season)

    wanted = list(rounds) if rounds is not None else [int(race["round"]) for race in races]
    entry_payloads: list[dict] = []
    skipped: list[tuple[int, str]] = []
    for round_number in wanted:
        try:
            entry_payloads.extend(client.qualifying(season, round_number))
        except JolpicaEmptyError:
            try:
                entry_payloads.extend(client.results(season, round_number))
            except JolpicaEmptyError as exc:
                skipped.append((round_number, "no qualifying or race result yet"))
                logger.warning("reference_round_skipped season=%d round=%d reason=%s", season, round_number, exc)

    entries = build.build_driver_entries(entry_payloads, season) if entry_payloads else None
    if entries is None:
        raise JolpicaEmptyError(f"season {season}: no round produced a driver entry")

    built = {
        "dim_event": event_table,
        "dim_session_schedule": schedule,
        "dim_driver": driver_table,
        "dim_constructor": constructor_table,
        "driver_entry": entries,
    }

    snapshot_root = Path(JOLPICA_RAW_ROOT) / snapshot_date / str(season)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    for name, payload in (("races", races), ("drivers", drivers), ("constructors", constructors),
                          ("driver_entries", entry_payloads)):
        (snapshot_root / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_root = Path(out_root) if out_root is not None else (REFERENCE_ROOT / str(season))
    out_root.mkdir(parents=True, exist_ok=True)
    for name, frame in built.items():
        frame.to_parquet(out_root / TABLE_FILES[name], index=False)

    elapsed = time.perf_counter() - started
    (out_root / "reference_meta.json").write_text(
        json.dumps(
            {
                "season": season,
                "snapshot": str(snapshot_root),
                "snapshot_date": snapshot_date,
                "source": "https://api.jolpi.ca/ergast/f1",
                "rounds_ingested": [r for r in wanted if r not in {s for s, _ in skipped}],
                "rounds_skipped": [{"round": r, "reason": why} for r, why in skipped],
                "counts": {name: len(frame) for name, frame in built.items()},
                "requests_made": client.requests_made,
                "cache_hits": client.cache_hits,
                "elapsed_s": elapsed,
                "note": (
                    "A driver's constructor comes from driver_entry, per round, never from matching "
                    "FastF1's team string against a constructor name: the two agree only 6 times in 10."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "reference_complete season=%d events=%d sessions=%d drivers=%d constructors=%d entries=%d "
        "rounds=%d skipped=%d requests=%d cache_hits=%d elapsed_s=%.2f",
        season, len(event_table), len(schedule), len(driver_table), len(constructor_table), len(entries),
        len(wanted) - len(skipped), len(skipped), client.requests_made, client.cache_hits, elapsed,
    )
    return ReferenceResult(
        season=season,
        root=out_root,
        snapshot_root=snapshot_root,
        tables=built,
        rounds_ingested=tuple(r for r in wanted if r not in {s for s, _ in skipped}),
        rounds_skipped=tuple(skipped),
        requests_made=client.requests_made,
        cache_hits=client.cache_hits,
        elapsed_s=elapsed,
    )


def load_reference(season: int, root: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read the reference tables for a season back off disk."""
    root = Path(root) if root is not None else (REFERENCE_ROOT / str(season))
    frames: dict[str, pd.DataFrame] = {}
    for name, filename in TABLE_FILES.items():
        path = root / filename
        if path.exists():
            frames[name] = pd.read_parquet(path)
        else:
            logger.warning("reference_table_missing season=%d table=%s path=%s", season, name, path)
    return frames
