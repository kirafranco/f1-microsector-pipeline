"""One row per ingested session, so the method can be judged per circuit.

F010 validates a session against official timing and writes its verdict beside
that session's processed data. That is the right place for it and the wrong
place to read it from: the question this feature exists to answer -- does the
alignment hold at circuits other than the one it was built on -- is a question
about all of them at once.

So this walks the processed layer, reads each `ground_truth_report.json`, and
lays them out as a table. Nothing is recomputed: a row here is exactly what
that session's own validation concluded, which means the table cannot drift
from the reports and a session missing from it is a session that was never
validated rather than one that failed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path, PurePosixPath

import pandas as pd

from src.config import FASTF1_RAW_ROOT, PROCESSED_ROOT

logger = logging.getLogger(__name__)

REPORT_NAME = "ground_truth_report.json"

#: Column order of the summary. Identity first, then the figures F010 gates on,
#: then the ones it reports without gating, then the verdicts.
COLUMNS = (
    "session", "circuit", "season", "round", "session_code", "snapshot_date",
    "laps", "laps_gated", "flagged",
    "lap_residual_median_s", "lap_residual_std_s", "lap_residual_p95_s",
    "closure_p50_s", "closure_p95_s",
    "s1_median_s", "s2_median_s", "s3_median_s",
    "s1_std_s", "s2_std_s", "s3_std_s",
    "driven_median_m", "official_lap_length_m", "driven_pct_of_official",
    "line_start_m", "line_end_m", "line_start_std_m", "line_end_std_m",
    "v_min_std_median_kmh", "v_min_std_p95_kmh",
    "lap_reconstruction_ok", "delta_closure_ok", "sector_times_ok",
    "driven_distance_ok", "v_min_stability_ok", "timing_line_spread_ok", "all_ok",
)


class SeasonSummaryError(RuntimeError):
    """The processed layer holds nothing that can be summarised."""


def _row(path: Path, meta: dict | None) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    acceptance = payload["acceptance"]
    checks = acceptance.get("checks", {})
    sectors_med = acceptance.get("sector_median_s", {})
    sectors_std = acceptance.get("sector_std_s", {})
    official = payload.get("official_lap_length_m")
    driven = acceptance.get("driven_median_m")
    meta = meta or {}

    return {
        "session": path.parent.name,
        "circuit": meta.get("location"),
        "season": meta.get("season"),
        "round": meta.get("round_number"),
        "session_code": meta.get("session_requested") or meta.get("session_name"),
        "snapshot_date": meta.get("snapshot_date"),
        "laps": acceptance.get("laps"),
        "laps_gated": acceptance.get("laps_gated"),
        "flagged": len(acceptance.get("flagged", [])),
        "lap_residual_median_s": acceptance.get("lap_residual_median_s"),
        "lap_residual_std_s": acceptance.get("lap_residual_std_s"),
        "lap_residual_p95_s": (acceptance.get("lap_residual") or {}).get("p95"),
        "closure_p50_s": (acceptance.get("closure") or {}).get("p50"),
        "closure_p95_s": (acceptance.get("closure") or {}).get("p95"),
        "s1_median_s": sectors_med.get("s1"),
        "s2_median_s": sectors_med.get("s2"),
        "s3_median_s": sectors_med.get("s3"),
        "s1_std_s": sectors_std.get("s1"),
        "s2_std_s": sectors_std.get("s2"),
        "s3_std_s": sectors_std.get("s3"),
        "driven_median_m": driven,
        "official_lap_length_m": official,
        "driven_pct_of_official": (
            (driven - official) / official * 100.0
            if driven is not None and official else None
        ),
        "line_start_m": acceptance.get("line_start_m"),
        "line_end_m": acceptance.get("line_end_m"),
        "line_start_std_m": acceptance.get("line_start_std_m"),
        "line_end_std_m": acceptance.get("line_end_std_m"),
        "v_min_std_median_kmh": acceptance.get("v_min_std_median_kmh"),
        "v_min_std_p95_kmh": acceptance.get("v_min_std_p95_kmh"),
        "lap_reconstruction_ok": checks.get("lap_reconstruction"),
        "delta_closure_ok": checks.get("delta_closure"),
        "sector_times_ok": checks.get("sector_times"),
        "driven_distance_ok": checks.get("driven_distance"),
        "v_min_stability_ok": checks.get("v_min_stability"),
        "timing_line_spread_ok": checks.get("timing_line_spread"),
        "all_ok": checks.get("all"),
    }


def _session_meta(report_path: Path) -> dict | None:
    """The snapshot metadata behind a report: the circuit, the round, the date.

    The report records the snapshot it was built from, but that path may have
    been written inside the Airflow container. Only its tail identifies the
    snapshot -- `<date>/<slug>` -- so it is resolved against this machine's own
    raw root rather than trusted as written.
    """
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    recorded = str(payload.get("snapshot") or "")
    if not recorded:
        return None

    tail = PurePosixPath(recorded.replace("\\", "/")).parts[-2:]
    if len(tail) < 2:
        return None
    meta_path = FASTF1_RAW_ROOT.joinpath(*tail) / "session_meta.json"
    if not meta_path.exists():
        logger.warning("season_snapshot_missing report=%s snapshot=%s", report_path.parent.name, recorded)
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["snapshot_date"] = tail[0]
    return meta


def summarise_season(processed_root: Path | None = None) -> pd.DataFrame:
    """Every validated session in the processed layer, one row each."""
    processed_root = Path(processed_root) if processed_root else PROCESSED_ROOT
    if not processed_root.exists():
        raise SeasonSummaryError(f"no processed layer at {processed_root}")

    reports = sorted(processed_root.glob(f"*/{REPORT_NAME}"))
    if not reports:
        raise SeasonSummaryError(f"no {REPORT_NAME} under {processed_root}")

    rows = [_row(path, _session_meta(path)) for path in reports]
    frame = pd.DataFrame(rows)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[list(COLUMNS)].sort_values(["season", "round", "session_code"],
                                             na_position="last").reset_index(drop=True)
    logger.info("season_summarised sessions=%d passing=%s", len(frame),
                int(frame["all_ok"].fillna(False).sum()))
    return frame


def write_summary(frame: pd.DataFrame, out_root: Path | None = None,
                  name: str = "validation_by_session") -> Path:
    """Write the summary beside the sessions it summarises."""
    out_root = Path(out_root) if out_root else (PROCESSED_ROOT / "season")
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    logger.info("season_summary_written path=%s rows=%d", path, len(frame))
    return path


def by_circuit(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per circuit: how the method behaved wherever it was asked to run.

    This is the table D5's fallback clause was always going to need. A circuit
    that fails here is not a circuit to quietly switch methods on -- it is
    evidence for a decision, taken in the open.
    """
    if frame.empty:
        return frame
    grouped = frame.groupby("circuit", dropna=False).agg(
        sessions=("session", "count"),
        passing=("all_ok", lambda s: int(s.fillna(False).sum())),
        laps=("laps", "sum"),
        lap_residual_median_s=("lap_residual_median_s", "median"),
        lap_residual_std_s=("lap_residual_std_s", "median"),
        closure_p95_s=("closure_p95_s", "median"),
        driven_pct_of_official=("driven_pct_of_official", "median"),
        line_start_std_m=("line_start_std_m", "max"),
    )
    grouped["all_sessions_pass"] = grouped["passing"] == grouped["sessions"]
    return grouped.sort_values("lap_residual_std_s", ascending=False)
