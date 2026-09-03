"""Session-level segmentation: grid + corners + alignment frame in, micro-sector tables out."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.config import INTERIM_ROOT
from src.grid.resample import GRID_SPACING_M
from src.segment.corners import corner_positions, load_frame, parse_corner_label, parse_reference_lap
from src.segment.events import EventParams, detect_events, median_traces
from src.segment.phases import build_corner_phases, build_fixed_bins
from src.segment.validation import SegmentationReport, measure_session

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentationResult:
    root: Path
    lap_length_m: float
    events: pd.DataFrame
    microsectors: pd.DataFrame
    corners: pd.DataFrame
    report: SegmentationReport
    elapsed_s: float


def _corner_event_ids(corners: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """Event each corner labels, or NA for the corners that belong to none."""
    lookup: dict[int, int] = {}
    for _, e in events.iterrows():
        for number in parse_corner_label(e["corners"]):
            lookup.setdefault(number, int(e["event_id"]))
    return corners["number"].astype(int).map(lookup).astype("Int16")


def segment_session(
    grid_root: Path,
    snapshot_root: Path,
    aligned_root: Path,
    out_root: Path | None = None,
    params: EventParams = EventParams(),
    grid_m: float = GRID_SPACING_M,
    jackknife_trials: int = 10,
) -> SegmentationResult:
    """Detect events and write both micro-sector grains for one session.

    Idempotent: identical inputs produce identical tables, and re-running
    replaces the files in place.
    """
    started = time.perf_counter()
    grid = pd.read_parquet(grid_root / "grid.parquet")
    corners_raw = pd.read_parquet(snapshot_root / "circuit_corners.parquet")
    meta = json.loads((aligned_root / "alignment_meta.json").read_text(encoding="utf-8"))

    frame = load_frame(meta)
    reference_lap = parse_reference_lap(meta["reference_line_lap"])
    corners = corner_positions(grid, corners_raw, frame, reference_lap)

    traces = median_traces(grid)
    lap_length_m = float(len(traces) * grid_m)
    events = detect_events(traces, corners, params, grid_m)
    logger.info(
        "events_detected n=%d marginal=%d lap_length_m=%.0f labels=%s",
        len(events),
        int(events["marginal"].sum()) if len(events) else 0,
        lap_length_m,
        events["corners"].tolist() if len(events) else [],
    )

    microsectors = pd.concat(
        [build_corner_phases(events, lap_length_m, grid_m), build_fixed_bins(lap_length_m, 100.0, grid_m)],
        ignore_index=True,
    )
    duplicates = int(microsectors.duplicated(["grain", "microsector_id"]).sum())
    if duplicates:
        raise RuntimeError(f"{grid_root}: {duplicates} duplicate (grain, microsector_id) rows")

    corners = corners.assign(event_id=_corner_event_ids(corners, events))
    report = measure_session(grid, corners, events, microsectors, lap_length_m, params, grid_m, jackknife_trials)

    out_root = out_root or (INTERIM_ROOT / "microsectors" / grid_root.name)
    out_root.mkdir(parents=True, exist_ok=True)
    microsectors.to_parquet(out_root / "microsectors.parquet", index=False)
    events.to_parquet(out_root / "events.parquet", index=False)
    corners.to_parquet(out_root / "corners_aligned.parquet", index=False)

    elapsed = time.perf_counter() - started
    (out_root / "segmentation_meta.json").write_text(
        json.dumps(
            {
                "grid": str(grid_root),
                "snapshot": str(snapshot_root),
                "aligned": str(aligned_root),
                "reference_lap": f"{reference_lap[0]} L{reference_lap[1]}",
                "grid_m": grid_m,
                "lap_length_m": lap_length_m,
                "params": asdict(params),
                "events": int(len(events)),
                "microsectors": {
                    grain: int(count) for grain, count in microsectors["grain"].value_counts().items()
                },
                "elapsed_s": elapsed,
                "acceptance": report.to_dict(),
                "limitation": (
                    "Boundaries are session medians at grid resolution (D6: ~+/-20 m). Per-lap "
                    "metrics must be computed inside event windows, not from a grid point's phase "
                    "label. microsector_id is per session; compare across sessions by corner label."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = SegmentationResult(
        root=out_root,
        lap_length_m=lap_length_m,
        events=events,
        microsectors=microsectors,
        corners=corners,
        report=report,
        elapsed_s=elapsed,
    )
    logger.info(
        "segmentation_complete events=%d microsectors=%d partition_ok=%s apex_p95_m=%.1f "
        "brake_p95_m=%.1f jackknife=%d/%d acceptance_ok=%s elapsed_s=%.2f",
        len(events),
        len(microsectors),
        report.partition_ok,
        report.apex_dev.p95,
        report.brake_dev.p95,
        report.jackknife_count_matches,
        report.jackknife_trials,
        report.ok,
        elapsed,
    )
    return result
