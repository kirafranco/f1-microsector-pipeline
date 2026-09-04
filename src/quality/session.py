"""Session-level quality check: load every artefact, validate, write the report."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.config import PROCESSED_ROOT
from src.quality.contracts import CONTRACTS, OPTIONAL_TABLES, REPORTED_RANGES
from src.quality.engine import QualityReport, TableContract, findings_frame, validate_tables

logger = logging.getLogger(__name__)

#: Where each artefact lives, relative to the roots handed to check_session.
ARTEFACTS: dict[str, tuple[str, str]] = {
    "laps": ("snapshot", "laps.parquet"),
    "car_telemetry": ("snapshot", "car_telemetry.parquet"),
    "pos_data": ("snapshot", "pos_data.parquet"),
    "weather": ("snapshot", "weather.parquet"),
    "circuit_corners": ("snapshot", "circuit_corners.parquet"),
    "telemetry_aligned": ("aligned", "telemetry_aligned.parquet"),
    "grid": ("grid", "grid.parquet"),
    "events": ("microsectors", "events.parquet"),
    "microsectors": ("microsectors", "microsectors.parquet"),
    "corners_aligned": ("microsectors", "corners_aligned.parquet"),
    "delta_t": ("processed", "delta_t.parquet"),
    "microsector_times": ("processed", "microsector_times.parquet"),
    "microsector_summary": ("processed", "microsector_summary.parquet"),
    "corner_metrics": ("processed", "corner_metrics.parquet"),
    "lap_summary": ("processed", "lap_summary.parquet"),
    "ground_truth": ("processed", "ground_truth.parquet"),
    "v_min_stability": ("processed", "v_min_stability.parquet"),
}


@dataclass(frozen=True)
class QualityResult:
    root: Path
    report: QualityReport
    findings: pd.DataFrame
    ranges: dict
    elapsed_s: float

    @property
    def ok(self) -> bool:
        return self.report.ok


def load_artefacts(
    snapshot_root: Path, aligned_root: Path, grid_root: Path, microsector_root: Path, processed_root: Path
) -> dict[str, pd.DataFrame]:
    """Read every artefact that exists. A missing one is left out, not faked."""
    roots = {
        "snapshot": Path(snapshot_root),
        "aligned": Path(aligned_root),
        "grid": Path(grid_root),
        "microsectors": Path(microsector_root),
        "processed": Path(processed_root),
    }
    frames: dict[str, pd.DataFrame] = {}
    for name, (root_key, filename) in ARTEFACTS.items():
        path = roots[root_key] / filename
        if path.exists():
            frames[name] = pd.read_parquet(path)
        elif name not in OPTIONAL_TABLES:
            logger.warning("quality_artefact_missing table=%s path=%s", name, path)
    return frames


def observed_ranges(frames: Mapping[str, pd.DataFrame]) -> dict:
    """Min and max of the watched columns, reported next to the contract bounds."""
    out: dict[str, dict[str, list[float]]] = {}
    for table, columns in REPORTED_RANGES.items():
        frame = frames.get(table)
        if frame is None or frame.empty:
            continue
        table_ranges: dict[str, list[float]] = {}
        for column in columns:
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if len(values):
                table_ranges[column] = [float(values.min()), float(values.max())]
        if table_ranges:
            out[table] = table_ranges
    return out


def check_session(
    snapshot_root: Path,
    aligned_root: Path,
    grid_root: Path,
    microsector_root: Path,
    processed_root: Path,
    out_root: Path | None = None,
    contracts: Mapping[str, TableContract] = CONTRACTS,
) -> QualityResult:
    """Validate a whole session and write the report next to its processed data."""
    started = time.perf_counter()
    frames = load_artefacts(snapshot_root, aligned_root, grid_root, microsector_root, processed_root)
    if not frames:
        raise FileNotFoundError(f"no artefacts found under {processed_root} and its sibling roots")

    report = validate_tables(frames, contracts)
    findings = findings_frame(report)
    ranges = observed_ranges(frames)
    elapsed = time.perf_counter() - started

    out_root = Path(out_root) if out_root is not None else (PROCESSED_ROOT / Path(grid_root).name)
    out_root.mkdir(parents=True, exist_ok=True)
    findings.to_parquet(out_root / "quality_findings.parquet", index=False)
    payload = report.to_dict()
    payload["elapsed_s"] = elapsed
    payload["tables_checked"] = sorted(frames)
    payload["observed_ranges"] = ranges
    payload["limitation"] = (
        "Ranges are physical envelopes, not statistics: a slow lap passes and a broken unit does not. "
        "Cross-table arithmetic (do sector times sum to the lap?) is F010's ground truth, not a load gate."
    )
    (out_root / "quality_report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    logger.info(
        "quality_complete tables=%d rows=%d errors=%d warnings=%d ok=%s elapsed_s=%.2f",
        len(frames), int(sum(len(f) for f in frames.values())),
        len(report.errors), len(report.warnings), report.ok, elapsed,
    )
    return QualityResult(root=out_root, report=report, findings=findings, ranges=ranges, elapsed_s=elapsed)
