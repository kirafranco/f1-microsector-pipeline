"""Session-level metrics: grid + micro-sectors + official laps in, feature tables out."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_ROOT
from src.grid.resample import GRID_SPACING_M
from src.metrics import validation
from src.metrics.corners import corner_metrics
from src.metrics.delta import delta_t, time_curves
from src.metrics.microsectors import microsector_times, summarise_microsectors
from src.metrics.reference import ReferenceSpec, lap_label, resolve_reference

logger = logging.getLogger(__name__)

LAP_SUMMARY_SCHEMA: dict[str, str] = {
    "driver": "string",
    "lap_number": "Int16",
    "lap_time_s": "float32",
    "grid_time_s": "float32",
    "n_points": "Int32",
    "delta_t_end_s": "float32",
    "official_delta_s": "float32",
    "residual_s": "float32",
    "s1_grid_s": "float32",
    "s1_official_s": "float32",
    "s2_grid_s": "float32",
    "s2_official_s": "float32",
    "s3_grid_s": "float32",
    "s3_official_s": "float32",
    "compound": "string",
    "stint": "Int16",
    "tyre_life": "Int16",
    "team": "string",
    "reference": "string",
    "is_reference": "boolean",
}


@dataclass(frozen=True)
class MetricsResult:
    root: Path
    reference: pd.Series
    report: validation.MetricsReport
    delta: pd.DataFrame
    times: pd.DataFrame
    summary: pd.DataFrame
    corners: pd.DataFrame
    lap_summary: pd.DataFrame
    elapsed_s: float


def _sector_boundaries(meta: dict) -> tuple[float, float] | None:
    """S1 and S2 positions from F008's ``alignment_meta.json``, if present."""
    medians = {row.get("boundary"): row.get("median_m") for row in meta.get("sector_consistency", [])}
    if "S1" in medians and "S2" in medians:
        return float(medians["S1"]), float(medians["S2"])
    return None


def _lap_attributes(laps: pd.DataFrame, index: pd.MultiIndex) -> pd.DataFrame:
    frame = laps.copy()
    frame["driver"] = frame["driver"].astype(str)
    frame["lap_number"] = frame["lap_number"].astype(int)
    frame = frame.set_index(["driver", "lap_number"]).reindex(index)
    out = pd.DataFrame(index=index)
    out["lap_time_s"] = frame["lap_time"].astype(float)
    for column in ("compound", "stint", "tyre_life", "team"):
        out[column] = frame[column] if column in frame.columns else pd.NA
    return out


def compute_metrics(
    grid_root: Path,
    microsector_root: Path,
    snapshot_root: Path,
    aligned_root: Path,
    out_root: Path | None = None,
    reference: ReferenceSpec = ReferenceSpec(),
    grid_m: float = GRID_SPACING_M,
) -> MetricsResult:
    """Compute every F004 table for one session and write them to ``data/processed``.

    Idempotent: identical inputs and reference give identical tables.
    """
    started = time.perf_counter()
    grid = pd.read_parquet(grid_root / "grid.parquet")
    microsectors = pd.read_parquet(microsector_root / "microsectors.parquet")
    events = pd.read_parquet(microsector_root / "events.parquet")
    laps = pd.read_parquet(snapshot_root / "laps.parquet")
    meta = json.loads((aligned_root / "alignment_meta.json").read_text(encoding="utf-8"))
    seg_meta_path = microsector_root / "segmentation_meta.json"
    lap_length_m = (
        float(json.loads(seg_meta_path.read_text(encoding="utf-8"))["lap_length_m"])
        if seg_meta_path.exists()
        else float(microsectors["end_m"].max())
    )

    curves = time_curves(grid)
    ref = resolve_reference(laps, curves.index, reference)
    ref_label = reference.label if reference.kind != "session_fastest" else lap_label(ref.iloc[0])
    logger.info("reference kind=%s label=%s laps=%d", reference.kind, ref_label, len(curves))

    delta = delta_t(curves, ref, reference.kind)
    times = microsector_times(curves, microsectors, ref)
    summary_key = (
        ref.iloc[0]
        if reference.kind != "driver_best"
        else resolve_reference(laps, curves.index, ReferenceSpec("session_fastest")).iloc[0]
    )
    summary = summarise_microsectors(times, microsectors, summary_key)
    corners = corner_metrics(grid, events, lap_length_m, grid_m)

    endpoints = validation.endpoint_residuals(curves, laps, ref)
    boundaries = _sector_boundaries(meta)
    if boundaries is not None:
        marshalling = validation.marshalling_sectors(curves, laps, *boundaries, grid_m=grid_m)
    else:
        marshalling = pd.DataFrame(
            np.nan,
            index=curves.index,
            columns=["s1_grid_s", "s2_grid_s", "s3_grid_s", "s1_official_s", "s2_official_s", "s3_official_s"],
        )
        logger.warning("alignment_meta.json has no sector boundaries; S1/S2/S3 checks skipped")
    lap_summary = pd.concat([_lap_attributes(laps, curves.index), endpoints, marshalling], axis=1)
    lap_summary["reference"] = [lap_label(k) for k in ref.reindex(curves.index)]
    lap_summary = lap_summary.reset_index()[list(LAP_SUMMARY_SCHEMA)].astype(LAP_SUMMARY_SCHEMA)

    report = validation.measure(
        delta, times, summary, corners, microsectors, lap_summary, ref, events, reference.kind, ref_label
    )

    out_root = out_root or (PROCESSED_ROOT / grid_root.name)
    out_root.mkdir(parents=True, exist_ok=True)
    delta.to_parquet(out_root / "delta_t.parquet", index=False)
    times.to_parquet(out_root / "microsector_times.parquet", index=False)
    summary.to_parquet(out_root / "microsector_summary.parquet", index=False)
    corners.to_parquet(out_root / "corner_metrics.parquet", index=False)
    lap_summary.to_parquet(out_root / "lap_summary.parquet", index=False)

    elapsed = time.perf_counter() - started
    (out_root / "metrics_meta.json").write_text(
        json.dumps(
            {
                "grid": str(grid_root),
                "microsectors": str(microsector_root),
                "snapshot": str(snapshot_root),
                "aligned": str(aligned_root),
                "reference": {**asdict(reference), "label": ref_label},
                "lap_length_m": lap_length_m,
                "sector_boundaries_m": list(boundaries) if boundaries else None,
                "rows": {
                    "delta_t": int(len(delta)),
                    "microsector_times": int(len(times)),
                    "microsector_summary": int(len(summary)),
                    "corner_metrics": int(len(corners)),
                    "lap_summary": int(len(lap_summary)),
                },
                "elapsed_s": elapsed,
                "acceptance": report.to_dict(),
                "limitation": (
                    "Curves are re-zeroed at grid 0, not at the telemetry window. Sector times for "
                    "sectors shorter than ~3 bins are individually noisy (filter on length_m); their "
                    "deltas still sum correctly. brake_gap_m is the D6 uncertainty of each braking point. "
                    "S1/S3 offsets against the official timing line are reported for F010, not gated."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = MetricsResult(
        root=out_root,
        reference=ref,
        report=report,
        delta=delta,
        times=times,
        summary=summary,
        corners=corners,
        lap_summary=lap_summary,
        elapsed_s=elapsed,
    )
    logger.info(
        "metrics_complete laps=%d reference=%s delta_rows=%d closure_err_s=%.2e s2_median_s=%+.3f "
        "s2_std_s=%.3f endpoint_p50_s=%.3f endpoint_p95_s=%.3f brake_coverage=%.3f acceptance_ok=%s elapsed_s=%.2f",
        report.laps,
        ref_label,
        len(delta),
        report.closure_max_err_s,
        report.s2_median_s,
        report.s2_std_s,
        report.endpoint.p50,
        report.endpoint.p95,
        report.brake_coverage,
        report.ok,
        elapsed,
    )
    return result
