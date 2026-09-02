"""Session-level resampling: aligned interim layer in, grid interim layer out."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import INTERIM_ROOT
from src.grid.resample import GRID_SCHEMA, GRID_SPACING_M, KEY_COLUMNS, ResampleError, resample_lap
from src.grid.validation import GridReport, measure_session

logger = logging.getLogger(__name__)

#: One row per (driver, lap, grid point). Validated before writing.
BUSINESS_KEY = (*KEY_COLUMNS, "grid_index")


@dataclass(frozen=True)
class ResampleResult:
    root: Path
    grid_m: float
    laps_total: int
    laps_resampled: int
    laps_rejected: int
    rows: int
    report: GridReport
    elapsed_s: float

    @property
    def resampled_fraction(self) -> float:
        return self.laps_resampled / self.laps_total if self.laps_total else 0.0


def resample_session(
    aligned_root: Path,
    out_root: Path | None = None,
    grid_m: float = GRID_SPACING_M,
) -> ResampleResult:
    """Resample every aligned lap in a session and write ``grid.parquet``.

    A lap that cannot be resampled is recorded in ``rejected_laps.parquet``
    with its reason and the batch continues. The write is idempotent: the same
    input produces byte-identical output, and re-running replaces the files.
    """
    started = time.perf_counter()
    telemetry = pd.read_parquet(aligned_root / "telemetry_aligned.parquet")

    grids: list[pd.DataFrame] = []
    pairs: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    rejected: list[dict] = []
    laps_total = 0

    for (driver, lap_number), lap in telemetry.groupby(list(KEY_COLUMNS), observed=True, sort=True):
        laps_total += 1
        lap = lap.sort_values("session_time").reset_index(drop=True)
        lap_started = time.perf_counter()
        try:
            grid = resample_lap(lap, grid_m=grid_m)
        except ResampleError as exc:
            rejected.append({"driver": driver, "lap_number": lap_number, "reason": str(exc)})
            logger.warning("lap_rejected driver=%s lap=%s reason=%s", driver, lap_number, exc)
            continue
        grids.append(grid)
        pairs.append((lap, grid))
        logger.debug(
            "lap_resampled driver=%s lap=%s source_rows=%d grid_rows=%d ms=%.1f",
            driver,
            lap_number,
            len(lap),
            len(grid),
            1000.0 * (time.perf_counter() - lap_started),
        )

    if not grids:
        raise RuntimeError(f"{aligned_root}: no lap resampled ({len(rejected)} rejected)")

    output = pd.concat(grids, ignore_index=True).astype(GRID_SCHEMA)
    duplicates = int(output.duplicated(list(BUSINESS_KEY)).sum())
    if duplicates:
        raise RuntimeError(f"{aligned_root}: {duplicates} duplicate {BUSINESS_KEY} rows")

    report = measure_session(pairs, grid_m)
    rejected_out = pd.DataFrame(rejected, columns=["driver", "lap_number", "reason"])

    out_root = out_root or (INTERIM_ROOT / "grid" / aligned_root.name)
    out_root.mkdir(parents=True, exist_ok=True)
    output.to_parquet(out_root / "grid.parquet", index=False)
    rejected_out.to_parquet(out_root / "rejected_laps.parquet", index=False)

    elapsed = time.perf_counter() - started
    meta = {
        "source": str(aligned_root),
        "grid_m": grid_m,
        "laps_total": laps_total,
        "laps_resampled": len(grids),
        "laps_rejected": len(rejected),
        "rows": int(len(output)),
        "elapsed_s": elapsed,
        "acceptance": report.to_dict(),
        "limitation": (
            "Source sampling is ~4 Hz; empty_bin_fraction of grid bins hold no source "
            "sample and are pure interpolation. source_gap_m carries the bracketing "
            "source spacing per grid point (NaN = outside the sampled range, value held)."
        ),
    }
    (out_root / "grid_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    result = ResampleResult(
        root=out_root,
        grid_m=grid_m,
        laps_total=laps_total,
        laps_resampled=len(grids),
        laps_rejected=len(rejected),
        rows=int(len(output)),
        report=report,
        elapsed_s=elapsed,
    )
    logger.info(
        "resample_complete laps=%d/%d rejected=%d rows=%d grid_m=%.1f "
        "speed_p95=%.2f throttle_p95=%.2f rpm_p95=%.0f brake_edge_p95_m=%.2f "
        "empty_bins=%.3f acceptance_ok=%s elapsed_s=%.2f",
        result.laps_resampled,
        result.laps_total,
        result.laps_rejected,
        result.rows,
        grid_m,
        report.speed.p95,
        report.throttle.p95,
        report.rpm.p95,
        report.brake_edge.p95,
        report.empty_bin_fraction,
        report.ok,
        elapsed,
    )
    return result
