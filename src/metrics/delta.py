"""Time curves re-zeroed at the line, and delta-t against a reference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics.reference import lap_index, lap_label

DELTA_SCHEMA: dict[str, str] = {
    "driver": "string",
    "lap_number": "Int16",
    "grid_index": "Int32",
    "t_s": "float32",
    "delta_t_s": "float32",
    "reference": "string",
    "reference_kind": "string",
}


class DeltaError(ValueError):
    """The grid cannot be turned into time curves."""


def time_curves(grid: pd.DataFrame) -> pd.DataFrame:
    """Wide ``(driver, lap_number) x grid_index`` table of ``t_s``.

    ``t_s = elapsed_time - elapsed_time[grid 0]``: time since the lap reached
    distance 0 on the shared axis, not since its telemetry window opened.
    NaN beyond each lap's last grid point.
    """
    for column in ("driver", "lap_number", "grid_index", "elapsed_time"):
        if column not in grid.columns:
            raise DeltaError(f"grid is missing {column!r}")
    if grid.empty:
        raise DeltaError("grid is empty")

    wide = grid.pivot_table(
        index=["driver", "lap_number"], columns="grid_index", values="elapsed_time", aggfunc="first"
    ).astype(float)
    wide.index = lap_index(wide.index)
    wide.columns = wide.columns.astype(int)
    # A grid index nobody has would otherwise vanish from the pivot and hide a gap.
    wide = wide.reindex(columns=range(int(wide.columns.max()) + 1))
    if 0 not in wide.columns or wide[0].isna().any():
        missing = wide.index[wide[0].isna()].tolist() if 0 in wide.columns else wide.index.tolist()
        raise DeltaError(f"laps without grid index 0: {missing[:5]}")

    values = wide.to_numpy()
    # Each lap must be contiguous from 0 -- F003 guarantees it; check anyway.
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    last = np.where(finite.any(axis=1), finite.shape[1] - 1 - np.argmax(finite[:, ::-1], axis=1), -1)
    if not np.array_equal(counts, last + 1):
        bad = wide.index[counts != last + 1].tolist()
        raise DeltaError(f"laps with gaps in their grid: {bad[:5]}")

    values = values - values[:, [0]]
    return pd.DataFrame(values, index=wide.index, columns=wide.columns)


def lap_lengths(curves: pd.DataFrame) -> pd.Series:
    """Number of grid points per lap."""
    return pd.Series(np.isfinite(curves.to_numpy()).sum(axis=1), index=curves.index, name="n_points")


def reference_curves(curves: pd.DataFrame, reference: pd.Series) -> np.ndarray:
    """Row *i* is the time curve of lap *i*'s reference lap."""
    ref_index = lap_index(reference.reindex(curves.index).tolist())
    missing = [k for k in set(ref_index) if k not in curves.index]
    if missing:
        raise DeltaError(f"reference lap(s) not on the grid: {missing[:5]}")
    return curves.reindex(ref_index).to_numpy()


def delta_t(curves: pd.DataFrame, reference: pd.Series, kind: str) -> pd.DataFrame:
    """Long delta-t table typed per :data:`DELTA_SCHEMA`.

    One row per grid point each lap actually has; ``delta_t_s`` is NaN where
    the reference lap is shorter than the lap.
    """
    values = curves.to_numpy()
    delta = values - reference_curves(curves, reference)
    n_laps, n_idx = values.shape
    labels = np.array([lap_label(k) for k in reference.reindex(curves.index)], dtype=object)

    drivers = np.repeat(curves.index.get_level_values("driver").to_numpy(), n_idx)
    lap_numbers = np.repeat(curves.index.get_level_values("lap_number").to_numpy(), n_idx)
    long = pd.DataFrame(
        {
            "driver": drivers,
            "lap_number": lap_numbers,
            "grid_index": np.tile(curves.columns.to_numpy(), n_laps),
            "t_s": values.ravel(),
            "delta_t_s": delta.ravel(),
            "reference": np.repeat(labels, n_idx),
            "reference_kind": kind,
        }
    )
    long = long[np.isfinite(long["t_s"].to_numpy())].reset_index(drop=True)
    return long.astype(DELTA_SCHEMA)
