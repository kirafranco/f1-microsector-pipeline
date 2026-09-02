"""Per-lap resampling onto a uniform distance grid.

Pure NumPy, one lap in, one lap out, no I/O and no JVM (D2). F013 later wraps
exactly this function in a Spark grouped-map UDF without touching the maths.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

#: Grid spacing decided in D6. Deliberately oversamples the ~4 Hz source; the
#: obligation that comes with it is the resolution note in the package docstring.
GRID_SPACING_M = 10.0

DISTANCE_COLUMN = "distance_aligned"
TIME_COLUMN = "session_time"
KEY_COLUMNS = ("driver", "lap_number")

#: Linearly interpolated.
CONTINUOUS_CHANNELS = ("speed", "throttle", "rpm", "x", "y")

#: Step/previous only. Project convention: a gear of 6.4 or a half-pressed
#: boolean brake is not a physical state, so these are never interpolated.
DISCRETE_CHANNELS = ("n_gear", "brake", "drs")

REQUIRED_COLUMNS = (
    *KEY_COLUMNS,
    TIME_COLUMN,
    DISTANCE_COLUMN,
    *CONTINUOUS_CHANNELS,
    *DISCRETE_CHANNELS,
)

#: Output contract, in column order. Continuous channels are float4, gear/DRS
#: smallint-compatible, brake boolean -- matching the fact table types.
GRID_SCHEMA: dict[str, str] = {
    "driver": "string",
    "lap_number": "Int16",
    "grid_index": "Int32",
    "distance_m": "float32",
    "elapsed_time": "float32",
    "speed": "float32",
    "throttle": "float32",
    "rpm": "float32",
    "x": "float32",
    "y": "float32",
    "n_gear": "Int8",
    "drs": "Int8",
    "brake": "boolean",
    "source_gap_m": "float32",
}


class ResampleError(ValueError):
    """The lap cannot be placed on a distance grid."""


def grid_point_count(lap_length_m: float, grid_m: float = GRID_SPACING_M) -> int:
    """Exactly ``floor(L / grid_m) + 1`` points, the last one at or before L."""
    if grid_m <= 0:
        raise ResampleError(f"grid spacing must be positive, got {grid_m}")
    if not math.isfinite(lap_length_m) or lap_length_m < 0:
        raise ResampleError(f"lap length must be finite and non-negative, got {lap_length_m}")
    return int(math.floor(lap_length_m / grid_m)) + 1


def make_grid(lap_length_m: float, grid_m: float = GRID_SPACING_M) -> np.ndarray:
    """Uniform grid ``0, grid_m, 2*grid_m, ...`` up to the lap length.

    Built as ``arange(n) * grid_m`` rather than ``arange(0, L, grid_m)`` so the
    point count never depends on floating-point behaviour at the upper bound.
    """
    return np.arange(grid_point_count(lap_length_m, grid_m), dtype=float) * grid_m


def _last_of_each_distance(distance: np.ndarray) -> np.ndarray:
    """Indices keeping the last sample of every run of equal distance.

    F008's monotonic guard can leave consecutive samples at the same distance.
    ``np.interp`` needs strictly increasing abscissae, and the step channels are
    defined as "the most recent state at or before this point", so the last
    sample of a run is the one that represents the car leaving that point.
    """
    keep = np.empty(len(distance), dtype=bool)
    keep[:-1] = distance[1:] != distance[:-1]
    keep[-1] = True
    return np.flatnonzero(keep)


def _step_previous(grid: np.ndarray, distance: np.ndarray, values: pd.Series) -> pd.Series:
    """Most recent source value at or before each grid point.

    Grid points before the first sample take the first sample, mirroring the
    constant extrapolation ``np.interp`` applies to continuous channels.
    """
    idx = np.searchsorted(distance, grid, side="right") - 1
    idx = np.clip(idx, 0, len(distance) - 1)
    return values.iloc[idx].reset_index(drop=True)


def _source_gap(grid: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Spacing between the two source samples bracketing each grid point.

    Zero means a source sample sits exactly on the grid point. NaN means the
    grid point lies outside the sampled range and its value is held from the
    nearest sample rather than interpolated.
    """
    lo = np.searchsorted(distance, grid, side="right") - 1
    hi = np.searchsorted(distance, grid, side="left")
    inside = (lo >= 0) & (hi < len(distance))
    gap = np.full(len(grid), np.nan)
    gap[inside] = distance[hi[inside]] - distance[lo[inside]]
    return gap


def _validate(lap: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in lap.columns]
    if missing:
        raise ResampleError(f"lap is missing columns {missing}")
    if len(lap) < 2:
        raise ResampleError(f"need at least 2 samples, got {len(lap)}")
    for key in KEY_COLUMNS:
        if lap[key].nunique(dropna=False) != 1:
            raise ResampleError(f"lap frame spans more than one {key}")

    distance = lap[DISTANCE_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(distance).all():
        raise ResampleError(f"{DISTANCE_COLUMN} contains non-finite values")
    if (np.diff(distance) < 0).any():
        raise ResampleError(f"{DISTANCE_COLUMN} is not monotonically non-decreasing")
    time_s = lap[TIME_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(time_s).all() or (np.diff(time_s) < 0).any():
        raise ResampleError(f"{TIME_COLUMN} is not finite and monotonically non-decreasing")


def resample_lap(lap: pd.DataFrame, grid_m: float = GRID_SPACING_M) -> pd.DataFrame:
    """Resample one aligned lap onto a uniform distance grid.

    Parameters
    ----------
    lap:
        One driver-lap of F008 output, ordered in time, carrying
        ``distance_aligned`` plus the continuous and discrete channels.
    grid_m:
        Grid spacing in metres.

    Returns
    -------
    One row per grid point, typed per :data:`GRID_SCHEMA`. The input frame is
    not modified and the result depends only on the inputs.
    """
    _validate(lap)

    distance_all = lap[DISTANCE_COLUMN].to_numpy(dtype=float)
    keep = _last_of_each_distance(distance_all)
    if len(keep) < 2:
        raise ResampleError("lap covers a single distance value; nothing to resample")

    distance = distance_all[keep]
    grid = make_grid(float(distance[-1]), grid_m)

    # Time origin is the lap's very first sample, before de-duplication, so
    # elapsed_time is measured from where the telemetry window opens.
    time_all = lap[TIME_COLUMN].to_numpy(dtype=float)
    elapsed = (time_all - time_all[0])[keep]

    out: dict[str, object] = {
        "driver": np.repeat(lap["driver"].iloc[0], len(grid)),
        "lap_number": np.repeat(lap["lap_number"].iloc[0], len(grid)),
        "grid_index": np.arange(len(grid), dtype=np.int32),
        "distance_m": grid,
        "elapsed_time": np.interp(grid, distance, elapsed),
    }
    for channel in CONTINUOUS_CHANNELS:
        values = lap[channel].to_numpy(dtype=float)[keep]
        out[channel] = np.interp(grid, distance, values)
    for channel in DISCRETE_CHANNELS:
        values = lap[channel].iloc[keep].reset_index(drop=True)
        out[channel] = _step_previous(grid, distance, values)
    out["source_gap_m"] = _source_gap(grid, distance)

    frame = pd.DataFrame(out)[list(GRID_SCHEMA)]
    return frame.astype(GRID_SCHEMA)
