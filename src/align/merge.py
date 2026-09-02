"""Time-domain channel merge and raw distance.

F002 stores car telemetry and position data as separate artefacts on different
sample timebases. Alignment needs speed and X/Y together, so this puts them on
one timebase first.

This is a *time*-domain merge. The *distance*-domain resampling onto a uniform
grid is F003 and is a different operation on this module's output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.align.units import positions_to_metres

#: Linearly interpolated between samples.
CONTINUOUS_CAR = ("speed", "throttle", "rpm")
CONTINUOUS_POS = ("x", "y", "z")

#: Step/previous only. Project convention: never linearly interpolated, because
#: a gear of 6.4 or a half-pressed boolean brake is not a physical state.
DISCRETE_CAR = ("n_gear", "brake", "drs")


class MergeError(ValueError):
    """Channels cannot be placed on a common timebase."""


def _step_previous(target_t: np.ndarray, source_t: np.ndarray, values: pd.Series) -> pd.Series:
    """Value of the most recent sample at or before each target time."""
    idx = np.searchsorted(source_t, target_t, side="right") - 1
    idx = np.clip(idx, 0, len(source_t) - 1)
    return values.iloc[idx].reset_index(drop=True)


def merge_lap_channels(car: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    """Put one lap's car and position channels on a common timebase.

    The output timebase is the union of both inputs, clipped to their overlap
    so that X/Y is never extrapolated beyond where position data exists.
    """
    if len(car) < 2 or len(pos) < 2:
        raise MergeError(f"need >=2 samples per channel set (car={len(car)}, pos={len(pos)})")

    car = car.sort_values("session_time").reset_index(drop=True)
    pos = pos.sort_values("session_time").reset_index(drop=True)

    car_t = car["session_time"].to_numpy(dtype=float)
    pos_t = pos["session_time"].to_numpy(dtype=float)

    lo = max(car_t[0], pos_t[0])
    hi = min(car_t[-1], pos_t[-1])
    if not hi > lo:
        raise MergeError("car and position data do not overlap in time")

    grid = np.union1d(car_t, pos_t)
    grid = grid[(grid >= lo) & (grid <= hi)]
    if len(grid) < 2:
        raise MergeError("overlap window contains fewer than 2 samples")

    out = pd.DataFrame({"session_time": grid})

    for column in CONTINUOUS_CAR:
        out[column] = np.interp(grid, car_t, car[column].to_numpy(dtype=float))
    for column in CONTINUOUS_POS:
        out[column] = np.interp(grid, pos_t, pos[column].to_numpy(dtype=float))
    for column in DISCRETE_CAR:
        out[column] = _step_previous(grid, car_t, car[column])

    for column in ("driver", "lap_number"):
        out[column] = car[column].iloc[0]

    ordered = ["driver", "lap_number", "session_time", *CONTINUOUS_CAR, *DISCRETE_CAR, *CONTINUOUS_POS]
    # X/Y/Z arrive in units of 1/10 m; everything downstream works in metres.
    return positions_to_metres(out[ordered])


def add_raw_distance(frame: pd.DataFrame) -> pd.DataFrame:
    """Append `distance_raw`: cumulative trapezoidal integration of speed.

    Speed arrives in km/h; distance comes out in metres. This is the lap's own
    distance axis — it carries the line the driver took and the speed sensor's
    error, which is precisely why it cannot be compared across drivers without
    the alignment this module's caller performs.
    """
    out = frame.copy()
    speed_ms = out["speed"].to_numpy(dtype=float) / 3.6
    time_s = out["session_time"].to_numpy(dtype=float)

    if len(out) < 2:
        out["distance_raw"] = np.zeros(len(out))
        return out

    dt = np.diff(time_s)
    if (dt < 0).any():
        raise MergeError("session_time is not monotonically increasing")

    segment = 0.5 * (speed_ms[1:] + speed_ms[:-1]) * dt
    out["distance_raw"] = np.concatenate([[0.0], np.cumsum(segment)])
    return out


def build_lap_frame(car: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    """Merge one lap's channels and give it a raw distance axis."""
    return add_raw_distance(merge_lap_channels(car, pos))
