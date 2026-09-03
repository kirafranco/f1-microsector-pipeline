"""Where the official start/finish line falls on the aligned distance axis.

Each lap carries its own answer: FastF1 gives the session time at which the lap
started and how long it took, and the aligned telemetry gives the speed
integral. Interpolating the raw distance at those two instants and carrying it
onto the aligned axis from a sample safely inside the reference line puts the
timing line on the same coordinate as everything else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: A sample must be at least this far inside the reference line to anchor the
#: transfer, so that a clamped or near-vertex sample never becomes the anchor.
ANCHOR_MARGIN_M = 20.0

CROSSING_SCHEMA: dict[str, str] = {
    "driver": "string",
    "lap_number": "Int16",
    "line_start_m": "float32",
    "line_end_m": "float32",
    "window_open_s": "float32",
    "window_close_s": "float32",
    "driven_m": "float32",
    "start_extrapolated": "boolean",
    "end_extrapolated": "boolean",
}


class TimingLineError(ValueError):
    """The timing line cannot be located on the axis."""


def _raw_at(time_s: np.ndarray, raw_m: np.ndarray, speed_ms: np.ndarray, when: float) -> tuple[float, bool]:
    """Raw distance at ``when``; constant-speed extrapolation outside the window."""
    if when < time_s[0]:
        return float(raw_m[0] - speed_ms[0] * (time_s[0] - when)), True
    if when > time_s[-1]:
        return float(raw_m[-1] + speed_ms[-1] * (when - time_s[-1])), True
    return float(np.interp(when, time_s, raw_m)), False


def line_crossings(
    aligned: pd.DataFrame, laps: pd.DataFrame, line_length_m: float, margin_m: float = ANCHOR_MARGIN_M
) -> pd.DataFrame:
    """Per-lap position of both timing-line crossings on the aligned axis."""
    required = {"driver", "lap_number", "session_time", "speed", "distance_raw", "distance_aligned"}
    missing = required - set(aligned.columns)
    if missing:
        raise TimingLineError(f"aligned telemetry is missing {sorted(missing)}")

    official = laps.copy()
    official["driver"] = official["driver"].astype(str)
    official["lap_number"] = official["lap_number"].astype(int)
    official = official.set_index(["driver", "lap_number"])

    rows: list[dict] = []
    for (driver, lap_number), lap in aligned.groupby(["driver", "lap_number"], observed=True):
        key = (str(driver), int(lap_number))
        if key not in official.index:
            continue
        record = official.loc[key]
        start, duration = record.get("lap_start_time"), record.get("lap_time")
        if pd.isna(start) or pd.isna(duration):
            continue

        lap = lap.sort_values("session_time")
        time_s = lap["session_time"].to_numpy(dtype=float)
        raw_m = lap["distance_raw"].to_numpy(dtype=float)
        aligned_m = lap["distance_aligned"].to_numpy(dtype=float)
        speed_ms = lap["speed"].to_numpy(dtype=float) / 3.6
        if len(time_s) < 2:
            continue

        inside = (aligned_m >= margin_m) & (aligned_m <= line_length_m - margin_m)
        if not inside.any():
            continue
        first, last = int(np.argmax(inside)), int(len(inside) - 1 - np.argmax(inside[::-1]))

        t_start, t_end = float(start), float(start) + float(duration)
        raw_start, start_extrapolated = _raw_at(time_s, raw_m, speed_ms, t_start)
        raw_end, end_extrapolated = _raw_at(time_s, raw_m, speed_ms, t_end)
        rows.append(
            {
                "driver": key[0],
                "lap_number": key[1],
                "line_start_m": aligned_m[first] - (raw_m[first] - raw_start),
                "line_end_m": aligned_m[last] + (raw_end - raw_m[last]),
                "window_open_s": time_s[0] - t_start,
                "window_close_s": time_s[-1] - t_end,
                "driven_m": raw_end - raw_start,
                "start_extrapolated": start_extrapolated,
                "end_extrapolated": end_extrapolated,
            }
        )

    if not rows:
        raise TimingLineError("no lap could be located against the timing line")
    return pd.DataFrame(rows)[list(CROSSING_SCHEMA)].astype(CROSSING_SCHEMA)


def session_line_positions(crossings: pd.DataFrame) -> tuple[float, float]:
    """Session-level line positions: the median of the per-lap estimates."""
    if crossings.empty:
        raise TimingLineError("no crossings to summarise")
    return (
        float(crossings["line_start_m"].median()),
        float(crossings["line_end_m"].median()),
    )
