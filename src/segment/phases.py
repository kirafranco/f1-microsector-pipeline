"""Micro-sector tables: corner phases plus straights, and fixed bins.

Both grains partition ``[0, lap_length)`` exactly -- no gaps, no overlaps,
every sector at least one grid bin -- so that per-sector times on any lap sum
to that lap's total by construction.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.grid.resample import GRID_SPACING_M

GRAIN_CORNER_PHASE = "corner_phase"
GRAIN_FIXED_100M = "fixed_100m"
PHASES = ("braking", "entry", "apex", "exit")

MICROSECTOR_SCHEMA: dict[str, str] = {
    "grain": "string",
    "microsector_id": "Int16",
    "phase": "string",
    "event_id": "Int16",
    "start_m": "float32",
    "end_m": "float32",
    "start_index": "Int32",
    "end_index": "Int32",
    "corners": "string",
    "marginal": "boolean",
}


class SegmentationError(ValueError):
    """The events do not yield a valid partition of the lap."""


def _unwrap(value: float, anchor: float, lap_length: float) -> float:
    """Bring a modulo-L boundary onto the same cycle as ``anchor``."""
    if not np.isfinite(value):
        return value
    while value - anchor > lap_length / 2:
        value -= lap_length
    while anchor - value > lap_length / 2:
        value += lap_length
    return value


def event_intervals(event: pd.Series, lap_length: float) -> list[tuple[str, float, float]]:
    """Non-empty phase intervals of one event in unwrapped metres, in lap order."""
    apex = float(event["apex_m"])
    b = {
        key: _unwrap(float(event[key]), apex, lap_length)
        for key in ("lift_m", "brake_on_m", "brake_off_m", "apex_start_m", "apex_end_m", "exit_end_m")
    }
    a0, a1 = b["apex_start_m"], b["apex_end_m"]
    out: list[tuple[str, float, float]] = []
    if bool(event["has_braking"]):
        out.append(("braking", b["brake_on_m"], min(b["brake_off_m"], a0)))
        out.append(("entry", b["brake_off_m"], a0))
    else:
        out.append(("entry", b["lift_m"], a0))
    out.append(("apex", a0, a1))
    out.append(("exit", a1, b["exit_end_m"]))
    return [(phase, s, e) for phase, s, e in out if np.isfinite(s) and np.isfinite(e) and e > s]


def _split_at_lap(start: float, end: float, lap_length: float) -> list[tuple[float, float]]:
    """Cut an unwrapped interval at the start/finish line."""
    if start < 0:
        pieces = [(start + lap_length, lap_length)]
        if end > 0:
            pieces.append((0.0, end))
        return pieces
    if end > lap_length:
        return [(start, lap_length), (0.0, end - lap_length)]
    return [(start, end)]


def build_corner_phases(
    events: pd.DataFrame, lap_length_m: float, grid_m: float = GRID_SPACING_M
) -> pd.DataFrame:
    """Corner-phase grain: event phases with straights filling the rest."""
    rows: list[dict] = []
    for _, event in events.iterrows():
        for phase, s, e in event_intervals(event, lap_length_m):
            for ps, pe in _split_at_lap(s, e, lap_length_m):
                if pe > ps:
                    rows.append(
                        dict(phase=phase, event_id=int(event["event_id"]), start_m=ps, end_m=pe,
                             corners=event["corners"], marginal=bool(event["marginal"]))
                    )
    rows.sort(key=lambda r: r["start_m"])

    for a, b in zip(rows, rows[1:]):
        if b["start_m"] < a["end_m"] - 1e-6:
            raise SegmentationError(
                f"phases overlap: event {a['event_id']} {a['phase']} ends {a['end_m']:.0f} m, "
                f"event {b['event_id']} {b['phase']} starts {b['start_m']:.0f} m"
            )

    filled: list[dict] = []
    cursor = 0.0
    for r in rows:
        if r["start_m"] > cursor + 1e-6:
            filled.append(dict(phase="straight", event_id=pd.NA, start_m=cursor, end_m=r["start_m"],
                               corners=pd.NA, marginal=False))
        filled.append(r)
        cursor = r["end_m"]
    if cursor < lap_length_m - 1e-6:
        filled.append(dict(phase="straight", event_id=pd.NA, start_m=cursor, end_m=lap_length_m,
                           corners=pd.NA, marginal=False))

    return _finish(filled, GRAIN_CORNER_PHASE, grid_m)


def build_fixed_bins(
    lap_length_m: float, bin_m: float = 100.0, grid_m: float = GRID_SPACING_M
) -> pd.DataFrame:
    """Fixed grain: ``ceil(L / bin_m)`` bins from zero, the last one truncated."""
    if bin_m <= 0 or lap_length_m <= 0:
        raise SegmentationError("bin size and lap length must be positive")
    count = int(math.ceil(lap_length_m / bin_m - 1e-9))
    rows = [
        dict(phase="bin", event_id=pd.NA, start_m=i * bin_m, end_m=min((i + 1) * bin_m, lap_length_m),
             corners=pd.NA, marginal=False)
        for i in range(count)
    ]
    return _finish(rows, GRAIN_FIXED_100M, grid_m)


def _finish(rows: list[dict], grain: str, grid_m: float) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["phase", "event_id", "start_m", "end_m", "corners", "marginal"])
    frame.insert(0, "microsector_id", np.arange(len(frame), dtype=np.int16))
    frame.insert(0, "grain", grain)
    frame["start_index"] = np.rint(frame["start_m"].to_numpy(dtype=float) / grid_m).astype(np.int32)
    frame["end_index"] = np.rint(frame["end_m"].to_numpy(dtype=float) / grid_m).astype(np.int32)
    frame["event_id"] = frame["event_id"].astype("Int16")
    frame["corners"] = frame["corners"].astype("string")
    return frame[list(MICROSECTOR_SCHEMA)].astype(MICROSECTOR_SCHEMA)


def check_partition(sectors: pd.DataFrame, lap_length_m: float, grid_m: float = GRID_SPACING_M) -> bool:
    """Exact cover of ``[0, L)``: contiguous, ordered, every sector >= one bin."""
    if sectors.empty:
        return False
    start = sectors["start_m"].to_numpy(dtype=float)
    end = sectors["end_m"].to_numpy(dtype=float)
    if not np.isclose(start[0], 0.0) or not np.isclose(end[-1], lap_length_m):
        return False
    if not np.allclose(start[1:], end[:-1]):
        return False
    return bool((end - start >= grid_m - 1e-6).all())
