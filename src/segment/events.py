"""Braking-and-cornering events from the session-median speed trace.

An event is a trough of median speed with prominence at or above a threshold,
found on the circular lap. Every boundary is then read off the median traces
in index space and stored in metres modulo the lap length, so an event that
straddles the start/finish line is still one event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.grid.resample import GRID_SPACING_M
from src.segment.corners import corner_label

EVENT_SCHEMA: dict[str, str] = {
    "event_id": "Int16",
    "apex_m": "float32",
    "v_min_kmh": "float32",
    "prominence_kmh": "float32",
    "left_max_m": "float32",
    "lift_m": "float32",
    "brake_on_m": "float32",
    "brake_off_m": "float32",
    "apex_start_m": "float32",
    "apex_end_m": "float32",
    "exit_end_m": "float32",
    "has_braking": "boolean",
    "corners": "string",
    "marginal": "boolean",
}


class EventError(ValueError):
    """The median traces cannot be segmented."""


@dataclass(frozen=True)
class EventParams:
    """Every tunable, with the defaults measured on Suzuka 2024 Q."""

    #: Minimum speed loss for a trough to count as an event.
    min_prominence_kmh: float = 8.0
    #: Events below ``marginal_factor * min_prominence_kmh`` are flagged.
    marginal_factor: float = 2.0
    #: Apex window: speed within this fraction of the prominence above V_min.
    apex_fraction: float = 0.05
    #: Exit ends where this fraction of the lost speed has been regained.
    exit_recovery_fraction: float = 0.5
    #: Share of laps braking for a grid point to count as brake-on.
    brake_on_fraction: float = 0.5
    #: Median throttle below this marks the lift.
    throttle_lift_pct: float = 90.0
    #: Corners within this margin of an event's extent are its labels.
    label_margin_m: float = 30.0
    #: Fallback reach behind the apex band for an event the margin above leaves
    #: unlabelled -- a flat-out corner, whose marker sits at the turn-in ahead
    #: of the band (F018). Only markers no other event claimed are taken.
    fallback_margin_m: float = 60.0


@dataclass(frozen=True)
class Trough:
    index: int
    speed: float
    prominence: float


def median_traces(grid: pd.DataFrame, min_lap_fraction: float = 0.8) -> pd.DataFrame:
    """Per-grid-index medians over every lap in the grid.

    Indices present in fewer than ``min_lap_fraction`` of laps are dropped and
    the result is cut at the first missing index so the trace is contiguous
    from zero -- the last one or two points of the longest laps go.
    """
    n_laps = grid.groupby(["driver", "lap_number"], observed=True).ngroups
    if n_laps == 0:
        raise EventError("grid contains no laps")
    frame = grid.assign(brake_f=grid["brake"].astype("Float64").astype(float))
    agg = frame.groupby("grid_index").agg(
        speed=("speed", "median"),
        brake=("brake_f", "mean"),
        throttle=("throttle", "median"),
        n_laps=("speed", "size"),
    )
    keep = agg[agg["n_laps"] >= max(1, int(np.ceil(min_lap_fraction * n_laps)))]
    index = keep.index.to_numpy(dtype=np.int64)
    if len(index) == 0 or index[0] != 0:
        raise EventError("median trace does not start at grid index 0")
    breaks = np.flatnonzero(np.diff(index) != 1)
    if len(breaks):
        keep = keep.iloc[: breaks[0] + 1]
    out = keep.copy()
    out.insert(0, "distance_m", out.index.to_numpy(dtype=float) * GRID_SPACING_M)
    return out


def find_troughs(speed: np.ndarray, min_prominence_kmh: float = 8.0) -> list[Trough]:
    """Local minima of a circular trace with prominence >= the threshold.

    Prominence is the smaller of the highest speeds reached on each side before
    the trace drops below the minimum again. On a flat-bottomed trough the
    first point of the floor is reported.
    """
    sp = np.asarray(speed, dtype=float)
    n = len(sp)
    if n < 3 or not np.isfinite(sp).all():
        return []
    out: list[Trough] = []
    for i in range(n):
        if not (sp[i] < sp[i - 1] and sp[i] <= sp[(i + 1) % n]):
            continue
        left = -np.inf
        j = i - 1
        while (j % n) != i and sp[j % n] >= sp[i]:
            left = max(left, sp[j % n])
            j -= 1
        right = -np.inf
        j = i + 1
        while (j % n) != i and sp[j % n] >= sp[i]:
            right = max(right, sp[j % n])
            j += 1
        prominence = float(min(left, right) - sp[i])
        if prominence >= min_prominence_kmh:
            out.append(Trough(index=i, speed=float(sp[i]), prominence=prominence))
    return out


def _wrap_window(corner_d: np.ndarray, start: float, end: float, lap_length: float) -> np.ndarray:
    """Membership of corner distances in ``[start, end]`` on the circular lap."""
    start %= lap_length
    end %= lap_length
    if end >= start:
        return (corner_d >= start) & (corner_d <= end)
    return (corner_d >= start) | (corner_d <= end)


def detect_events(
    traces: pd.DataFrame,
    corners: pd.DataFrame | None = None,
    params: EventParams = EventParams(),
    grid_m: float = GRID_SPACING_M,
) -> pd.DataFrame:
    """Event table from median traces, typed per :data:`EVENT_SCHEMA`.

    All boundaries are computed on unwrapped index coordinates so that an event
    across the start/finish line is handled once, then stored modulo the lap
    length. ``corners`` (output of ``corner_positions``) supplies labels; it may
    be ``None`` or empty, in which case every label is null.
    """
    for column in ("speed", "brake", "throttle"):
        if column not in traces.columns:
            raise EventError(f"traces are missing {column!r}")
    sp = traces["speed"].to_numpy(dtype=float)
    br = np.nan_to_num(traces["brake"].to_numpy(dtype=float), nan=0.0)
    th = traces["throttle"].to_numpy(dtype=float)
    n = len(sp)
    lap_length = n * grid_m

    troughs = find_troughs(sp, params.min_prominence_kmh)
    if not troughs:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in EVENT_SCHEMA.items()})

    idx = [t.index for t in troughs]
    count = len(idx)

    def prev_index(k: int) -> int:
        return idx[k - 1] if k > 0 else idx[-1] - n

    def next_index(k: int) -> int:
        return idx[k + 1] if k + 1 < count else idx[0] + n

    # Pass 1: approach-side boundaries and the apex window, in raw indices.
    raw: list[dict] = []
    for k, trough in enumerate(troughs):
        i, i_prev, i_next = trough.index, prev_index(k), next_index(k)
        between = np.arange(i_prev + 1, i)
        left_max = int(between[sp[between % n].argmax()]) if len(between) else i
        approach = np.arange(left_max, i + 1)
        on = br[approach % n] >= params.brake_on_fraction
        if on.any():
            brake_on = int(approach[np.flatnonzero(on)[0]])
            brake_off = int(approach[np.flatnonzero(on)[-1]]) + 1
        else:
            brake_on = brake_off = None
        limit = trough.speed + params.apex_fraction * trough.prominence
        a0 = i
        while a0 - 1 > i_prev and sp[(a0 - 1) % n] <= limit:
            a0 -= 1
        a1 = i
        while a1 + 1 < i_next and sp[(a1 + 1) % n] <= limit:
            a1 += 1
        a1 += 1  # half-open end

        # The lift is sought on the approach *before the apex band*, not before
        # the trough (F018). At a flat-out corner the median driver is still at
        # full throttle when speed has already entered the band, so searching to
        # the trough put the lift 10-20 m inside it and broke F011's ordering
        # invariant. ``lift == a0`` now reads "no lift before the band" and
        # yields no entry phase, which is how those events already segmented.
        approach_pre = np.arange(left_max, a0 + 1)
        open_throttle = np.flatnonzero(th[approach_pre % n] >= params.throttle_lift_pct)
        lift = int(approach_pre[open_throttle[-1]]) + 1 if len(open_throttle) else left_max
        lift = min(lift, a0)

        start = min(lift, a0) if brake_on is None else min(lift, brake_on, a0)
        raw.append(
            dict(i=i, i_next=i_next, left_max=left_max, lift=lift, brake_on=brake_on,
                 brake_off=brake_off, a0=a0, a1=a1, start=start)
        )

    # Pass 2: exit end, bounded by where the next event begins.
    for k, r in enumerate(raw):
        nxt = raw[k + 1] if k + 1 < count else {key: (v + n if isinstance(v, int) else v) for key, v in raw[0].items()}
        after = np.arange(r["a1"], r["i_next"])
        target = troughs[k].speed + params.exit_recovery_fraction * troughs[k].prominence
        hit = np.flatnonzero(sp[after % n] >= target) if len(after) else np.empty(0, dtype=int)
        exit_end = int(after[hit[0]]) if len(hit) else int(nxt["left_max"])
        r["exit_end"] = max(min(exit_end, int(nxt["start"])), r["a1"])

    corner_d = corners["distance_m"].to_numpy(dtype=float) if corners is not None and len(corners) else np.empty(0)
    corner_n = corners["number"].to_numpy() if len(corner_d) else np.empty(0, dtype=int)
    corner_l = corners["letter"].fillna("").to_numpy() if len(corner_d) and "letter" in corners else np.array([""] * len(corner_d))

    def metres(index: int | None) -> float:
        return float("nan") if index is None else float((index % n) * grid_m)

    # Labels in two passes: every event's own window first, so the fallback can
    # only take a marker no other event claimed.
    empty = np.zeros(len(corner_d), dtype=bool)
    windows = [
        _wrap_window(corner_d, r["start"] * grid_m - params.label_margin_m,
                     r["exit_end"] * grid_m + params.label_margin_m, lap_length)
        if len(corner_d) else empty
        for r in raw
    ]
    if len(corner_d):
        claimed = np.logical_or.reduce(windows) if windows else empty
        for k, r in enumerate(raw):
            if windows[k].any():
                continue
            # A flat-out corner has no approach phase, so its window collapses
            # onto the apex band and misses the marker at the turn-in ahead of
            # it (F018). Reach back from the band for unclaimed markers only.
            a0_m = r["a0"] * grid_m
            near = _wrap_window(corner_d, a0_m - params.fallback_margin_m, a0_m, lap_length) & ~claimed
            windows[k] = near
            claimed = claimed | near

    rows = []
    for k, (trough, r) in enumerate(zip(troughs, raw)):
        members = windows[k]
        rows.append(
            dict(
                event_id=k,
                apex_m=metres(trough.index),
                v_min_kmh=trough.speed,
                prominence_kmh=trough.prominence,
                left_max_m=metres(r["left_max"]),
                lift_m=metres(r["lift"]),
                brake_on_m=metres(r["brake_on"]),
                brake_off_m=metres(r["brake_off"]),
                apex_start_m=metres(r["a0"]),
                apex_end_m=metres(r["a1"]),
                exit_end_m=metres(r["exit_end"]),
                has_braking=r["brake_on"] is not None,
                corners=corner_label([int(x) for x in corner_n[members]], [str(x) for x in corner_l[members]]),
                marginal=trough.prominence < params.marginal_factor * params.min_prominence_kmh,
            )
        )
    return pd.DataFrame(rows)[list(EVENT_SCHEMA)].astype(EVENT_SCHEMA)
