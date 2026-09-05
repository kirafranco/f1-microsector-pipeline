"""Named predicates and structural invariants used by the contracts.

Each function's ``__name__`` appears in the finding it produces, so a report
says *why* a null was permitted or *which* invariant broke, not merely that a
rule fired.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

LAP_KEY = ["driver", "lap_number"]


def _false(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=frame.index)


def _lap_groups(frame: pd.DataFrame):
    return frame.groupby(LAP_KEY, observed=True, sort=False).groups.items()


# --- permitted-null predicates -------------------------------------------------

def lap_not_accurate(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """FastF1 marks a lap inaccurate; it then carries no timing."""
    return ~frame["is_accurate"].fillna(False).astype(bool)


def at_grid_zero(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """Grid point 0 can precede the first source sample, so it brackets nothing."""
    return frame["grid_index"].astype("Int64").fillna(-1) == 0


def before_first_sample(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """Grid points ahead of a lap's first real sample bracket no source pair.

    Grid zero is the usual case and was once the only one assumed. It is not:
    the shared axis has one origin for the whole session, so a lap whose
    telemetry opened later than the rest can have several leading grid points
    with nothing behind them -- six of them on one lap of the 2024 Australian
    Grand Prix (F015). Those points are interpolated from one side and carry no
    source gap, which is a fact about them rather than a defect.
    """
    if "source_gap_m" not in frame.columns:
        return frame["grid_index"].astype("Int64").fillna(-1) == 0
    index = frame["grid_index"].astype("Int64").fillna(-1)
    first_real = (
        frame.assign(_i=index)
        .loc[frame["source_gap_m"].notna()]
        .groupby(["driver", "lap_number"], observed=True)["_i"]
        .min()
    )
    per_row = pd.MultiIndex.from_arrays([frame["driver"], frame["lap_number"]])
    threshold = pd.Series(first_real.reindex(per_row).to_numpy(), index=frame.index)
    return index.to_numpy() < threshold.fillna(0).to_numpy()


def event_has_no_braking(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """A lift-only corner has no braking point to record."""
    return ~frame["has_braking"].fillna(False).astype(bool)


def corner_event_has_no_braking(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """Same, seen from a per-lap row that must look the event up."""
    events = parents.get("events")
    if events is None or events.empty:
        return _false(frame)
    braking = events.set_index(events["event_id"].astype("Int64"))["has_braking"].fillna(False).astype(bool)
    mapped = frame["event_id"].astype("Int64").map(braking)
    return ~mapped.fillna(False).astype(bool)


def sector_is_not_an_event(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """Straights and fixed bins belong to no corner event."""
    return frame["phase"].isin(["straight", "bin"])


def row_is_partial(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """The lap ended inside this micro-sector, so it has no time."""
    return frame["partial"].fillna(False).astype(bool)


def sector_has_no_laps(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """No lap completed this sector, so its statistics are undefined."""
    return frame["n_laps"].fillna(0).astype("Int64") == 0


def is_the_reference_lap(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """A lap compared with itself has no closure residual."""
    return frame["is_reference"].fillna(False).astype(bool)


# --- structural invariants -----------------------------------------------------

def grid_distance_matches_index(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """F003: distance_m is exactly grid_index x 10 m."""
    expected = frame["grid_index"].astype(float) * 10.0
    return ~np.isclose(frame["distance_m"].astype(float), expected, atol=1e-3)


def grid_index_contiguous_per_lap(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """F003: every lap runs 0, 1, 2, ... with no hole and no repeat."""
    bad = _false(frame)
    for _, index in _lap_groups(frame):
        values = np.sort(frame.loc[index, "grid_index"].to_numpy(dtype=np.int64))
        if not np.array_equal(values, np.arange(len(values))):
            bad.loc[index] = True
    return bad


def elapsed_time_increasing_per_lap(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """F003 criterion 7: time never stands still or goes backwards along a lap."""
    bad = _false(frame)
    for _, index in _lap_groups(frame):
        sub = frame.loc[index].sort_values("grid_index")
        values = sub["elapsed_time"].to_numpy(dtype=float)
        if len(values) > 1 and not (np.diff(values) > 0).all():
            bad.loc[index] = True
    return bad


def aligned_distance_non_decreasing(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """F008: the axis is monotonic within a lap."""
    bad = _false(frame)
    for _, index in _lap_groups(frame):
        sub = frame.loc[index].sort_values("session_time")
        values = sub["distance_aligned"].to_numpy(dtype=float)
        if len(values) > 1 and (np.diff(values) < -1e-6).any():
            bad.loc[index] = True
    return bad


def aligned_laps_are_accurate(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """F008 aligns flying laps only, so every aligned lap must be an accurate one."""
    laps = parents.get("laps")
    if laps is None or laps.empty:
        return _false(frame)
    accurate = {
        (str(d), int(n))
        for d, n, ok in zip(laps["driver"], laps["lap_number"], laps["is_accurate"].fillna(False))
        if bool(ok) and not pd.isna(n)
    }
    keys = zip(frame["driver"].astype(str), frame["lap_number"].astype("Int64"))
    return pd.Series([(d, int(n)) not in accurate if not pd.isna(n) else True for d, n in keys], index=frame.index)


def microsectors_partition_the_lap(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """F009 criterion 1: each grain covers [0, L) with no gap and no overlap."""
    bad = _false(frame)
    for _, index in frame.groupby("grain", observed=True, sort=False).groups.items():
        sub = frame.loc[index].sort_values("microsector_id")
        start = sub["start_m"].to_numpy(dtype=float)
        end = sub["end_m"].to_numpy(dtype=float)
        contiguous = len(sub) < 2 or np.allclose(start[1:], end[:-1], atol=1e-3)
        if not (contiguous and np.isclose(start[0], 0.0, atol=1e-3) and (end > start).all()):
            bad.loc[index] = True
    return bad


def delta_is_zero_at_the_line(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """F004 criterion 1: every lap's delta starts at zero, by construction."""
    at_zero = frame["grid_index"].astype("Int64").fillna(-1) == 0
    nonzero = frame["delta_t_s"].fillna(0.0).astype(float) != 0.0
    return at_zero & nonzero


def event_boundaries_are_ordered(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """F009: lift <= apex start < apex end <= exit end, on every event."""
    lift = frame["lift_m"].astype(float)
    start = frame["apex_start_m"].astype(float)
    end = frame["apex_end_m"].astype(float)
    exit_end = frame["exit_end_m"].astype(float)
    ordered = (lift <= start) & (start < end) & (end <= exit_end)
    return ~ordered.fillna(True).astype(bool)


def sector_indices_match_distances(frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
    """F009: the index bounds are the metre bounds on the 10 m grid."""
    start_ok = np.isclose(frame["start_index"].astype(float) * 10.0, frame["start_m"].astype(float), atol=1e-3)
    end_ok = np.isclose(frame["end_index"].astype(float) * 10.0, frame["end_m"].astype(float), atol=1e-3)
    return ~(start_ok & end_ok)
