"""Raw channels over a stretch of track, for corroborating a timing difference.

A micro-sector delta says time was lost somewhere. It does not say the car was
slower, because time can move across a boundary when the source samples fall
differently on two laps -- and the two largest per-sector deltas on the Suzuka
pair are adjacent and of opposite sign, which is what that looks like.

The traces are the independent witness. If one lap is slower in raw speed
across the whole corner, and applies throttle later, then the sector time is
measuring driving. If it is not, the sector time was measuring the grid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.report.pairs import LapKey, ReportError

logger = logging.getLogger(__name__)

CHANNELS = ("speed", "throttle", "n_gear", "brake", "source_gap_m")


def _lap(grid: pd.DataFrame, key: LapKey) -> pd.DataFrame:
    rows = grid[(grid["driver"] == key[0]) & (grid["lap_number"] == key[1])]
    if rows.empty:
        raise ReportError(f"no grid rows for {key[0]} lap {key[1]}")
    return rows.sort_values("grid_index").set_index("grid_index")


def window(grid: pd.DataFrame, a: LapKey, b: LapKey, start_m: float, end_m: float) -> pd.DataFrame:
    """Both laps' channels between two distances, on their shared grid points.

    The cumulative delta is re-zeroed at the start of the window, so it reads
    as time gained or lost inside the window rather than carrying in whatever
    the two laps had accumulated before it.
    """
    left, right = _lap(grid, a), _lap(grid, b)
    shared = left.index.intersection(right.index)
    inside = [index for index in shared
              if start_m <= float(left.loc[index, "distance_m"]) <= end_m]
    if not inside:
        raise ReportError(f"no shared grid points between {start_m} m and {end_m} m")

    left, right = left.loc[inside], right.loc[inside]
    frame = pd.DataFrame({"distance_m": left["distance_m"].astype(float)}, index=inside)
    for channel in CHANNELS:
        if channel not in left.columns:
            continue
        frame[f"{channel}_a"] = left[channel].values
        frame[f"{channel}_b"] = right[channel].values
    frame["d_speed_kmh"] = frame["speed_b"] - frame["speed_a"]

    elapsed_a = left["elapsed_time"].astype(float)
    elapsed_b = right["elapsed_time"].astype(float)
    frame["cumulative_delta_s"] = ((elapsed_b - elapsed_b.iloc[0]) - (elapsed_a - elapsed_a.iloc[0])).values
    return frame


@dataclass(frozen=True)
class SpeedDeficit:
    """How much slower B was than A across a stretch of track.

    `mean_kmh` is the headline: negative means B was slower on average. The
    per-point counts are reported alongside because a mean can be produced by a
    single sample landing badly, and `share_slower` cannot.
    """

    start_m: float
    end_m: float
    mean_kmh: float
    min_kmh: float
    max_kmh: float
    points: int
    slower_points: int

    @property
    def share_slower(self) -> float:
        """Fraction of grid points at which B was the slower car."""
        return self.slower_points / self.points if self.points else float("nan")

    @property
    def one_sided(self) -> bool:
        """B on the same side of A at every point: rare, and not required."""
        return self.min_kmh > 0 or self.max_kmh < 0

    def to_dict(self) -> dict:
        return {"start_m": self.start_m, "end_m": self.end_m, "mean_kmh": self.mean_kmh,
                "min_kmh": self.min_kmh, "max_kmh": self.max_kmh, "points": self.points,
                "slower_points": self.slower_points, "share_slower": self.share_slower,
                "one_sided": self.one_sided}


def speed_deficit(grid: pd.DataFrame, a: LapKey, b: LapKey, start_m: float, end_m: float) -> SpeedDeficit:
    """B minus A speed over a window. Negative means B was slower."""
    frame = window(grid, a, b, start_m, end_m)
    difference = frame["d_speed_kmh"].to_numpy(dtype=float)
    return SpeedDeficit(start_m=start_m, end_m=end_m, mean_kmh=float(np.mean(difference)),
                        min_kmh=float(np.min(difference)), max_kmh=float(np.max(difference)),
                        points=int(len(difference)), slower_points=int((difference < 0).sum()))


def deficit_across_pairings(grid: pd.DataFrame, laps_a: list[LapKey], laps_b: list[LapKey],
                            start_m: float, end_m: float) -> pd.DataFrame:
    """The same speed comparison for every lap pairing.

    One lap being slower through a corner is an anecdote; the same driver being
    slower there on every pairing of their quick laps is the claim.
    """
    rows = []
    for a in laps_a:
        for b in laps_b:
            deficit = speed_deficit(grid, a, b, start_m, end_m)
            rows.append({"pair": f"{a[0]} L{a[1]} vs {b[0]} L{b[1]}", **deficit.to_dict()})
    return pd.DataFrame(rows).set_index("pair")
