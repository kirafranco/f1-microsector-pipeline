"""Decomposing the gap between two laps, with the noise that bounds it.

Every delta here is B minus A, so a positive number is time lost by B. Sectors
where either lap ended part-way through are excluded from every total: their
times cover different distances and subtracting them would compare a sector
with a fraction of one.

The uncertainty attached to each group is the within-driver spread of those
same micro-sectors across the session (F004), combined in quadrature. It is a
generous bound -- it contains genuine lap-to-lap driving variation as well as
measurement noise -- and it is used precisely because the conclusions here are
mostly negative: a difference that cannot clear a generous bound certainly
cannot clear a tight one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: The grain the write-up uses. Fixed 100 m bins exist too (F009) but they cut
#: across corner phases, which is the distinction the question is about.
GRAIN = "corner_phase"

#: One lap: the driver's three-letter code and the lap number.
LapKey = tuple[str, int]


class ReportError(ValueError):
    """The frames cannot answer the question asked of them."""


def quadrature(values: pd.Series | np.ndarray) -> float:
    """Independent one-sigma errors combined over a group."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan")
    return float(np.sqrt(np.square(finite).sum()))


def _lap_times(times: pd.DataFrame, key: LapKey, grain: str) -> pd.DataFrame:
    rows = times[(times["grain"] == grain)
                 & (times["driver"] == key[0])
                 & (times["lap_number"] == key[1])]
    if rows.empty:
        raise ReportError(f"no {grain} times for {key[0]} lap {key[1]}")
    return rows.set_index("microsector_id")


def sector_frame(microsectors: pd.DataFrame, summary: pd.DataFrame, grain: str = GRAIN) -> pd.DataFrame:
    """The session's micro-sector geometry with the noise attached to each one.

    Boundaries come from the segmentation artefact rather than being inferred
    from cumulative lengths, so a grain that does not partition the lap end to
    end still lands in the right place.
    """
    geometry = (microsectors[microsectors["grain"] == grain]
                .set_index("microsector_id")
                .sort_index()[["phase", "event_id", "corners", "start_m", "end_m"]])
    if geometry.empty:
        raise ReportError(f"no micro-sectors at grain {grain!r}")
    geometry["length_m"] = geometry["end_m"] - geometry["start_m"]
    noise = (summary[summary["grain"] == grain]
             .set_index("microsector_id")["within_driver_std_s"])
    geometry["sigma_s"] = noise.reindex(geometry.index)
    return geometry


@dataclass(frozen=True)
class PairDecomposition:
    """Two laps, taken apart micro-sector by micro-sector."""

    a: LapKey
    b: LapKey
    sectors: pd.DataFrame
    events: pd.DataFrame
    phases: pd.DataFrame
    total_s: float
    complete: int
    of: int

    @property
    def label(self) -> str:
        return f"{self.a[0]} L{self.a[1]} vs {self.b[0]} L{self.b[1]}"

    @property
    def above_noise(self) -> pd.DataFrame:
        """Events whose delta clears twice their own spread. Usually empty."""
        return self.events[self.events["ratio"] >= 2.0]

    def to_dict(self) -> dict:
        return {
            "a": f"{self.a[0]} L{self.a[1]}",
            "b": f"{self.b[0]} L{self.b[1]}",
            "total_s": self.total_s,
            "complete_sectors": self.complete,
            "of_sectors": self.of,
            "phases": {
                str(phase): {"delta_s": float(row.delta_s), "sigma_s": float(row.sigma_s),
                             "ratio": float(row.ratio), "sectors": int(row.sectors),
                             "length_m": float(row.length_m)}
                for phase, row in self.phases.iterrows()
            },
            "events": {
                str(row.corners): {"delta_s": float(row.delta_s), "sigma_s": float(row.sigma_s),
                                   "ratio": float(row.ratio)}
                for _, row in self.events.iterrows()
            },
            "events_above_2_sigma": [str(row.corners) for _, row in self.above_noise.iterrows()],
        }


def decompose(times: pd.DataFrame, microsectors: pd.DataFrame, summary: pd.DataFrame,
              a: LapKey, b: LapKey, grain: str = GRAIN) -> PairDecomposition:
    """Where the time between two laps went, per sector, event and phase."""
    frame = sector_frame(microsectors, summary, grain)
    left, right = _lap_times(times, a, grain), _lap_times(times, b, grain)

    frame["t_a"] = left["time_s"].reindex(frame.index)
    frame["t_b"] = right["time_s"].reindex(frame.index)
    partial = (left["partial"].reindex(frame.index).fillna(True)
               | right["partial"].reindex(frame.index).fillna(True))
    frame["partial"] = partial.astype(bool)
    frame["delta_s"] = frame["t_b"] - frame["t_a"]
    frame["ratio"] = (frame["delta_s"].abs() / frame["sigma_s"]).replace([np.inf, -np.inf], np.nan)

    complete = frame[~frame["partial"] & frame["delta_s"].notna()]
    if complete.empty:
        raise ReportError(f"no complete micro-sector shared by {a} and {b}")

    events = (complete[complete["phase"] != "straight"]
              .groupby("event_id", observed=True)
              .agg(corners=("corners", "first"), sectors=("delta_s", "size"),
                   start_m=("start_m", "min"), length_m=("length_m", "sum"),
                   delta_s=("delta_s", "sum"), sigma_s=("sigma_s", quadrature)))
    events["ratio"] = events["delta_s"].abs() / events["sigma_s"]

    phases = (complete.groupby("phase", observed=True)
              .agg(sectors=("delta_s", "size"), length_m=("length_m", "sum"),
                   delta_s=("delta_s", "sum"), sigma_s=("sigma_s", quadrature)))
    phases["ratio"] = phases["delta_s"].abs() / phases["sigma_s"]

    total = float(complete["delta_s"].sum())
    logger.info("pair_decomposed a=%s%d b=%s%d total_s=%.4f complete=%d of=%d",
                a[0], a[1], b[0], b[1], total, len(complete), len(frame))
    return PairDecomposition(a=a, b=b, sectors=frame, events=events, phases=phases,
                             total_s=total, complete=len(complete), of=len(frame))


def timed_laps(lap_summary: pd.DataFrame, driver: str) -> list[int]:
    """A driver's laps that have an official time, quickest first.

    In- and out-laps carry no lap time and are not comparable pace; they are
    dropped here rather than filtered at every call site.
    """
    rows = lap_summary[(lap_summary["driver"] == driver) & lap_summary["lap_time_s"].notna()]
    return [int(n) for n in rows.sort_values("lap_time_s")["lap_number"]]


def pairings(times: pd.DataFrame, microsectors: pd.DataFrame, summary: pd.DataFrame,
             lap_summary: pd.DataFrame, driver_a: str, driver_b: str,
             grain: str = GRAIN) -> pd.DataFrame:
    """Every lap of A against every lap of B, one row per pairing.

    A single lap-pair delta is one draw from a distribution whose width is
    about the size of the thing being measured. Repeating the comparison over
    every pairing turns it into a question about signs, which is answerable.
    """
    laps_a, laps_b = timed_laps(lap_summary, driver_a), timed_laps(lap_summary, driver_b)
    if not laps_a or not laps_b:
        raise ReportError(f"no timed laps for {driver_a if not laps_a else driver_b}")

    times_by_lap = {(driver, lap): lap_summary.loc[
        (lap_summary["driver"] == driver) & (lap_summary["lap_number"] == lap), "lap_time_s"].iloc[0]
        for driver, laps in ((driver_a, laps_a), (driver_b, laps_b)) for lap in laps}

    rows = []
    for lap_a in laps_a:
        for lap_b in laps_b:
            pair = decompose(times, microsectors, summary, (driver_a, lap_a), (driver_b, lap_b), grain)
            row = {"pair": f"{driver_a} L{lap_a} vs {driver_b} L{lap_b}",
                   "lap_a": lap_a, "lap_b": lap_b,
                   "gap_s": float(times_by_lap[(driver_b, lap_b)] - times_by_lap[(driver_a, lap_a)]),
                   "total_s": pair.total_s,
                   "straights_s": float(pair.phases.loc["straight", "delta_s"]) if "straight" in pair.phases.index else float("nan")}
            row.update({str(event.corners): float(event.delta_s) for _, event in pair.events.iterrows()})
            rows.append(row)
    return pd.DataFrame(rows).set_index("pair")


def event_noise(microsectors: pd.DataFrame, summary: pd.DataFrame, grain: str = GRAIN) -> pd.Series:
    """One sigma per corner event, labelled the way `pairings` labels its columns."""
    frame = sector_frame(microsectors, summary, grain)
    corners = frame[frame["phase"] != "straight"]
    grouped = corners.groupby("event_id", observed=True).agg(
        corners=("corners", "first"), sigma_s=("sigma_s", quadrature))
    return pd.Series(grouped["sigma_s"].values, index=grouped["corners"].astype(str), name="sigma_s")


def consistency(frame: pd.DataFrame, noise: pd.Series) -> pd.DataFrame:
    """How often each corner points the same way, and by how much.

    `positive` is the sign test the write-up leans on. `above_2_sigma` counts
    pairings large enough to notice on their own, which is a different and much
    rarer thing.
    """
    events = [column for column in frame.columns if column in noise.index]
    rows = []
    for event in events:
        values = frame[event].dropna()
        sigma = float(noise[event])
        rows.append({
            "corners": event,
            "pairings": int(len(values)),
            "positive": int((values > 0).sum()),
            "median_s": float(values.median()),
            "min_s": float(values.min()),
            "max_s": float(values.max()),
            "sigma_s": sigma,
            "above_2_sigma": int((values.abs() >= 2 * sigma).sum()),
        })
    out = pd.DataFrame(rows).set_index("corners")
    out["always_same_sign"] = (out["positive"] == out["pairings"]) | (out["positive"] == 0)
    return out.sort_values("median_s", ascending=False)


def control(times: pd.DataFrame, microsectors: pd.DataFrame, summary: pd.DataFrame,
            driver: str, lap_1: int, lap_2: int, grain: str = GRAIN) -> PairDecomposition:
    """A driver against his own other lap: what the same car and hands produce.

    Whatever this shows at a corner is the floor for what a between-driver
    difference at that corner has to beat.
    """
    return decompose(times, microsectors, summary, (driver, lap_1), (driver, lap_2), grain)
