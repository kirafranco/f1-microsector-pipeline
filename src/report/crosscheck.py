"""Checking the decomposition against things it did not come from.

Two cross-checks, both of which caught something.

The first reconciles grid sector gaps with official sector gaps. They disagree
by up to 0.06 s, which would be alarming if F010 had not already measured, per
lap, how far that lap's grid time sits from its official time. The difference
of those two residuals should equal the disagreement -- and it does, to within
0.003 s per sector. The pipeline knows why it disagrees with the timing feed.

The second counts brake applications instead of trusting a first-brake-on
metric. At Suzuka's T8 kink, 71 of 74 laps dab the brake before braking for
Degner. A lap that skips the dab reports a braking point 100 m later than one
that does not, and the two laps are braking in the same place. The metric is
not wrong about what it measures; it is measuring the wrong application.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.report.pairs import LapKey, ReportError

logger = logging.getLogger(__name__)

#: Decision D6: the braking-point resolution of a 10 m grid over ~4 Hz source
#: data. A difference smaller than this is not a difference.
BRAKE_WINDOW_M = 20.0

SECTORS = ("s1", "s2", "s3")


def _lap_row(frame: pd.DataFrame, key: LapKey, what: str) -> pd.Series:
    rows = frame[(frame["driver"] == key[0]) & (frame["lap_number"] == key[1])]
    if len(rows) != 1:
        raise ReportError(f"{what}: {len(rows)} rows for {key[0]} lap {key[1]}, expected 1")
    return rows.iloc[0]


def grid_time_at(delta_t: pd.DataFrame, key: LapKey) -> pd.Series:
    """One lap's re-zeroed time curve, indexed by grid point."""
    rows = delta_t[(delta_t["driver"] == key[0]) & (delta_t["lap_number"] == key[1])]
    if rows.empty:
        raise ReportError(f"no time curve for {key[0]} lap {key[1]}")
    return rows.set_index("grid_index")["t_s"].sort_index()


def sector_reconciliation(lap_summary: pd.DataFrame, ground_truth: pd.DataFrame,
                          delta_t: pd.DataFrame, boundaries: tuple[float, float],
                          a: LapKey, b: LapKey, spacing_m: float = 10.0) -> pd.DataFrame:
    """Official sector gaps against grid sector gaps, and what explains the difference.

    The grid figures are read off the two laps' time curves at the sector
    boundaries, which is what the micro-sector decomposition implicitly sums to.
    The last column is what is left once F010's per-lap registration residuals
    are accounted for; it is the number that has to be small.
    """
    la, lb = _lap_row(lap_summary, a, "lap_summary"), _lap_row(lap_summary, b, "lap_summary")
    ta, tb = _lap_row(ground_truth, a, "ground_truth"), _lap_row(ground_truth, b, "ground_truth")

    curve_a, curve_b = grid_time_at(delta_t, a), grid_time_at(delta_t, b)
    shared = curve_a.index.intersection(curve_b.index)
    if len(shared) < 2:
        raise ReportError(f"{a} and {b} share {len(shared)} grid points")
    delta = (curve_b - curve_a).loc[shared]

    edges = [int(round(boundary / spacing_m)) for boundary in boundaries] + [int(shared.max())]
    for edge in edges:
        if edge not in delta.index:
            raise ReportError(f"grid index {edge} is not shared by both laps")
    cumulative = [float(delta.loc[edge]) for edge in edges]
    grid_gap = [cumulative[0], cumulative[1] - cumulative[0], cumulative[2] - cumulative[1]]

    official_gap = [float(getattr(lb, f"{name}_official_s") - getattr(la, f"{name}_official_s"))
                    for name in SECTORS]
    residual_difference = [float(getattr(tb, f"{name}_residual_s") - getattr(ta, f"{name}_residual_s"))
                           for name in SECTORS]

    frame = pd.DataFrame({
        "official_gap_s": official_gap,
        "grid_gap_s": grid_gap,
        "boundary_m": [boundaries[0], boundaries[1], float(shared.max()) * spacing_m],
    }, index=list(SECTORS))
    frame["difference_s"] = frame["grid_gap_s"] - frame["official_gap_s"]
    frame["f010_residual_difference_s"] = residual_difference
    frame["unexplained_s"] = frame["difference_s"] - frame["f010_residual_difference_s"]
    logger.info("sector_reconciliation max_unexplained_s=%.4f", frame["unexplained_s"].abs().max())
    return frame


def brake_runs(grid: pd.DataFrame, key: LapKey, start_m: float, end_m: float,
               spacing_m: float = 10.0) -> list[tuple[float, float]]:
    """Contiguous stretches with the brake on, inside a distance window.

    Returned as (first metre on, last metre on). The brake channel is boolean,
    so this counts applications rather than measuring pressure -- FastF1 offers
    nothing finer (D4).
    """
    lap = grid[(grid["driver"] == key[0]) & (grid["lap_number"] == key[1])]
    if lap.empty:
        raise ReportError(f"no grid rows for {key[0]} lap {key[1]}")
    lap = lap.sort_values("grid_index")
    window = lap[(lap["distance_m"] >= start_m) & (lap["distance_m"] <= end_m)]
    on = window["brake"].astype(bool).to_numpy()
    distance = window["distance_m"].to_numpy(dtype=float)

    runs: list[tuple[float, float]] = []
    index = 0
    while index < len(on):
        if not on[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(on) and on[index + 1]:
            index += 1
        runs.append((float(distance[start]), float(distance[index])))
        index += 1
    return runs


def brake_applications(grid: pd.DataFrame, events: pd.DataFrame, key: LapKey) -> pd.DataFrame:
    """Every brake application on one lap, one row per application per event.

    The search window runs from the point where the car was last at full speed
    before the corner to the apex, which is the same window F009 used to find
    the event in the first place.
    """
    rows = []
    for _, event in events.iterrows():
        start = float(event["left_max_m"]) if pd.notna(event["left_max_m"]) else float(event["apex_m"])
        end = float(event["apex_m"])
        runs = brake_runs(grid, key, start, end)
        for order, (on_m, off_m) in enumerate(runs, start=1):
            rows.append({
                "event_id": event["event_id"], "corners": event["corners"],
                "application": order, "of": len(runs),
                "on_m": on_m, "off_m": off_m, "length_m": off_m - on_m,
                "leads_into_apex": order == len(runs),
            })
        if not runs:
            rows.append({
                "event_id": event["event_id"], "corners": event["corners"],
                "application": 0, "of": 0, "on_m": np.nan, "off_m": np.nan,
                "length_m": np.nan, "leads_into_apex": False,
            })
    return pd.DataFrame(rows)


def _application_count(applications: pd.DataFrame) -> pd.Series:
    """Applications per event, zero included."""
    return applications.groupby("event_id", observed=True)["of"].max()


def leading_application(applications: pd.DataFrame) -> pd.DataFrame:
    """The application that takes the car into the corner, per event.

    Not the first one: at a kink before a braking zone the first application is
    a dab that some laps make and others do not, so comparing first applications
    compares different things.
    """
    leading = applications[applications["leads_into_apex"]]
    return leading.set_index("event_id")[["corners", "application", "of", "on_m", "off_m"]]


def braking_comparison(grid: pd.DataFrame, events: pd.DataFrame, corner_metrics: pd.DataFrame,
                       a: LapKey, b: LapKey, window_m: float = BRAKE_WINDOW_M) -> pd.DataFrame:
    """Braking points of two laps, as the metric reports them and as the traces show them.

    `metric_delta_m` is what `fact_corner_metric` says; `leading_delta_m` is
    the same comparison made between the applications that actually lead into
    each apex. Where the two disagree, one lap made an extra dab.
    """
    apps_a = brake_applications(grid, events, a)
    apps_b = brake_applications(grid, events, b)
    lead_a, lead_b = leading_application(apps_a), leading_application(apps_b)

    metric_a = _metrics_by_event(corner_metrics, a)
    metric_b = _metrics_by_event(corner_metrics, b)

    frame = pd.DataFrame(index=events.set_index("event_id").index)
    frame["corners"] = events.set_index("event_id")["corners"]
    frame["has_braking"] = events.set_index("event_id")["has_braking"]
    # Counted from every application, not from the leading one: an event a lap
    # never braked for has no leading application, and reading the count from
    # there would make it missing rather than zero.
    frame["applications_a"] = _application_count(apps_a).reindex(frame.index).fillna(0).astype(int)
    frame["applications_b"] = _application_count(apps_b).reindex(frame.index).fillna(0).astype(int)
    frame["metric_on_a_m"] = metric_a["brake_on_m"].reindex(frame.index)
    frame["metric_on_b_m"] = metric_b["brake_on_m"].reindex(frame.index)
    frame["metric_delta_m"] = frame["metric_on_b_m"] - frame["metric_on_a_m"]
    frame["leading_on_a_m"] = lead_a["on_m"].reindex(frame.index)
    frame["leading_on_b_m"] = lead_b["on_m"].reindex(frame.index)
    frame["leading_delta_m"] = frame["leading_on_b_m"] - frame["leading_on_a_m"]
    frame["gap_a_m"] = metric_a["brake_gap_m"].reindex(frame.index)
    frame["gap_b_m"] = metric_b["brake_gap_m"].reindex(frame.index)
    frame["metric_outside_window"] = frame["metric_delta_m"].abs() > window_m
    frame["leading_outside_window"] = frame["leading_delta_m"].abs() > window_m
    frame["extra_dab"] = frame["applications_a"] != frame["applications_b"]
    frame["multi_application"] = (frame[["applications_a", "applications_b"]].max(axis=1) > 1)
    frame["verdict"] = [_verdict(row, window_m) for _, row in frame.iterrows()]
    return frame


def _verdict(row: pd.Series, window_m: float) -> str:
    """What a braking-point difference at this event actually is.

    - `agree`: both readings put the two laps in the same place.
    - `metric artefact`: the stored metric separates the laps but the
      applications that lead into the apex do not, because one lap made an
      extra application earlier and the metric took it.
    - `confirmed`: both readings separate the laps, so the difference is real.
    - `definition-sensitive`: the two readings disagree at an event where both
      laps braked the same number of times, so the difference is which
      application is being called *the* braking point, not what the cars did.
    """
    if not bool(row["has_braking"]) or pd.isna(row["metric_delta_m"]) or pd.isna(row["leading_delta_m"]):
        return "no braking"
    metric_out, leading_out = bool(row["metric_outside_window"]), bool(row["leading_outside_window"])
    if not metric_out and not leading_out:
        return "agree"
    if metric_out and leading_out:
        return "confirmed"
    if bool(row["extra_dab"]):
        return "metric artefact"
    return "definition-sensitive"


def braking_summary(frame: pd.DataFrame, window_m: float = BRAKE_WINDOW_M) -> dict:
    """The braking findings in the shape the write-up quotes them.

    `metric_outliers_explained` is the claim worth making: every event where
    the stored metric puts the two laps more than the D6 window apart is either
    confirmed by the traces or explained by an extra application. It is not a
    claim that the drivers braked in the same places.
    """
    braked = frame[frame["has_braking"].astype(bool)]
    verdicts = braked["verdict"]
    outliers = braked[braked["metric_outside_window"]]
    explained = outliers["verdict"].isin(("confirmed", "metric artefact"))
    return {
        "events": int(len(braked)),
        "window_m": window_m,
        "agree": int((verdicts == "agree").sum()),
        "confirmed": [str(row.corners) for row in braked[verdicts == "confirmed"].itertuples()],
        "metric_artefact": [str(row.corners) for row in braked[verdicts == "metric artefact"].itertuples()],
        "definition_sensitive": [str(row.corners) for row in braked[verdicts == "definition-sensitive"].itertuples()],
        "multi_application_events": [str(row.corners) for row in braked[braked["multi_application"]].itertuples()],
        "metric_outside_window": int(braked["metric_outside_window"].sum()),
        "leading_outside_window": int(braked["leading_outside_window"].sum()),
        "metric_outliers_explained": bool(explained.all()),
        "per_event": {str(row.corners): {"verdict": str(row.verdict),
                                         "metric_delta_m": float(row.metric_delta_m),
                                         "leading_delta_m": float(row.leading_delta_m),
                                         "applications_a": int(row.applications_a),
                                         "applications_b": int(row.applications_b)}
                      for row in braked.itertuples()},
    }


def _metrics_by_event(corner_metrics: pd.DataFrame, key: LapKey) -> pd.DataFrame:
    rows = corner_metrics[(corner_metrics["driver"] == key[0])
                          & (corner_metrics["lap_number"] == key[1])]
    if rows.empty:
        raise ReportError(f"no corner metrics for {key[0]} lap {key[1]}")
    return rows.set_index("event_id")


def dab_prevalence(grid: pd.DataFrame, events: pd.DataFrame, event_id: int,
                   dab_window_m: tuple[float, float]) -> pd.DataFrame:
    """How many of the session's laps brake inside a given stretch.

    Used to establish that the dab is the norm and its absence the exception,
    rather than the other way round.
    """
    event = events[events["event_id"] == event_id]
    if event.empty:
        raise ReportError(f"no event {event_id}")
    low, high = dab_window_m
    window = grid[(grid["distance_m"] >= low) & (grid["distance_m"] <= high)]
    dabbed = window.groupby(["driver", "lap_number"], observed=True)["brake"].any()
    return dabbed.rename("dabbed").reset_index()
