"""Dimension and fact rows built from the pipeline artefacts.

Pure functions: they take frames and return frames, so every join can be
tested without a database. The one rule worth stating twice is that a lap's
constructor comes from F012's per-round driver entry, never from FastF1's team
string.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

LAP_KEY = ["driver", "lap_number"]

DIM_LAP_COLUMNS = [
    "session_id", "season", "round", "session_code", "code", "lap_number",
    "constructor_id", "team_alias", "compound", "stint", "tyre_life",
    "is_reference", "reference_lap", "n_points",
    "lap_time_s", "grid_time_s", "lap_grid_s", "lap_residual_s", "closure_residual_s",
    "s1_official_s", "s2_official_s", "s3_official_s", "s1_grid_s", "s2_grid_s", "s3_grid_s",
    "driven_m", "line_start_m", "line_end_m", "window_open_s", "window_close_s",
    "start_coverage_poor", "end_coverage_poor",
]

DIM_MICROSECTOR_COLUMNS = [
    "session_id", "grain", "microsector_id", "phase", "event_id", "corners", "marginal",
    "start_m", "end_m", "start_index", "end_index", "length_m",
]

DIM_CORNER_EVENT_COLUMNS = [
    "session_id", "event_id", "corners", "marginal", "has_braking", "apex_m", "v_min_kmh",
    "prominence_kmh", "lift_m", "brake_on_m", "brake_off_m", "apex_start_m", "apex_end_m", "exit_end_m",
]

FACT_GRID_COLUMNS = [
    "season", "round", "session_code", "lap_id", "grid_index", "distance_m", "t_s", "delta_t_s",
    "speed", "throttle", "rpm", "x", "y", "n_gear", "drs", "brake", "source_gap_m",
]

FACT_MICROSECTOR_COLUMNS = [
    "season", "round", "session_code", "lap_id", "grain", "microsector_id", "time_s", "delta_s", "partial",
]

FACT_CORNER_COLUMNS = [
    "lap_id", "event_id", "v_min_kmh", "v_min_m", "apex_dev_m", "brake_on_m", "brake_dev_m", "brake_gap_m",
]


class DimensionError(ValueError):
    """A dimension cannot be built from what was supplied."""


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def session_identity(session_meta: Mapping) -> tuple[int, int, str]:
    """``(season, round, session_code)`` from a F002 snapshot's metadata."""
    try:
        return int(session_meta["season"]), int(session_meta["round_number"]), str(session_meta["session_requested"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DimensionError(f"session metadata has no usable identity: {exc}") from exc


def build_dim_session(
    session_meta: Mapping,
    alignment_meta: Mapping,
    reference: Mapping[str, pd.DataFrame] | None = None,
    snapshot_date: str | None = None,
    contract_version: str | None = None,
) -> dict:
    """The one row describing this session.

    Circuit identity and the session start come from F012 when the reference
    tables are available; the snapshot's own metadata is the fallback, so a
    session can still be loaded before its season's reference data exists.
    """
    season, round_number, session_code = session_identity(session_meta)
    row = {
        "season": season,
        "round": round_number,
        "session_code": session_code,
        "session_name": session_meta.get("session_name") or session_code,
        "event_name": session_meta.get("event_name") or session_meta.get("event_requested") or "",
        "circuit_id": None,
        "circuit_name": None,
        "locality": session_meta.get("location"),
        "country": session_meta.get("country"),
        "session_start_utc": None,
        "official_lap_length_m": alignment_meta.get("official_lap_length_m"),
        "reference_line_length_m": alignment_meta.get("reference_line_length_m"),
        "alignment_method": alignment_meta.get("method"),
        "snapshot_date": snapshot_date,
        "contract_version": contract_version,
    }

    events = (reference or {}).get("dim_event")
    if events is not None and not events.empty:
        match = events[(events["season"] == season) & (events["round"] == round_number)]
        if not match.empty:
            event = match.iloc[0]
            row.update(
                event_name=str(event["event_name"]),
                circuit_id=str(event["circuit_id"]),
                circuit_name=str(event["circuit_name"]),
                locality=str(event["locality"]),
                country=str(event["country"]),
            )

    schedule = (reference or {}).get("dim_session_schedule")
    if schedule is not None and not schedule.empty:
        match = schedule[
            (schedule["season"] == season)
            & (schedule["round"] == round_number)
            & (schedule["session"] == session_code)
        ]
        if not match.empty:
            row["session_start_utc"] = match["session_start_utc"].iloc[0]

    if row["session_start_utc"] is None and session_meta.get("session_date"):
        row["session_start_utc"] = pd.Timestamp(session_meta["session_date"], tz="UTC")
    return row


def resolve_constructors(
    codes: pd.Series, driver_entry: pd.DataFrame | None, season: int, round_number: int
) -> pd.Series:
    """Constructor for each driver code, from that round's entry.

    Never a team-name match: FastF1 and Jolpica agree on six team strings in
    ten (F012), so matching them would mislabel four constructors in ten.
    """
    if driver_entry is None or driver_entry.empty:
        return pd.Series([None] * len(codes), index=codes.index, dtype="object")
    entries = driver_entry[
        (driver_entry["season"] == season) & (driver_entry["round"] == round_number)
    ]
    lookup = dict(zip(entries["code"].astype(str), entries["constructor_id"].astype(str)))
    return codes.astype(str).map(lookup)


def build_dim_lap(
    lap_summary: pd.DataFrame,
    ground_truth: pd.DataFrame | None,
    session_id: int,
    season: int,
    round_number: int,
    session_code: str,
    driver_entry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per lap on the grid, with everything a filter might need."""
    if lap_summary.empty:
        raise DimensionError("lap_summary is empty; there is no lap to load")

    frame = lap_summary.copy()
    frame["driver"] = frame["driver"].astype(str)
    frame["lap_number"] = frame["lap_number"].astype(int)

    if ground_truth is not None and not ground_truth.empty:
        truth = ground_truth.copy()
        truth["driver"] = truth["driver"].astype(str)
        truth["lap_number"] = truth["lap_number"].astype(int)
        extra = [c for c in truth.columns if c not in frame.columns or c in LAP_KEY]
        frame = frame.merge(truth[extra], on=LAP_KEY, how="left")

    frame["session_id"] = session_id
    frame["season"] = season
    frame["round"] = round_number
    frame["session_code"] = session_code
    frame["code"] = frame["driver"]
    frame["team_alias"] = frame.get("team")
    frame["reference_lap"] = frame.get("reference")
    frame["constructor_id"] = resolve_constructors(frame["code"], driver_entry, season, round_number)

    for column in DIM_LAP_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[DIM_LAP_COLUMNS].sort_values(["code", "lap_number"]).reset_index(drop=True)


def build_dim_microsector(
    microsectors: pd.DataFrame, session_id: int, summary: pd.DataFrame | None = None
) -> pd.DataFrame:
    frame = microsectors.copy()
    frame["session_id"] = session_id
    if "length_m" not in frame.columns:
        frame["length_m"] = frame["end_m"].astype(float) - frame["start_m"].astype(float)
    for column in DIM_MICROSECTOR_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[DIM_MICROSECTOR_COLUMNS].reset_index(drop=True)


def build_dim_corner_event(events: pd.DataFrame, session_id: int) -> pd.DataFrame:
    frame = events.copy()
    frame["session_id"] = session_id
    for column in DIM_CORNER_EVENT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[DIM_CORNER_EVENT_COLUMNS].reset_index(drop=True)


def lap_id_column(frame: pd.DataFrame, lap_ids: Mapping[tuple, int]) -> pd.Series:
    """Map ``(code, lap_number)`` to the surrogate key, refusing unknown laps."""
    keys = list(zip(frame["driver"].astype(str), frame["lap_number"].astype(int)))
    missing = {k for k in keys if k not in lap_ids}
    if missing:
        raise DimensionError(f"{len(missing)} lap(s) have no lap_id, e.g. {sorted(missing)[:3]}")
    return pd.Series([lap_ids[k] for k in keys], index=frame.index, dtype="int64")


def build_fact_grid(
    grid: pd.DataFrame, delta: pd.DataFrame | None, lap_ids: Mapping[tuple, int],
    season: int, round_number: int, session_code: str,
) -> pd.DataFrame:
    """Grid points with time and delta attached, keyed by lap_id."""
    frame = grid.copy()
    if delta is not None and not delta.empty:
        wanted = ["driver", "lap_number", "grid_index", "t_s", "delta_t_s"]
        frame = frame.merge(delta[wanted], on=["driver", "lap_number", "grid_index"], how="left")
    else:
        frame["t_s"] = frame["elapsed_time"].astype(float)
        frame["delta_t_s"] = np.nan

    frame["lap_id"] = lap_id_column(frame, lap_ids)
    frame["season"], frame["round"], frame["session_code"] = season, round_number, session_code
    missing_time = int(frame["t_s"].isna().sum())
    if missing_time:
        raise DimensionError(f"{missing_time} grid rows have no t_s; delta_t and grid disagree")
    return frame[FACT_GRID_COLUMNS].reset_index(drop=True)


def build_fact_microsector(
    times: pd.DataFrame, lap_ids: Mapping[tuple, int], season: int, round_number: int, session_code: str
) -> pd.DataFrame:
    frame = times.copy()
    frame["lap_id"] = lap_id_column(frame, lap_ids)
    frame["season"], frame["round"], frame["session_code"] = season, round_number, session_code
    return frame[FACT_MICROSECTOR_COLUMNS].reset_index(drop=True)


def build_fact_corner_metric(corner_metrics: pd.DataFrame, lap_ids: Mapping[tuple, int]) -> pd.DataFrame:
    frame = corner_metrics.copy()
    frame["lap_id"] = lap_id_column(frame, lap_ids)
    return frame[FACT_CORNER_COLUMNS].reset_index(drop=True)
