"""Which sessions are due, and which of those are not in the warehouse yet.

The season schedule comes from Jolpica (F012) with each session's UTC start
time. A session is *due* once enough time has passed for it to have run and
for the timing backend to have published it; it is *outstanding* if the
warehouse has no row for it.

Everything here is a pure function of a schedule frame, a clock and a set of
loaded keys, so the dispatcher's judgement can be tested against the real 2024
schedule without a scheduler, a network or a database.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger(__name__)

#: D10 scopes ingestion to qualifying and race sessions.
DEFAULT_CODES = ("Q", "R")

#: How long after a session starts before its data could plausibly exist: the
#: session itself plus D4's publishing lag. A race is two hours, and the lag is
#: tens of minutes to a few hours, so three hours is the earliest worth asking.
#: The sensor then waits properly; this only avoids pointless runs.
SETTLE_MINUTES = 180


class CalendarError(ValueError):
    """The schedule cannot answer what is being asked of it."""


def _require(frame: pd.DataFrame, columns: tuple[str, ...], what: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise CalendarError(f"{what} is missing {missing}")


def sessions_due(schedule: pd.DataFrame, events: pd.DataFrame, now: datetime,
                 codes: tuple[str, ...] = DEFAULT_CODES,
                 settle_minutes: int = SETTLE_MINUTES) -> pd.DataFrame:
    """Sessions of the wanted kinds whose data should exist by `now`.

    Sessions with no scheduled time are excluded rather than guessed at: a
    session the schedule cannot place cannot be judged due.
    """
    _require(schedule, ("season", "round", "session", "session_start_utc"), "schedule")
    _require(events, ("season", "round", "event_name"), "events")
    if now.tzinfo is None:
        raise CalendarError("now must be timezone-aware; the schedule is in UTC")

    wanted = schedule[schedule["session"].isin(codes)].copy()
    wanted = wanted[wanted["session_start_utc"].notna()]
    cutoff = pd.Timestamp(now.astimezone(timezone.utc)) - timedelta(minutes=settle_minutes)
    due = wanted[wanted["session_start_utc"] <= cutoff].copy()

    names = events.set_index(["season", "round"])["event_name"]
    due["event_name"] = [
        names.get((season, round_number))
        for season, round_number in zip(due["season"], due["round"])
    ]
    unnamed = due["event_name"].isna()
    if unnamed.any():
        logger.warning("calendar_rounds_without_a_name rounds=%s",
                       sorted(due.loc[unnamed, "round"].unique().tolist()))
        due = due[~unnamed]

    due = due.rename(columns={"session": "session_code"})
    due = due.sort_values("session_start_utc").reset_index(drop=True)
    logger.info("calendar_due season_codes=%s count=%d cutoff=%s", codes, len(due), cutoff)
    return due[["season", "round", "session_code", "event_name", "session_start_utc"]]


def not_yet_loaded(due: pd.DataFrame, loaded: set[tuple[int, int, str]]) -> pd.DataFrame:
    """Due sessions the warehouse does not already hold.

    A session that failed last hour is simply still outstanding this hour, so
    per-record fault tolerance falls out of asking the question this way rather
    than tracking failures.
    """
    if due.empty:
        return due
    keys = [(int(season), int(round_number), str(code))
            for season, round_number, code in
            zip(due["season"], due["round"], due["session_code"])]
    mask = [key not in loaded for key in keys]
    outstanding = due[pd.Series(mask, index=due.index)].reset_index(drop=True)
    logger.info("calendar_outstanding due=%d loaded=%d outstanding=%d",
                len(due), len(due) - len(outstanding), len(outstanding))
    return outstanding


def run_params(row: pd.Series, method: str | None = None) -> dict:
    """The params one `f1_session_pipeline` run is triggered with."""
    params = {
        "season": int(row["season"]),
        "event": str(row["event_name"]),
        "session": str(row["session_code"]),
    }
    if method:
        params["method"] = str(method)
    return params


def run_id(row: pd.Series) -> str:
    """A stable id per session, so re-offering one cannot duplicate its run.

    Airflow refuses a trigger whose run id already exists. That is the desired
    behaviour: while a session's run is alive, the hourly dispatcher must not
    start a second one, and it does not have to remember that it triggered it.
    """
    return f"session__{int(row['season'])}_{int(row['round'])}_{str(row['session_code']).lower()}"
