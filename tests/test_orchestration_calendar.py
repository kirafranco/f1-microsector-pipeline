"""F006 calendar: which sessions are due, and which still need running.

Exercised against the real 2024 schedule F012 ingested, with a fixed clock, so
the dispatcher's judgement is checked on the data it will actually see.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.config import INTERIM_ROOT
from src.orchestration import calendar
from src.orchestration.calendar import CalendarError

SCHEDULE_PATH = INTERIM_ROOT / "reference" / "2024" / "dim_session_schedule.parquet"
EVENTS_PATH = INTERIM_ROOT / "reference" / "2024" / "dim_event.parquet"

#: After the whole 2024 season.
AFTER_THE_SEASON = datetime(2025, 1, 1, tzinfo=timezone.utc)


def frame(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    out["session_start_utc"] = pd.to_datetime(out["session_start_utc"], utc=True)
    return out


SCHEDULE = frame([
    {"season": 2024, "round": 1, "session": "Q", "session_start_utc": "2024-03-01T16:00:00Z"},
    {"season": 2024, "round": 1, "session": "R", "session_start_utc": "2024-03-02T15:00:00Z"},
    {"season": 2024, "round": 1, "session": "FP1", "session_start_utc": "2024-02-29T11:30:00Z"},
    {"season": 2024, "round": 2, "session": "Q", "session_start_utc": "2024-03-08T17:00:00Z"},
    {"season": 2024, "round": 2, "session": "R", "session_start_utc": "2024-03-09T17:00:00Z"},
])
EVENTS = pd.DataFrame([
    {"season": 2024, "round": 1, "event_name": "Bahrain Grand Prix"},
    {"season": 2024, "round": 2, "event_name": "Saudi Arabian Grand Prix"},
])


class TestSessionsDue:
    def test_only_the_wanted_codes(self) -> None:
        """D10 scopes ingestion to qualifying and race sessions."""
        due = calendar.sessions_due(SCHEDULE, EVENTS, AFTER_THE_SEASON)
        assert set(due["session_code"]) == {"Q", "R"}
        assert "FP1" not in set(due["session_code"])

    def test_a_session_that_has_not_settled_yet_is_not_due(self) -> None:
        """A race that started an hour ago has not published (D4)."""
        now = datetime(2024, 3, 2, 16, 0, tzinfo=timezone.utc)  # one hour into round 1's race
        due = calendar.sessions_due(SCHEDULE, EVENTS, now)
        assert (1, "R") not in set(zip(due["round"], due["session_code"]))
        assert (1, "Q") in set(zip(due["round"], due["session_code"]))

    def test_the_settling_window_is_what_makes_it_due(self) -> None:
        start = pd.Timestamp("2024-03-02T15:00:00Z")
        just_before = (start + timedelta(minutes=calendar.SETTLE_MINUTES - 1)).to_pydatetime()
        just_after = (start + timedelta(minutes=calendar.SETTLE_MINUTES + 1)).to_pydatetime()
        assert (1, "R") not in set(zip(*_keys(calendar.sessions_due(SCHEDULE, EVENTS, just_before))))
        assert (1, "R") in set(zip(*_keys(calendar.sessions_due(SCHEDULE, EVENTS, just_after))))

    def test_it_is_ordered_oldest_first(self) -> None:
        """A backfill should work forwards through the season, not randomly."""
        due = calendar.sessions_due(SCHEDULE, EVENTS, AFTER_THE_SEASON)
        assert due["session_start_utc"].is_monotonic_increasing

    def test_each_row_carries_the_event_name_the_pipeline_needs(self) -> None:
        due = calendar.sessions_due(SCHEDULE, EVENTS, AFTER_THE_SEASON)
        assert due.loc[0, "event_name"] == "Bahrain Grand Prix"

    def test_a_round_with_no_name_is_dropped_rather_than_run_unnamed(self) -> None:
        """The event name is the slug; a run without one cannot be named."""
        due = calendar.sessions_due(SCHEDULE, EVENTS.head(1), AFTER_THE_SEASON)
        assert set(due["round"]) == {1}

    def test_a_session_with_no_scheduled_time_is_not_guessed_at(self) -> None:
        schedule = pd.concat([SCHEDULE, frame([
            {"season": 2024, "round": 3, "session": "Q", "session_start_utc": None}])])
        events = pd.concat([EVENTS, pd.DataFrame([
            {"season": 2024, "round": 3, "event_name": "Australian Grand Prix"}])])
        due = calendar.sessions_due(schedule, events, AFTER_THE_SEASON)
        assert 3 not in set(due["round"])

    def test_a_naive_clock_is_refused(self) -> None:
        """The schedule is UTC; comparing it with a naive datetime is a bug."""
        with pytest.raises(CalendarError, match="timezone-aware"):
            calendar.sessions_due(SCHEDULE, EVENTS, datetime(2025, 1, 1))

    def test_a_schedule_missing_a_column_says_which(self) -> None:
        with pytest.raises(CalendarError, match=r"missing \['session_start_utc'\]"):
            calendar.sessions_due(SCHEDULE.drop(columns=["session_start_utc"]), EVENTS, AFTER_THE_SEASON)


def _keys(due: pd.DataFrame) -> tuple[list, list]:
    return list(due["round"]), list(due["session_code"])


class TestNotYetLoaded:
    def test_what_the_warehouse_has_is_subtracted(self) -> None:
        due = calendar.sessions_due(SCHEDULE, EVENTS, AFTER_THE_SEASON)
        outstanding = calendar.not_yet_loaded(due, {(2024, 1, "Q"), (2024, 1, "R")})
        assert set(zip(outstanding["round"], outstanding["session_code"])) == {(2, "Q"), (2, "R")}

    def test_an_empty_warehouse_leaves_everything_outstanding(self) -> None:
        due = calendar.sessions_due(SCHEDULE, EVENTS, AFTER_THE_SEASON)
        assert len(calendar.not_yet_loaded(due, set())) == len(due)

    def test_a_failed_session_is_simply_still_outstanding(self) -> None:
        """Per-record fault tolerance without tracking failures: a run that
        failed left no dim_session row, so it is offered again."""
        due = calendar.sessions_due(SCHEDULE, EVENTS, AFTER_THE_SEASON)
        loaded = {(2024, 1, "Q")}  # round 1 race attempted and failed
        outstanding = calendar.not_yet_loaded(due, loaded)
        assert (1, "R") in set(zip(outstanding["round"], outstanding["session_code"]))

    def test_nothing_due_is_not_an_error(self) -> None:
        empty = calendar.sessions_due(SCHEDULE, EVENTS, datetime(2024, 1, 1, tzinfo=timezone.utc))
        assert calendar.not_yet_loaded(empty, set()).empty


class TestRunIdentity:
    def test_params_are_what_the_pipeline_dag_expects(self) -> None:
        due = calendar.sessions_due(SCHEDULE, EVENTS, AFTER_THE_SEASON)
        params = calendar.run_params(due.iloc[0])
        assert params == {"season": 2024, "event": "Bahrain Grand Prix", "session": "Q"}

    def test_the_run_id_is_stable_for_a_session(self) -> None:
        """Airflow refuses a duplicate run id, which is how a live run is
        protected from being triggered a second time an hour later."""
        due = calendar.sessions_due(SCHEDULE, EVENTS, AFTER_THE_SEASON)
        first = calendar.run_id(due.iloc[0])
        assert first == "session__2024_1_q"
        assert calendar.run_id(due.iloc[0]) == first

    def test_run_ids_are_unique_per_session(self) -> None:
        due = calendar.sessions_due(SCHEDULE, EVENTS, AFTER_THE_SEASON)
        ids = [calendar.run_id(row) for _, row in due.iterrows()]
        assert len(set(ids)) == len(ids)


@pytest.mark.data
class TestTheReal2024Schedule:
    @staticmethod
    def load() -> tuple[pd.DataFrame, pd.DataFrame]:
        if not SCHEDULE_PATH.exists():
            pytest.skip("2024 reference data is not on this machine")
        return pd.read_parquet(SCHEDULE_PATH), pd.read_parquet(EVENTS_PATH)

    def test_the_whole_season_is_due_and_none_of_it_is_loaded(self) -> None:
        schedule, events = self.load()
        due = calendar.sessions_due(schedule, events, AFTER_THE_SEASON)
        assert len(due) == 48, "24 rounds x (Q + R)"
        assert set(due["session_code"]) == {"Q", "R"}
        assert due["event_name"].notna().all()

    def test_with_suzuka_loaded_one_fewer_is_outstanding(self) -> None:
        """The state this machine is actually in after F005."""
        schedule, events = self.load()
        due = calendar.sessions_due(schedule, events, AFTER_THE_SEASON)
        outstanding = calendar.not_yet_loaded(due, {(2024, 4, "Q")})
        assert len(outstanding) == 47
        assert (4, "Q") not in set(zip(outstanding["round"], outstanding["session_code"]))

    def test_a_backfill_is_taken_in_order_and_capped(self) -> None:
        schedule, events = self.load()
        due = calendar.sessions_due(schedule, events, AFTER_THE_SEASON)
        outstanding = calendar.not_yet_loaded(due, set())
        selected = outstanding.head(2)
        assert [calendar.run_id(row) for _, row in selected.iterrows()] == [
            "session__2024_1_q", "session__2024_1_r"]

    def test_mid_season_only_the_past_is_due(self) -> None:
        schedule, events = self.load()
        due = calendar.sessions_due(schedule, events, datetime(2024, 4, 10, tzinfo=timezone.utc))
        assert due["round"].max() == 4, "Suzuka is round 4, 2024-04-07"
        assert len(due) == 8, "four rounds, qualifying and race"
