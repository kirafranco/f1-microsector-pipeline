"""F012 table builders, on recorded payloads.

The fixtures are real responses: a modern weekend and a sprint weekend from
2024, and 1950 for the era that has dates without times and drivers without
three-letter codes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.reference.tables import (
    CONSTRUCTOR_SCHEMA,
    DRIVER_SCHEMA,
    ENTRY_SCHEMA,
    EVENT_SCHEMA,
    SCHEDULE_SCHEMA,
    ReferenceError,
    build_constructor_table,
    build_driver_entries,
    build_driver_table,
    build_event_table,
    build_session_schedule,
    resolve_constructor,
)

FIXTURES = Path(__file__).parent / "fixtures" / "jolpica"


def rows(name: str, table_key: str, list_key: str) -> list[dict]:
    payload = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return payload["MRData"][table_key][list_key]


def races_2024() -> list[dict]:
    return rows("races_2024", "RaceTable", "Races")


def qualifying_2024() -> list[dict]:
    return rows("qualifying_2024_4", "RaceTable", "Races")


class TestEventTable:
    def test_suzuka_is_round_four_with_its_circuit(self) -> None:
        table = build_event_table(races_2024(), 2024)
        suzuka = table[table["round"] == 4].iloc[0]
        assert suzuka["event_name"] == "Japanese Grand Prix"
        assert suzuka["circuit_id"] == "suzuka" and suzuka["country"] == "Japan"
        assert suzuka["lat"] == pytest.approx(34.8431) and suzuka["lon"] == pytest.approx(136.541)
        assert suzuka["race_date"] == "2024-04-07"

    def test_schema_and_ordering(self) -> None:
        table = build_event_table(races_2024(), 2024)
        assert list(table.columns) == list(EVENT_SCHEMA)
        for column, dtype in EVENT_SCHEMA.items():
            assert str(table[column].dtype) == dtype, column
        assert table["round"].is_monotonic_increasing

    def test_an_empty_payload_is_an_error(self) -> None:
        with pytest.raises(ReferenceError, match="no races"):
            build_event_table([], 2024)


class TestSessionSchedule:
    def test_a_normal_weekend_has_three_practices_and_no_sprint(self) -> None:
        schedule = build_session_schedule(races_2024(), 2024)
        round_four = schedule[schedule["round"] == 4]
        assert round_four["session"].tolist() == ["FP1", "FP2", "FP3", "Q", "R"]

    def test_a_sprint_weekend_has_sq_and_s_instead(self) -> None:
        schedule = build_session_schedule(races_2024(), 2024)
        round_five = schedule[schedule["round"] == 5]
        assert round_five["session"].tolist() == ["FP1", "SQ", "S", "Q", "R"]
        assert "FP2" not in set(round_five["session"])

    def test_qualifying_start_matches_the_fastf1_snapshot(self) -> None:
        """The snapshot recorded 2024-04-06 06:00 for Suzuka qualifying."""
        schedule = build_session_schedule(races_2024(), 2024)
        start = schedule[(schedule["round"] == 4) & (schedule["session"] == "Q")]["session_start_utc"].iloc[0]
        assert start == pd.Timestamp("2024-04-06 06:00:00", tz="UTC")

    def test_a_season_without_session_times_is_kept_and_flagged(self) -> None:
        schedule = build_session_schedule(rows("races_1950", "RaceTable", "Races"), 1950)
        assert not schedule.empty
        assert (schedule["session"] == "R").all(), "1950 lists no support sessions"
        assert not schedule["has_time"].any()
        assert schedule["session_start_utc"].notna().all()

    def test_schema(self) -> None:
        schedule = build_session_schedule(races_2024(), 2024)
        assert list(schedule.columns) == list(SCHEDULE_SCHEMA)
        for column, dtype in SCHEDULE_SCHEMA.items():
            assert str(schedule[column].dtype) == dtype, column

    def test_one_row_per_session(self) -> None:
        schedule = build_session_schedule(races_2024(), 2024)
        assert not schedule.duplicated(["season", "round", "session"]).any()


class TestDriverAndConstructorTables:
    def test_every_2024_driver_has_a_code_and_a_number(self) -> None:
        table = build_driver_table(rows("drivers_2024", "DriverTable", "Drivers"), 2024)
        assert len(table) == 25
        assert table["code"].is_unique
        assert table["permanent_number"].notna().all()
        verstappen = table[table["code"] == "VER"].iloc[0]
        assert verstappen["driver_id"] == "max_verstappen"
        assert verstappen["full_name"] == "Max Verstappen"

    def test_drivers_without_a_code_are_dropped(self) -> None:
        """1950 has no three-letter codes, so nothing there can join to FastF1."""
        with pytest.raises(ReferenceError, match="no drivers with a three-letter code"):
            build_driver_table(rows("drivers_1950", "DriverTable", "Drivers"), 1950)

    def test_driver_schema(self) -> None:
        table = build_driver_table(rows("drivers_2024", "DriverTable", "Drivers"), 2024)
        assert list(table.columns) == list(DRIVER_SCHEMA)
        for column, dtype in DRIVER_SCHEMA.items():
            assert str(table[column].dtype) == dtype, column

    def test_constructors(self) -> None:
        table = build_constructor_table(rows("constructors_2024", "ConstructorTable", "Constructors"), 2024)
        assert len(table) == 10
        assert list(table.columns) == list(CONSTRUCTOR_SCHEMA)
        assert set(table["constructor_id"]) >= {"red_bull", "sauber", "rb", "alpine"}


class TestDriverEntries:
    def test_one_row_per_driver_in_the_round(self) -> None:
        entries = build_driver_entries(qualifying_2024(), 2024)
        assert len(entries) == 20
        assert not entries.duplicated(["season", "round", "code"]).any()
        assert (entries["round"] == 4).all()

    def test_schema(self) -> None:
        entries = build_driver_entries(qualifying_2024(), 2024)
        assert list(entries.columns) == list(ENTRY_SCHEMA)
        for column, dtype in ENTRY_SCHEMA.items():
            assert str(entries[column].dtype) == dtype, column

    @pytest.mark.parametrize(
        ("code", "constructor_id", "fastf1_team", "jolpica_name"),
        [
            ("GAS", "alpine", "Alpine", "Alpine F1 Team"),
            ("BOT", "sauber", "Kick Sauber", "Sauber"),
            ("RIC", "rb", "RB", "RB F1 Team"),
            ("PER", "red_bull", "Red Bull Racing", "Red Bull"),
        ],
    )
    def test_the_four_teams_whose_names_differ_still_resolve(
        self, code: str, constructor_id: str, fastf1_team: str, jolpica_name: str
    ) -> None:
        """The reason constructors are resolved by entry, never by team string."""
        entries = build_driver_entries(qualifying_2024(), 2024)
        assert resolve_constructor(entries, 2024, 4, code) == constructor_id
        row = entries[entries["code"] == code].iloc[0]
        assert row["constructor_name"] == jolpica_name
        assert row["constructor_name"] != fastf1_team, "these are exactly the names that do not match"

    def test_matching_on_team_strings_would_fail(self) -> None:
        """The mistake this design exists to prevent, demonstrated."""
        entries = build_driver_entries(qualifying_2024(), 2024)
        fastf1_teams = {"Alpine", "Kick Sauber", "RB", "Red Bull Racing", "Ferrari", "McLaren",
                        "Mercedes", "Aston Martin", "Williams", "Haas F1 Team"}
        jolpica_names = set(entries["constructor_name"].dropna())
        matched = fastf1_teams & jolpica_names
        assert len(matched) == 6, "only six of ten team strings agree"
        assert len(fastf1_teams - jolpica_names) == 4

    def test_resolve_returns_none_for_an_unknown_driver(self) -> None:
        entries = build_driver_entries(qualifying_2024(), 2024)
        assert resolve_constructor(entries, 2024, 4, "ZZZ") is None
        assert resolve_constructor(entries, 2024, 99, "VER") is None

    def test_race_results_are_accepted_when_qualifying_is_absent(self) -> None:
        payload = json.loads((FIXTURES / "qualifying_2024_4.json").read_text(encoding="utf-8"))
        race = payload["MRData"]["RaceTable"]["Races"][0]
        race["Results"] = race.pop("QualifyingResults")
        entries = build_driver_entries([race], 2024)
        assert len(entries) == 20

    def test_an_empty_payload_is_an_error(self) -> None:
        with pytest.raises(ReferenceError, match="no driver entries"):
            build_driver_entries([], 2024)
