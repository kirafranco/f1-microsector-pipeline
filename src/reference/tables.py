"""Typed reference tables built from Jolpica payloads.

Explicit dtypes everywhere (global 3.3): nothing here infers a schema from
whatever JSON happened to arrive.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import pandas as pd

#: Jolpica's session names mapped to the short codes FastF1 and this project
#: use. The race itself is the payload's own top-level date/time.
SESSION_KEYS: dict[str, str] = {
    "FirstPractice": "FP1",
    "SecondPractice": "FP2",
    "ThirdPractice": "FP3",
    "Qualifying": "Q",
    "SprintQualifying": "SQ",
    "SprintShootout": "SQ",
    "Sprint": "S",
}
SESSION_CODES = ("FP1", "FP2", "FP3", "SQ", "S", "Q", "R")

EVENT_SCHEMA: dict[str, str] = {
    "season": "Int16",
    "round": "Int8",
    "event_name": "string",
    "circuit_id": "string",
    "circuit_name": "string",
    "locality": "string",
    "country": "string",
    "lat": "float64",
    "lon": "float64",
    "race_date": "string",
    "wikipedia_url": "string",
}

SCHEDULE_SCHEMA: dict[str, str] = {
    "season": "Int16",
    "round": "Int8",
    "session": "string",
    "session_start_utc": "datetime64[ns, UTC]",
    "has_time": "boolean",
}

DRIVER_SCHEMA: dict[str, str] = {
    "season": "Int16",
    "code": "string",
    "driver_id": "string",
    "permanent_number": "Int16",
    "given_name": "string",
    "family_name": "string",
    "full_name": "string",
    "date_of_birth": "string",
    "nationality": "string",
    "wikipedia_url": "string",
}

CONSTRUCTOR_SCHEMA: dict[str, str] = {
    "season": "Int16",
    "constructor_id": "string",
    "name": "string",
    "nationality": "string",
    "wikipedia_url": "string",
}

ENTRY_SCHEMA: dict[str, str] = {
    "season": "Int16",
    "round": "Int8",
    "code": "string",
    "driver_id": "string",
    "constructor_id": "string",
    "constructor_name": "string",
    "car_number": "Int16",
}


class ReferenceError(ValueError):
    """A payload does not have the shape the tables need."""


def _typed(rows: list[dict], schema: Mapping[str, str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(schema))
    return frame.astype(dict(schema))


def _timestamp(date: str | None, time_of_day: str | None) -> tuple[pd.Timestamp | None, bool]:
    """UTC timestamp for a session, and whether the API gave a time at all.

    Older seasons carry a date and no time; those sessions are kept with the
    date at midnight and ``has_time`` false rather than dropped.
    """
    if not date:
        return None, False
    if not time_of_day:
        return pd.Timestamp(date, tz="UTC"), False
    return pd.Timestamp(f"{date} {time_of_day}".replace("Z", ""), tz="UTC"), True


def build_event_table(races: Iterable[dict], season: int) -> pd.DataFrame:
    """One row per round: the event and where it was held."""
    rows = []
    for race in races:
        circuit = race.get("Circuit") or {}
        location = circuit.get("Location") or {}
        rows.append(
            {
                "season": season,
                "round": int(race["round"]),
                "event_name": race.get("raceName"),
                "circuit_id": circuit.get("circuitId"),
                "circuit_name": circuit.get("circuitName"),
                "locality": location.get("locality"),
                "country": location.get("country"),
                "lat": float(location["lat"]) if location.get("lat") is not None else None,
                "lon": float(location["long"]) if location.get("long") is not None else None,
                "race_date": race.get("date"),
                "wikipedia_url": race.get("url"),
            }
        )
    if not rows:
        raise ReferenceError(f"no races to build an event table for season {season}")
    return _typed(rows, EVENT_SCHEMA).sort_values(["season", "round"]).reset_index(drop=True)


def build_session_schedule(races: Iterable[dict], season: int) -> pd.DataFrame:
    """One row per session: the calendar F006 will schedule from.

    A sprint weekend produces SQ and S rows and no FP2/FP3; a normal weekend
    produces three practices. The race is the payload's own date and time.
    """
    rows = []
    for race in races:
        round_number = int(race["round"])
        for key, code in SESSION_KEYS.items():
            block = race.get(key)
            if not block:
                continue
            start, has_time = _timestamp(block.get("date"), block.get("time"))
            rows.append({"season": season, "round": round_number, "session": code,
                         "session_start_utc": start, "has_time": has_time})
        start, has_time = _timestamp(race.get("date"), race.get("time"))
        rows.append({"season": season, "round": round_number, "session": "R",
                     "session_start_utc": start, "has_time": has_time})
    if not rows:
        raise ReferenceError(f"no sessions to schedule for season {season}")
    frame = _typed(rows, SCHEDULE_SCHEMA).drop_duplicates(["season", "round", "session"])
    order = {code: position for position, code in enumerate(SESSION_CODES)}
    frame = frame.assign(_order=frame["session"].map(order)).sort_values(["season", "round", "_order"])
    return frame.drop(columns="_order").reset_index(drop=True)


def build_driver_table(drivers: Iterable[dict], season: int) -> pd.DataFrame:
    """One row per driver in the season, keyed by the three-letter code.

    A driver without a code cannot be joined to FastF1 telemetry, which is what
    this table exists for, so those rows are dropped and counted by the caller.
    """
    rows = []
    for driver in drivers:
        code = driver.get("code")
        if not code:
            continue
        given, family = driver.get("givenName", ""), driver.get("familyName", "")
        rows.append(
            {
                "season": season,
                "code": code,
                "driver_id": driver.get("driverId"),
                "permanent_number": int(driver["permanentNumber"]) if driver.get("permanentNumber") else None,
                "given_name": given,
                "family_name": family,
                "full_name": f"{given} {family}".strip(),
                "date_of_birth": driver.get("dateOfBirth"),
                "nationality": driver.get("nationality"),
                "wikipedia_url": driver.get("url"),
            }
        )
    if not rows:
        raise ReferenceError(f"no drivers with a three-letter code in season {season}")
    return _typed(rows, DRIVER_SCHEMA).sort_values(["season", "code"]).reset_index(drop=True)


def build_constructor_table(constructors: Iterable[dict], season: int) -> pd.DataFrame:
    rows = [
        {
            "season": season,
            "constructor_id": constructor.get("constructorId"),
            "name": constructor.get("name"),
            "nationality": constructor.get("nationality"),
            "wikipedia_url": constructor.get("url"),
        }
        for constructor in constructors
    ]
    if not rows:
        raise ReferenceError(f"no constructors for season {season}")
    return _typed(rows, CONSTRUCTOR_SCHEMA).sort_values(["season", "constructor_id"]).reset_index(drop=True)


def build_driver_entries(races: Iterable[dict], season: int) -> pd.DataFrame:
    """Who drove for whom, per round: the only sound driver-to-constructor link.

    Measured on 2026-09-04: FastF1's team strings agree with Jolpica's
    constructor names only 6 times in 10, so this pairing -- taken from the
    driver's own entry in that round -- is what F005 joins on. The constructor
    name is carried for display, never as a key.
    """
    rows = []
    for race in races:
        round_number = int(race["round"])
        entries = race.get("QualifyingResults") or race.get("Results") or []
        for entry in entries:
            driver = entry.get("Driver") or {}
            constructor = entry.get("Constructor") or {}
            code = driver.get("code")
            if not code:
                continue
            number = entry.get("number") or driver.get("permanentNumber")
            rows.append(
                {
                    "season": season,
                    "round": round_number,
                    "code": code,
                    "driver_id": driver.get("driverId"),
                    "constructor_id": constructor.get("constructorId"),
                    "constructor_name": constructor.get("name"),
                    "car_number": int(number) if number else None,
                }
            )
    if not rows:
        raise ReferenceError(f"no driver entries for season {season}")
    frame = _typed(rows, ENTRY_SCHEMA).drop_duplicates(["season", "round", "code"])
    return frame.sort_values(["season", "round", "code"]).reset_index(drop=True)


def resolve_constructor(entries: pd.DataFrame, season: int, round_number: int, code: str) -> str | None:
    """The constructor a driver drove for in one round, by code.

    The function F005 uses. It exists so that no caller is ever tempted to
    match FastF1's ``team`` string against a constructor name.
    """
    match = entries[
        (entries["season"] == season) & (entries["round"] == round_number) & (entries["code"] == code)
    ]
    if match.empty:
        return None
    return str(match["constructor_id"].iloc[0])
