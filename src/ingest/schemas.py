"""Explicit schemas for the raw FastF1 layer.

Global CLAUDE.md 3.3: no schema inference in production code. Every artefact
declares its output columns, their source column in FastF1, and their dtype.
Anything not declared here is dropped rather than silently carried through.

Time columns are stored as float seconds, not timedeltas: parquet round-trips
them without dtype surprises and every downstream calculation wants seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

#: Columns whose FastF1 source is a Timedelta and which are stored as seconds.
_TIMEDELTA_SUFFIX = ("_time",)


@dataclass(frozen=True)
class ArtefactSchema:
    """Declared shape of one raw artefact."""

    name: str
    #: output column -> FastF1 source column
    source_columns: Mapping[str, str]
    #: output column -> pandas dtype
    dtypes: Mapping[str, str]
    #: business key; uniqueness is asserted before the file is written
    key: tuple[str, ...]
    #: columns that must not be entirely null
    required_non_null: tuple[str, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.source_columns)


LAPS = ArtefactSchema(
    name="laps",
    source_columns={
        "driver": "Driver",
        "driver_number": "DriverNumber",
        "team": "Team",
        "lap_number": "LapNumber",
        "lap_time": "LapTime",
        "sector1_time": "Sector1Time",
        "sector2_time": "Sector2Time",
        "sector3_time": "Sector3Time",
        "lap_start_time": "LapStartTime",
        "pit_in_time": "PitInTime",
        "pit_out_time": "PitOutTime",
        "stint": "Stint",
        "compound": "Compound",
        "tyre_life": "TyreLife",
        "fresh_tyre": "FreshTyre",
        "track_status": "TrackStatus",
        "is_accurate": "IsAccurate",
    },
    dtypes={
        "driver": "string",
        "driver_number": "string",
        "team": "string",
        "lap_number": "Int16",
        "lap_time": "float64",
        "sector1_time": "float64",
        "sector2_time": "float64",
        "sector3_time": "float64",
        "lap_start_time": "float64",
        "pit_in_time": "float64",
        "pit_out_time": "float64",
        "stint": "Int16",
        "compound": "string",
        "tyre_life": "Int16",
        "fresh_tyre": "boolean",
        "track_status": "string",
        "is_accurate": "boolean",
    },
    key=("driver", "lap_number"),
    required_non_null=("driver", "lap_number"),
)

CAR_TELEMETRY = ArtefactSchema(
    name="car_telemetry",
    source_columns={
        "driver": "_driver",
        "lap_number": "_lap_number",
        "session_time": "SessionTime",
        "speed": "Speed",
        "throttle": "Throttle",
        "brake": "Brake",
        "rpm": "RPM",
        "n_gear": "nGear",
        "drs": "DRS",
        "source": "Source",
    },
    dtypes={
        "driver": "string",
        "lap_number": "Int16",
        "session_time": "float64",
        "speed": "float32",
        "throttle": "float32",
        # D4: Brake is an on/off channel, never a pressure.
        "brake": "boolean",
        "rpm": "float32",
        # Int16, not Int8: the feed emits the occasional garbage gear and one
        # sample of the 2024 Japanese Grand Prix reads 128 -- exactly one past
        # int8, which crashed the ingest outright (F015). The warehouse column
        # is already smallint, and F011 tolerates a sliver of impossible gears
        # rather than the type refusing to carry them.
        "n_gear": "Int16",
        "drs": "Int8",
        "source": "string",
    },
    key=("driver", "lap_number", "session_time"),
    required_non_null=("driver", "lap_number", "session_time", "speed"),
)

POS_DATA = ArtefactSchema(
    name="pos_data",
    source_columns={
        "driver": "_driver",
        "lap_number": "_lap_number",
        "session_time": "SessionTime",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "status": "Status",
        "source": "Source",
    },
    dtypes={
        "driver": "string",
        "lap_number": "Int16",
        "session_time": "float64",
        "x": "float32",
        "y": "float32",
        "z": "float32",
        "status": "string",
        "source": "string",
    },
    key=("driver", "lap_number", "session_time"),
    required_non_null=("driver", "lap_number", "session_time", "x", "y"),
)

WEATHER = ArtefactSchema(
    name="weather",
    source_columns={
        "session_time": "Time",
        "air_temp": "AirTemp",
        "track_temp": "TrackTemp",
        "humidity": "Humidity",
        "pressure": "Pressure",
        "wind_speed": "WindSpeed",
        "wind_direction": "WindDirection",
        "rainfall": "Rainfall",
    },
    dtypes={
        "session_time": "float64",
        "air_temp": "float32",
        "track_temp": "float32",
        "humidity": "float32",
        "pressure": "float32",
        "wind_speed": "float32",
        "wind_direction": "Int16",
        "rainfall": "boolean",
    },
    key=("session_time",),
    required_non_null=("session_time",),
)

CIRCUIT_CORNERS = ArtefactSchema(
    name="circuit_corners",
    source_columns={
        "number": "Number",
        "letter": "Letter",
        "x": "X",
        "y": "Y",
        "angle": "Angle",
        "distance": "Distance",
    },
    dtypes={
        "number": "Int16",
        "letter": "string",
        "x": "float32",
        "y": "float32",
        "angle": "float32",
        "distance": "float32",
    },
    key=("number", "letter"),
    required_non_null=("number", "x", "y", "distance"),
)

ALL_SCHEMAS: tuple[ArtefactSchema, ...] = (
    LAPS,
    CAR_TELEMETRY,
    POS_DATA,
    WEATHER,
    CIRCUIT_CORNERS,
)


class SchemaError(ValueError):
    """Raised when incoming data cannot satisfy a declared schema."""


def _to_seconds(series: pd.Series) -> pd.Series:
    """Timedelta -> float seconds, leaving already-numeric series alone."""
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return pd.to_numeric(series, errors="coerce")


def coerce(frame: pd.DataFrame, schema: ArtefactSchema) -> pd.DataFrame:
    """Project `frame` onto `schema`: rename, convert, cast, validate.

    Raises SchemaError if a declared source column is absent or if a
    required column comes out entirely null.
    """
    missing = [src for src in schema.source_columns.values() if src not in frame.columns]
    if missing:
        raise SchemaError(
            f"{schema.name}: source columns absent from FastF1 output: {sorted(missing)}"
        )

    out = pd.DataFrame(index=frame.index)
    for target, source in schema.source_columns.items():
        column = frame[source]
        if target.endswith(_TIMEDELTA_SUFFIX):
            column = _to_seconds(column)
        out[target] = column

    for target, dtype in schema.dtypes.items():
        try:
            out[target] = _cast(out[target], dtype)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"{schema.name}.{target}: cannot cast to {dtype}: {exc}") from exc

    empty = [c for c in schema.required_non_null if out[c].isna().all()]
    if empty and len(out) > 0:
        raise SchemaError(f"{schema.name}: required columns are entirely null: {empty}")

    return out.reset_index(drop=True)


def _cast(series: pd.Series, dtype: str) -> pd.Series:
    """Cast with the rounding that integer targets need to survive floats."""
    if dtype.startswith(("Int", "int")):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.round().astype(dtype)
    if dtype == "boolean":
        # FastF1 hands back numpy bool, object, or float NaN depending on channel.
        return series.astype("object").map(_to_nullable_bool).astype("boolean")
    return series.astype(dtype)


def _to_nullable_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0", ""}:
            return False
        return None
    try:
        return bool(value)
    except (TypeError, ValueError):
        return None


def assert_unique(frame: pd.DataFrame, schema: ArtefactSchema) -> None:
    """Uniqueness gate on the declared business key (global CLAUDE.md 3.3)."""
    if not schema.key:
        return
    duplicated = frame.duplicated(subset=list(schema.key), keep=False)
    if duplicated.any():
        offenders = frame.loc[duplicated, list(schema.key)].head(5).to_dict("records")
        raise SchemaError(
            f"{schema.name}: business key {schema.key} is not unique "
            f"({int(duplicated.sum())} rows); first offenders: {offenders}"
        )
