"""Schema coercion and the validation gates it enforces."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingest import schemas
from src.ingest.schemas import SchemaError, assert_unique, coerce


def _laps_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Driver": ["VER", "NOR"],
            "DriverNumber": ["1", "4"],
            "Team": ["Red Bull Racing", "McLaren"],
            "LapNumber": [1.0, 1.0],
            "LapTime": pd.to_timedelta([89.123, 89.456], unit="s"),
            "Sector1Time": pd.to_timedelta([30.1, 30.2], unit="s"),
            "Sector2Time": pd.to_timedelta([29.0, 29.1], unit="s"),
            "Sector3Time": pd.to_timedelta([30.0, 30.1], unit="s"),
            "LapStartTime": pd.to_timedelta([0.0, 5.0], unit="s"),
            "PitInTime": pd.to_timedelta([None, None]),
            "PitOutTime": pd.to_timedelta([None, None]),
            "Stint": [1.0, 1.0],
            "Compound": ["SOFT", "SOFT"],
            "TyreLife": [3.0, 4.0],
            "FreshTyre": [True, False],
            "TrackStatus": ["1", "1"],
            "IsAccurate": [True, True],
        }
    )


class TestCoerce:
    def test_renames_and_types_laps(self) -> None:
        out = coerce(_laps_frame(), schemas.LAPS)

        assert list(out.columns) == list(schemas.LAPS.source_columns)
        assert out["lap_number"].dtype == "Int16"
        assert out["driver"].dtype == "string"
        assert out["is_accurate"].dtype == "boolean"

    def test_timedelta_columns_become_float_seconds(self) -> None:
        out = coerce(_laps_frame(), schemas.LAPS)

        assert out["lap_time"].dtype == "float64"
        assert out["lap_time"].iloc[0] == pytest.approx(89.123)
        assert out["sector1_time"].iloc[1] == pytest.approx(30.2)

    def test_all_null_optional_timedelta_survives(self) -> None:
        """PitInTime is null for every non-pitting lap; that is not an error."""
        out = coerce(_laps_frame(), schemas.LAPS)
        assert out["pit_in_time"].isna().all()

    def test_missing_source_column_is_rejected(self) -> None:
        frame = _laps_frame().drop(columns=["Compound"])
        with pytest.raises(SchemaError, match="source columns absent"):
            coerce(frame, schemas.LAPS)

    def test_required_column_entirely_null_is_rejected(self) -> None:
        frame = _laps_frame()
        frame["Driver"] = None
        with pytest.raises(SchemaError, match="entirely null"):
            coerce(frame, schemas.LAPS)

    def test_undeclared_columns_are_dropped(self) -> None:
        frame = _laps_frame()
        frame["SomeFutureFastF1Column"] = 1
        out = coerce(frame, schemas.LAPS)
        assert "SomeFutureFastF1Column" not in out.columns


class TestBrakeChannel:
    """D4: Brake is on/off, never a pressure. It must land as a boolean."""

    def _car_frame(self, brake_values: list[object]) -> pd.DataFrame:
        n = len(brake_values)
        return pd.DataFrame(
            {
                "_driver": ["VER"] * n,
                "_lap_number": list(range(1, n + 1)),
                "SessionTime": pd.to_timedelta([float(i) for i in range(n)], unit="s"),
                "Speed": [250.0] * n,
                "Throttle": [100.0] * n,
                "Brake": brake_values,
                "RPM": [11000.0] * n,
                "nGear": [7] * n,
                "DRS": [0] * n,
                "Source": ["car"] * n,
            }
        )

    def test_native_bools(self) -> None:
        out = coerce(self._car_frame([True, False]), schemas.CAR_TELEMETRY)
        assert out["brake"].dtype == "boolean"
        assert out["brake"].tolist() == [True, False]

    def test_numeric_and_string_representations(self) -> None:
        out = coerce(self._car_frame([1, 0, "true", "False"]), schemas.CAR_TELEMETRY)
        assert out["brake"].tolist() == [True, False, True, False]

    def test_missing_becomes_na_not_false(self) -> None:
        """A missing sample must not silently read as 'not braking'."""
        out = coerce(self._car_frame([True, float("nan")]), schemas.CAR_TELEMETRY)
        assert out["brake"].iloc[0] is True or out["brake"].iloc[0] == True  # noqa: E712
        assert pd.isna(out["brake"].iloc[1])

    def test_gear_and_drs_are_small_ints(self) -> None:
        out = coerce(self._car_frame([True, False]), schemas.CAR_TELEMETRY)
        assert out["drs"].dtype == "Int8", "a status byte, 0-15"
        assert out["n_gear"].dtype == "Int16"

    def test_a_garbage_gear_is_carried_rather_than_crashing_the_ingest(self) -> None:
        """One sample of the 2024 Japanese Grand Prix reads gear 128, exactly
        one past int8, and the safe cast refused it -- taking the whole session
        down at ingest. The type carries it now and F011 tolerates a sliver of
        impossible gears, which is the layer that should be judging it."""
        frame = self._car_frame([True, False])
        frame.loc[0, "nGear"] = 128
        out = coerce(frame, schemas.CAR_TELEMETRY)
        assert int(out["n_gear"].iloc[0]) == 128


class TestAssertUnique:
    def test_accepts_unique_key(self) -> None:
        assert_unique(coerce(_laps_frame(), schemas.LAPS), schemas.LAPS)

    def test_rejects_duplicate_business_key(self) -> None:
        frame = _laps_frame()
        frame.loc[1, "Driver"] = "VER"  # two rows for VER lap 1
        typed = coerce(frame, schemas.LAPS)
        with pytest.raises(SchemaError, match="not unique"):
            assert_unique(typed, schemas.LAPS)
