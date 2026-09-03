"""F004 time curves and delta-t, checked in closed form."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.grid.resample import GRID_SCHEMA
from src.metrics.delta import DELTA_SCHEMA, DeltaError, delta_t, lap_lengths, reference_curves, time_curves
from src.metrics.reference import lap_index
from tests import synthetic_session as syn


def constant_speed_grid(specs: list[tuple[str, int, float, int, float]]) -> pd.DataFrame:
    """Laps at constant speed: (driver, lap, speed_kmh, n_points, elapsed_offset_at_zero)."""
    frames = []
    for driver, lap, speed, n, offset in specs:
        dt = 10.0 / (speed / 3.6)
        frames.append(
            pd.DataFrame(
                {
                    "driver": driver, "lap_number": lap, "grid_index": np.arange(n),
                    "distance_m": np.arange(n) * 10.0, "elapsed_time": offset + dt * np.arange(n),
                    "speed": speed, "throttle": 100.0, "rpm": 10000.0, "x": 0.0, "y": 0.0,
                    "n_gear": 7, "drs": 0, "brake": False, "source_gap_m": 8.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True).astype(GRID_SCHEMA)


@pytest.fixture()
def two_laps() -> pd.DataFrame:
    # 100 km/h -> 0.36 s per bin; 90 km/h -> 0.40 s per bin. The slow lap opens
    # its telemetry window 0.3 s before the line and is one bin shorter.
    return constant_speed_grid([("AAA", 1, 100.0, 50, 0.0), ("BBB", 1, 90.0, 49, 0.3)])


class TestTimeCurves:
    def test_re_zeroed_at_the_line(self, two_laps: pd.DataFrame) -> None:
        curves = time_curves(two_laps)
        assert curves.index.tolist() == [("AAA", 1), ("BBB", 1)]
        assert list(curves.columns) == list(range(50))
        assert (curves[0] == 0.0).all()
        assert curves.loc[("BBB", 1), 10] == pytest.approx(4.0, abs=1e-5)

    def test_nan_beyond_the_lap_end(self, two_laps: pd.DataFrame) -> None:
        curves = time_curves(two_laps)
        assert np.isnan(curves.loc[("BBB", 1), 49])
        assert lap_lengths(curves).tolist() == [50, 49]

    def test_synthetic_session_curves(self) -> None:
        curves = time_curves(syn.grid())
        assert curves.shape == (4, syn.N_POINTS)
        assert (curves[0] == 0.0).all()
        assert curves.notna().all().all()

    def test_missing_grid_zero_is_an_error(self, two_laps: pd.DataFrame) -> None:
        with pytest.raises(DeltaError, match="grid index 0"):
            time_curves(two_laps[two_laps["grid_index"] > 0])

    def test_gap_in_a_lap_is_an_error(self, two_laps: pd.DataFrame) -> None:
        with pytest.raises(DeltaError, match="gaps"):
            time_curves(two_laps[two_laps["grid_index"] != 20])

    def test_missing_column_or_empty(self, two_laps: pd.DataFrame) -> None:
        with pytest.raises(DeltaError):
            time_curves(two_laps.drop(columns=["elapsed_time"]))
        with pytest.raises(DeltaError):
            time_curves(two_laps.iloc[:0])


class TestDeltaT:
    def test_closed_form(self, two_laps: pd.DataFrame) -> None:
        curves = time_curves(two_laps)
        reference = pd.Series([("AAA", 1)] * 2, index=curves.index)
        long = delta_t(curves, reference, "session_fastest")
        slow = long[long["driver"] == "BBB"]
        np.testing.assert_allclose(slow["delta_t_s"].to_numpy(dtype=float), 0.04 * slow["grid_index"].to_numpy(), atol=1e-4)
        assert (long.loc[long["driver"] == "AAA", "delta_t_s"] == 0.0).all()
        assert (long.loc[long["grid_index"] == 0, "delta_t_s"] == 0.0).all()

    def test_rows_only_where_the_lap_exists(self, two_laps: pd.DataFrame) -> None:
        curves = time_curves(two_laps)
        reference = pd.Series([("AAA", 1)] * 2, index=curves.index)
        long = delta_t(curves, reference, "session_fastest")
        assert len(long) == 50 + 49

    def test_nan_where_the_reference_is_shorter(self, two_laps: pd.DataFrame) -> None:
        curves = time_curves(two_laps)
        reference = pd.Series([("BBB", 1)] * 2, index=curves.index)
        long = delta_t(curves, reference, "lap")
        tail = long[(long["driver"] == "AAA") & (long["grid_index"] == 49)]
        assert tail["delta_t_s"].isna().all() and tail["t_s"].notna().all()
        assert (long["reference"] == "BBB L1").all() and (long["reference_kind"] == "lap").all()

    def test_schema(self, two_laps: pd.DataFrame) -> None:
        curves = time_curves(two_laps)
        long = delta_t(curves, pd.Series([("AAA", 1)] * 2, index=curves.index), "session_fastest")
        assert list(long.columns) == list(DELTA_SCHEMA)
        for column, dtype in DELTA_SCHEMA.items():
            assert str(long[column].dtype) == dtype, column

    def test_reference_off_grid_is_an_error(self, two_laps: pd.DataFrame) -> None:
        curves = time_curves(two_laps)
        with pytest.raises(DeltaError, match="not on the grid"):
            reference_curves(curves, pd.Series([("ZZZ", 9)] * 2, index=curves.index))

    def test_driver_best_zeros_each_drivers_own_reference(self) -> None:
        curves = time_curves(syn.grid())
        reference = pd.Series(
            [("AAA", 1), ("AAA", 1), ("BBB", 2), ("BBB", 2)], index=lap_index(curves.index)
        )
        long = delta_t(curves, reference, "driver_best")
        for key in (("AAA", 1), ("BBB", 2)):
            rows = long[(long["driver"] == key[0]) & (long["lap_number"] == key[1])]
            assert (rows["delta_t_s"] == 0.0).all()
        other = long[(long["driver"] == "AAA") & (long["lap_number"] == 2)]
        assert (other["delta_t_s"].iloc[1:] != 0.0).any()
