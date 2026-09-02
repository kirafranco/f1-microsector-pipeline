"""F003 resample_lap on a synthetic lap with analytically known channels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.grid.resample import (
    GRID_SCHEMA,
    ResampleError,
    grid_point_count,
    make_grid,
    resample_lap,
)

SOURCE_STEP_M = 4.0
LAP_LENGTH_M = 1000.0


def synthetic_lap(offset_m: float = 0.0, step_m: float = SOURCE_STEP_M) -> pd.DataFrame:
    """A lap whose channels are exact functions of distance.

    Continuous channels are linear in distance, so linear interpolation onto
    the grid is exact and the expected grid values can be computed directly.
    Discrete channels are piecewise constant with edges on source samples.
    """
    d = np.arange(0.0, LAP_LENGTH_M + step_m / 2, step_m)
    return pd.DataFrame(
        {
            "driver": "SYN",
            "lap_number": np.int16(7),
            # One sample every 0.1 s: elapsed time is linear in distance.
            "session_time": 1000.0 + 0.1 * np.arange(len(d)),
            "distance_aligned": d + offset_m,
            "speed": 100.0 + 0.2 * d,
            "throttle": 100.0 - 0.05 * d,
            "rpm": 8000.0 + 4.0 * d,
            "x": d,
            "y": 2.0 * d,
            "n_gear": pd.array(np.minimum(np.floor(d / 100.0) + 1, 8).astype(int), dtype="Int8"),
            "brake": pd.array(((d >= 200) & (d < 300)) | ((d >= 600) & (d < 650)), dtype="boolean"),
            "drs": pd.array(np.where(d >= 800, 12, 0), dtype="Int8"),
        }
    )


def expected_step(lap: pd.DataFrame, channel: str, grid_distance: np.ndarray) -> pd.Series:
    """Value of the last source sample at or before each grid point."""
    d = lap["distance_aligned"].to_numpy(dtype=float)
    idx = np.clip(np.searchsorted(d, grid_distance, side="right") - 1, 0, len(d) - 1)
    return lap[channel].iloc[idx].reset_index(drop=True)


class TestGridConstruction:
    @pytest.mark.parametrize(
        ("length", "expected"),
        [(1000.0, 101), (999.9, 100), (1000.1, 101), (5722.153, 573), (0.0, 1), (9.99, 1)],
    )
    def test_count_is_floor_over_spacing_plus_one(self, length: float, expected: int) -> None:
        assert grid_point_count(length) == expected
        assert len(make_grid(length)) == expected

    def test_spacing_parameter_changes_the_count(self) -> None:
        assert grid_point_count(1000.0, grid_m=25.0) == 41
        assert make_grid(1000.0, grid_m=25.0)[-1] == 1000.0

    def test_grid_is_exactly_uniform(self) -> None:
        grid = make_grid(5722.153)
        assert np.array_equal(grid, np.arange(573) * 10.0)

    @pytest.mark.parametrize("bad", [0.0, -10.0])
    def test_non_positive_spacing_is_rejected(self, bad: float) -> None:
        with pytest.raises(ResampleError):
            make_grid(1000.0, grid_m=bad)

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_invalid_length_is_rejected(self, bad: float) -> None:
        with pytest.raises(ResampleError):
            make_grid(bad)


class TestSyntheticLap:
    @pytest.fixture()
    def lap(self) -> pd.DataFrame:
        return synthetic_lap()

    @pytest.fixture()
    def grid(self, lap: pd.DataFrame) -> pd.DataFrame:
        return resample_lap(lap)

    def test_grid_structure(self, grid: pd.DataFrame) -> None:
        assert len(grid) == 101
        assert grid["grid_index"].tolist() == list(range(101))
        np.testing.assert_array_equal(grid["distance_m"].to_numpy(dtype=float), np.arange(101) * 10.0)

    def test_output_contract(self, grid: pd.DataFrame) -> None:
        assert list(grid.columns) == list(GRID_SCHEMA)
        for column, dtype in GRID_SCHEMA.items():
            assert str(grid[column].dtype) == dtype, column
        assert (grid["driver"] == "SYN").all()
        assert (grid["lap_number"] == 7).all()

    @pytest.mark.parametrize(
        ("channel", "fn"),
        [
            ("speed", lambda d: 100.0 + 0.2 * d),
            ("throttle", lambda d: 100.0 - 0.05 * d),
            ("rpm", lambda d: 8000.0 + 4.0 * d),
            ("x", lambda d: d),
            ("y", lambda d: 2.0 * d),
            ("elapsed_time", lambda d: d / (SOURCE_STEP_M / 0.1)),
        ],
    )
    def test_linear_channels_are_interpolated_exactly(
        self, grid: pd.DataFrame, channel: str, fn
    ) -> None:
        d = grid["distance_m"].to_numpy(dtype=float)
        np.testing.assert_allclose(grid[channel].to_numpy(dtype=float), fn(d), rtol=1e-6, atol=1e-3)

    @pytest.mark.parametrize("channel", ["n_gear", "brake", "drs"])
    def test_discrete_channels_are_step_previous(
        self, lap: pd.DataFrame, grid: pd.DataFrame, channel: str
    ) -> None:
        expected = expected_step(lap, channel, grid["distance_m"].to_numpy(dtype=float))
        assert grid[channel].tolist() == expected.tolist()

    def test_discrete_channels_invent_nothing(self, lap: pd.DataFrame, grid: pd.DataFrame) -> None:
        for channel in ("n_gear", "brake", "drs"):
            assert set(grid[channel].unique()) <= set(lap[channel].unique())

    def test_gear_is_never_fractional(self, grid: pd.DataFrame) -> None:
        """Grid point 250 sits between samples in gear 3; linear would give 3.5."""
        gears = grid["n_gear"].to_numpy(dtype=float)
        assert np.array_equal(gears, np.round(gears))
        assert grid.loc[grid["distance_m"] == 250.0, "n_gear"].item() == 3

    def test_brake_edges_land_on_the_first_grid_point_at_or_after_the_source_edge(
        self, grid: pd.DataFrame
    ) -> None:
        brake = grid["brake"].to_numpy(dtype=bool)
        d = grid["distance_m"].to_numpy(dtype=float)
        rising = d[1:][brake[1:] & ~brake[:-1]]
        assert rising.tolist() == [200.0, 600.0]

    def test_elapsed_time_strictly_increasing(self, grid: pd.DataFrame) -> None:
        assert (np.diff(grid["elapsed_time"].to_numpy(dtype=float)) > 0).all()

    def test_source_gap_reports_bracketing_spacing(self, grid: pd.DataFrame) -> None:
        d = grid["distance_m"].to_numpy(dtype=float)
        gap = grid["source_gap_m"].to_numpy(dtype=float)
        on_sample = np.isclose(d % SOURCE_STEP_M, 0.0)
        assert np.all(gap[on_sample] == 0.0)
        assert np.all(gap[~on_sample] == SOURCE_STEP_M)


class TestPurity:
    def test_input_is_not_modified(self) -> None:
        lap = synthetic_lap()
        before = lap.copy(deep=True)
        resample_lap(lap)
        pd.testing.assert_frame_equal(lap, before)

    def test_deterministic(self) -> None:
        lap = synthetic_lap()
        pd.testing.assert_frame_equal(resample_lap(lap), resample_lap(lap.copy()))

    def test_row_order_of_input_does_not_matter_for_a_sorted_lap(self) -> None:
        lap = synthetic_lap()
        shuffled = lap.sample(frac=1.0, random_state=0).sort_values("session_time")
        pd.testing.assert_frame_equal(resample_lap(lap), resample_lap(shuffled.reset_index(drop=True)))


class TestEdgeCases:
    def test_axis_starting_after_zero_holds_the_first_sample_at_grid_zero(self) -> None:
        """Real laps open ~2 m in; grid point 0 is before any sample."""
        lap = synthetic_lap(offset_m=2.0)
        grid = resample_lap(lap)
        assert len(grid) == grid_point_count(1002.0)
        first = grid.iloc[0]
        assert first["speed"] == pytest.approx(lap["speed"].iloc[0])
        assert first["elapsed_time"] == 0.0
        assert first["n_gear"] == lap["n_gear"].iloc[0]
        assert np.isnan(first["source_gap_m"])
        assert np.isfinite(grid["source_gap_m"].to_numpy(dtype=float)[1:]).all()

    def test_duplicate_distances_take_the_last_sample_of_the_run(self) -> None:
        """F008's monotonic guard leaves flat runs; np.interp cannot take them raw."""
        lap = synthetic_lap()
        # Three samples pinned to the same distance, the last in a different gear.
        lap.loc[50:52, "distance_aligned"] = 200.0
        lap.loc[52, "n_gear"] = 8
        grid = resample_lap(lap)
        assert len(grid) == 101
        assert grid.loc[grid["distance_m"] == 200.0, "n_gear"].item() == 8
        assert np.isfinite(grid["speed"].to_numpy(dtype=float)).all()

    def test_custom_spacing(self) -> None:
        grid = resample_lap(synthetic_lap(), grid_m=25.0)
        assert len(grid) == 41
        assert grid["distance_m"].iloc[-1] == 1000.0

    def test_nullable_discrete_values_are_carried_not_filled(self) -> None:
        lap = synthetic_lap()
        lap.loc[10:12, "drs"] = pd.NA
        grid = resample_lap(lap)
        assert str(grid["drs"].dtype) == "Int8"
        assert grid["drs"].isna().any()


class TestRejection:
    def test_too_few_samples(self) -> None:
        with pytest.raises(ResampleError, match="at least 2"):
            resample_lap(synthetic_lap().iloc[:1])

    def test_missing_channel(self) -> None:
        with pytest.raises(ResampleError, match="missing columns"):
            resample_lap(synthetic_lap().drop(columns=["rpm"]))

    def test_decreasing_distance(self) -> None:
        lap = synthetic_lap()
        lap.loc[30, "distance_aligned"] = 0.0
        with pytest.raises(ResampleError, match="non-decreasing"):
            resample_lap(lap)

    def test_non_finite_distance(self) -> None:
        lap = synthetic_lap()
        lap.loc[30, "distance_aligned"] = np.nan
        with pytest.raises(ResampleError, match="non-finite"):
            resample_lap(lap)

    def test_frame_spanning_two_laps(self) -> None:
        lap = synthetic_lap()
        lap.loc[100:, "lap_number"] = 8
        with pytest.raises(ResampleError, match="more than one lap_number"):
            resample_lap(lap)

    def test_single_distance_value(self) -> None:
        lap = synthetic_lap().iloc[:3].copy()
        lap["distance_aligned"] = 5.0
        with pytest.raises(ResampleError, match="single distance"):
            resample_lap(lap)
