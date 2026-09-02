"""F003 acceptance measurements, checked on synthetic laps where the truth is known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.grid import validation
from src.grid.resample import resample_lap
from tests.test_resample import synthetic_lap


@pytest.fixture()
def lap() -> pd.DataFrame:
    return synthetic_lap()


@pytest.fixture()
def grid(lap: pd.DataFrame) -> pd.DataFrame:
    return resample_lap(lap)


class TestGridStructure:
    def test_a_correct_grid_passes(self, grid: pd.DataFrame) -> None:
        assert validation.check_grid_structure(grid, 1000.0)

    def test_a_dropped_row_fails(self, grid: pd.DataFrame) -> None:
        assert not validation.check_grid_structure(grid.drop(index=5).reset_index(drop=True), 1000.0)

    def test_a_perturbed_distance_fails(self, grid: pd.DataFrame) -> None:
        broken = grid.copy()
        broken.loc[5, "distance_m"] = 51.0
        assert not validation.check_grid_structure(broken, 1000.0)

    def test_wrong_lap_length_fails(self, grid: pd.DataFrame) -> None:
        assert not validation.check_grid_structure(grid, 1010.0)


class TestRoundTrip:
    def test_linear_channel_round_trips_exactly(self, lap: pd.DataFrame, grid: pd.DataFrame) -> None:
        for channel in ("speed", "throttle", "rpm"):
            err = validation.round_trip_error(lap, grid, channel)
            assert len(err) == len(lap)
            assert err.max() < 1e-2, channel

    def test_curvature_between_grid_points_shows_up_as_error(self, lap: pd.DataFrame) -> None:
        """A quadratic cannot be represented by 10 m linear pieces sampled at 4 m."""
        curved = lap.copy()
        d = curved["distance_aligned"].to_numpy(dtype=float)
        curved["speed"] = 100.0 + 50.0 * np.sin(d / 15.0)
        err = validation.round_trip_error(curved, resample_lap(curved), "speed")
        assert err.max() > 0.5

    def test_samples_beyond_the_grid_are_ignored(self, lap: pd.DataFrame, grid: pd.DataFrame) -> None:
        extended = pd.concat([lap, lap.iloc[[-1]].assign(distance_aligned=1005.0)], ignore_index=True)
        assert len(validation.round_trip_error(extended, grid, "speed")) == len(lap)


class TestDiscrete:
    def test_no_invented_values_on_a_faithful_grid(self, lap: pd.DataFrame, grid: pd.DataFrame) -> None:
        for channel in ("n_gear", "brake", "drs"):
            assert validation.invented_values(lap, grid, channel) == set()

    def test_a_fabricated_gear_is_detected(self, lap: pd.DataFrame, grid: pd.DataFrame) -> None:
        broken = grid.copy()
        broken.loc[3, "n_gear"] = 5  # the lap only reaches gear 8 via 1..8, but 5 exists
        broken.loc[4, "n_gear"] = 9
        assert validation.invented_values(lap, broken, "n_gear") == {9}


class TestBrakeEdges:
    def test_edges_on_grid_points_have_zero_displacement(
        self, lap: pd.DataFrame, grid: pd.DataFrame
    ) -> None:
        disp = validation.brake_edge_displacement(lap, grid)
        assert disp.tolist() == [0.0, 0.0]

    def test_edge_between_grid_points_moves_to_the_next_point(self, lap: pd.DataFrame) -> None:
        shifted = lap.copy()
        d = shifted["distance_aligned"].to_numpy(dtype=float)
        shifted["brake"] = pd.array((d >= 204) & (d < 300), dtype="boolean")
        disp = validation.brake_edge_displacement(shifted, resample_lap(shifted))
        assert disp.tolist() == [pytest.approx(6.0)]

    def test_no_source_edges_gives_empty(self, lap: pd.DataFrame) -> None:
        quiet = lap.copy()
        quiet["brake"] = pd.array([False] * len(quiet), dtype="boolean")
        assert len(validation.brake_edge_displacement(quiet, resample_lap(quiet))) == 0

    def test_pulse_too_short_for_the_grid_is_nan_not_hidden(self, lap: pd.DataFrame) -> None:
        pulse = lap.copy()
        d = pulse["distance_aligned"].to_numpy(dtype=float)
        # 4 m of braking between grid points 40 and 50: never lands on the grid.
        pulse["brake"] = pd.array((d >= 44) & (d < 48), dtype="boolean")
        disp = validation.brake_edge_displacement(pulse, resample_lap(pulse))
        assert len(disp) == 1 and np.isnan(disp[0])


class TestElapsedAndBins:
    def test_elapsed_time_check(self, grid: pd.DataFrame) -> None:
        assert validation.elapsed_time_strictly_increasing(grid)
        flat = grid.copy()
        flat.loc[5, "elapsed_time"] = flat.loc[4, "elapsed_time"]
        assert not validation.elapsed_time_strictly_increasing(flat)

    def test_dense_source_leaves_no_empty_bins(self, lap: pd.DataFrame) -> None:
        assert validation.empty_bin_fraction(lap) == 0.0

    def test_sparse_source_leaves_empty_bins(self) -> None:
        sparse = synthetic_lap(step_m=25.0)  # 41 samples over 101 bins
        assert validation.empty_bin_fraction(sparse) == pytest.approx(1.0 - 41 / 101)


class TestReport:
    def test_synthetic_session_passes_every_gate(self, lap: pd.DataFrame, grid: pd.DataFrame) -> None:
        report = validation.measure_session([(lap, grid), (lap, grid)])
        assert report.laps == 2
        assert report.ok
        assert report.speed.p95 < 1e-2
        assert report.invented == {"n_gear": 0, "brake": 0, "drs": 0}
        assert report.brake_edge.max == 0.0
        assert report.empty_bin_fraction == 0.0
        # Every second grid point sits exactly on a 4 m source sample (gap 0),
        # the rest are bracketed by one 4 m interval.
        assert report.source_gap_median_m == 0.0
        assert report.source_gap.max == pytest.approx(4.0)

    def test_report_fails_when_a_gate_fails(self, lap: pd.DataFrame, grid: pd.DataFrame) -> None:
        broken = grid.copy()
        broken.loc[4, "n_gear"] = 9
        report = validation.measure_session([(lap, broken)])
        assert not report.discrete_ok
        assert not report.ok
        assert report.to_dict()["checks"]["discrete_no_invented_values"] is False

    def test_to_dict_is_json_ready(self, lap: pd.DataFrame, grid: pd.DataFrame) -> None:
        import json

        payload = validation.measure_session([(lap, grid)]).to_dict()
        json.dumps(payload)
        assert payload["checks"]["all"] is True
