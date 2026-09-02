"""Time-domain channel merge and raw distance integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.align.merge import MergeError, add_raw_distance, build_lap_frame, merge_lap_channels


def _car(times, speeds=None, gears=None, brakes=None) -> pd.DataFrame:
    n = len(times)
    return pd.DataFrame(
        {
            "driver": ["VER"] * n,
            "lap_number": [1] * n,
            "session_time": times,
            "speed": speeds if speeds is not None else [100.0] * n,
            "throttle": [100.0] * n,
            "rpm": [11000.0] * n,
            "n_gear": gears if gears is not None else [7] * n,
            "brake": brakes if brakes is not None else [False] * n,
            "drs": [0] * n,
        }
    )


def _pos(times, xs=None, ys=None) -> pd.DataFrame:
    n = len(times)
    return pd.DataFrame(
        {
            "driver": ["VER"] * n,
            "lap_number": [1] * n,
            "session_time": times,
            "x": xs if xs is not None else [float(i) for i in range(n)],
            "y": ys if ys is not None else [0.0] * n,
            "z": [0.0] * n,
        }
    )


class TestMergeTimebase:
    def test_output_is_the_union_of_both_timebases(self) -> None:
        merged = merge_lap_channels(_car([0.0, 1.0, 2.0]), _pos([0.0, 0.5, 1.5, 2.0]))
        assert merged["session_time"].tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]

    def test_clipped_to_overlap_so_position_is_never_extrapolated(self) -> None:
        """Car data runs longer than position data; the tail must be dropped."""
        merged = merge_lap_channels(_car([0.0, 1.0, 2.0, 3.0]), _pos([0.5, 1.0, 1.5]))
        assert merged["session_time"].min() == 0.5
        assert merged["session_time"].max() == 1.5

    def test_continuous_channels_are_linearly_interpolated(self) -> None:
        merged = merge_lap_channels(
            _car([0.0, 2.0], speeds=[100.0, 200.0]), _pos([0.0, 1.0, 2.0])
        )
        at_one = merged.loc[merged["session_time"] == 1.0, "speed"].iloc[0]
        assert at_one == pytest.approx(150.0)

    def test_discrete_channels_use_step_not_linear(self) -> None:
        """A gear of 6.5 is not a physical state (project convention)."""
        merged = merge_lap_channels(
            _car([0.0, 2.0], gears=[6, 7], brakes=[True, False]), _pos([0.0, 1.0, 2.0])
        )
        row = merged.loc[merged["session_time"] == 1.0].iloc[0]
        assert row["n_gear"] == 6
        assert bool(row["brake"]) is True

    def test_position_is_interpolated_onto_car_samples(self) -> None:
        merged = merge_lap_channels(_car([0.0, 1.0, 2.0]), _pos([0.0, 2.0], xs=[0.0, 20.0]))
        at_one = merged.loc[merged["session_time"] == 1.0, "x"].iloc[0]
        # Midpoint of 0..20 FastF1 units = 10 units = 1.0 m after conversion.
        assert at_one == pytest.approx(1.0)

    def test_positions_are_converted_to_metres(self) -> None:
        """FastF1 delivers X/Y in 1/10 m; corner distances are in metres."""
        merged = merge_lap_channels(_car([0.0, 1.0]), _pos([0.0, 1.0], xs=[0.0, 100.0]))
        assert merged["x"].iloc[-1] == pytest.approx(10.0)

    def test_identity_columns_are_carried(self) -> None:
        merged = merge_lap_channels(_car([0.0, 1.0]), _pos([0.0, 1.0]))
        assert set(merged["driver"]) == {"VER"}
        assert set(merged["lap_number"]) == {1}


class TestMergeFailures:
    def test_non_overlapping_windows_are_rejected(self) -> None:
        with pytest.raises(MergeError, match="do not overlap"):
            merge_lap_channels(_car([0.0, 1.0]), _pos([5.0, 6.0]))

    def test_too_few_samples_is_rejected(self) -> None:
        with pytest.raises(MergeError, match=">=2 samples"):
            merge_lap_channels(_car([0.0]), _pos([0.0, 1.0]))


class TestRawDistance:
    def test_constant_speed_integrates_to_speed_times_time(self) -> None:
        # 180 km/h = 50 m/s, over 10 s = 500 m
        frame = pd.DataFrame({"session_time": [0.0, 10.0], "speed": [180.0, 180.0]})
        out = add_raw_distance(frame)
        assert out["distance_raw"].iloc[-1] == pytest.approx(500.0)

    def test_starts_at_zero(self) -> None:
        frame = pd.DataFrame({"session_time": [3.0, 4.0], "speed": [100.0, 120.0]})
        assert add_raw_distance(frame)["distance_raw"].iloc[0] == 0.0

    def test_linear_ramp_uses_the_trapezoid_not_the_endpoint(self) -> None:
        # 0 -> 100 m/s over 10 s: trapezoid gives 500 m, not 1000 m.
        frame = pd.DataFrame({"session_time": [0.0, 10.0], "speed": [0.0, 360.0]})
        assert add_raw_distance(frame)["distance_raw"].iloc[-1] == pytest.approx(500.0)

    def test_is_monotonically_non_decreasing(self) -> None:
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {"session_time": np.arange(0.0, 20.0, 0.25), "speed": rng.uniform(50, 320, 80)}
        )
        distance = add_raw_distance(frame)["distance_raw"].to_numpy()
        assert np.all(np.diff(distance) >= 0)

    def test_rejects_unsorted_time(self) -> None:
        frame = pd.DataFrame({"session_time": [0.0, 2.0, 1.0], "speed": [100.0] * 3})
        with pytest.raises(MergeError, match="monotonically increasing"):
            add_raw_distance(frame)

    def test_does_not_mutate_its_input(self) -> None:
        frame = pd.DataFrame({"session_time": [0.0, 1.0], "speed": [100.0, 100.0]})
        add_raw_distance(frame)
        assert "distance_raw" not in frame.columns


class TestBuildLapFrame:
    def test_produces_a_frame_ready_for_alignment(self) -> None:
        frame = build_lap_frame(_car([0.0, 1.0, 2.0]), _pos([0.0, 1.0, 2.0]))
        for column in ("driver", "lap_number", "session_time", "speed", "x", "y", "distance_raw"):
            assert column in frame.columns
        assert frame["distance_raw"].iloc[0] == 0.0
