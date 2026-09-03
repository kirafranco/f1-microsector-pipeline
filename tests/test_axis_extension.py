"""F010 Part A: the axis continues past the reference line's ends instead of clamping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.align.centreline import build_reference_line, extend_axis_ends, project_lap

LINE_LENGTH_M = 1000.0


def clamped_frame(lead: int = 3, trail: int = 2, step_m: float = 10.0) -> pd.DataFrame:
    """A lap whose first `lead` and last `trail` samples were clamped by projection.

    Raw distance advances uniformly; the true aligned positions of the clamped
    samples are therefore known exactly: negative before the line start and
    beyond the line length after its end.
    """
    n = int(LINE_LENGTH_M / step_m) + 1 + lead + trail
    raw = np.arange(n, dtype=float) * step_m
    true = raw - lead * step_m
    aligned = np.clip(true, 0.0, LINE_LENGTH_M)
    return pd.DataFrame(
        {
            "driver": "SYN",
            "lap_number": 1,
            "session_time": np.arange(n, dtype=float) * 0.2,
            "distance_raw": raw,
            "distance_aligned": aligned,
            "true_distance": true,
        }
    )


class TestExtendAxisEnds:
    def test_recovers_the_true_positions_at_both_ends(self) -> None:
        frame = clamped_frame()
        out = extend_axis_ends(frame, LINE_LENGTH_M)
        np.testing.assert_allclose(out["distance_aligned"], frame["true_distance"])
        assert out["distance_aligned"].iloc[0] == -30.0
        assert out["distance_aligned"].iloc[-1] == LINE_LENGTH_M + 20.0

    def test_result_is_strictly_increasing(self) -> None:
        out = extend_axis_ends(clamped_frame(), LINE_LENGTH_M)
        assert (np.diff(out["distance_aligned"].to_numpy()) > 0).all()

    def test_unclamped_frame_is_unchanged(self) -> None:
        frame = clamped_frame(lead=0, trail=0)
        frame["distance_aligned"] = frame["true_distance"] + 2.0  # starts at 2 m, ends short
        out = extend_axis_ends(frame, LINE_LENGTH_M)
        pd.testing.assert_series_equal(out["distance_aligned"], frame["distance_aligned"])

    def test_only_one_end_clamped(self) -> None:
        lead_only = extend_axis_ends(clamped_frame(lead=4, trail=0), LINE_LENGTH_M)
        assert lead_only["distance_aligned"].iloc[0] == -40.0
        assert lead_only["distance_aligned"].iloc[-1] == LINE_LENGTH_M
        trail_only = extend_axis_ends(clamped_frame(lead=0, trail=3), LINE_LENGTH_M)
        assert trail_only["distance_aligned"].iloc[0] == 0.0
        assert trail_only["distance_aligned"].iloc[-1] == LINE_LENGTH_M + 30.0

    def test_interior_flat_runs_are_left_alone(self) -> None:
        frame = clamped_frame(lead=0, trail=0)
        frame.loc[40:42, "distance_aligned"] = frame.loc[40, "distance_aligned"]
        out = extend_axis_ends(frame, LINE_LENGTH_M)
        pd.testing.assert_series_equal(out["distance_aligned"], frame["distance_aligned"])

    def test_non_uniform_raw_spacing_is_honoured(self) -> None:
        frame = clamped_frame(lead=2, trail=0)
        frame.loc[0, "distance_raw"] = 3.0  # the first sample was 7 m before the second, not 10
        out = extend_axis_ends(frame, LINE_LENGTH_M)
        assert out["distance_aligned"].iloc[0] == pytest.approx(-17.0)
        assert out["distance_aligned"].iloc[1] == pytest.approx(-10.0)

    def test_without_raw_distance_or_too_short_returns_input(self) -> None:
        frame = clamped_frame().drop(columns=["distance_raw"])
        assert extend_axis_ends(frame, LINE_LENGTH_M) is frame
        one = clamped_frame().iloc[:1]
        assert extend_axis_ends(one, LINE_LENGTH_M) is one

    def test_fully_clamped_frame_is_unchanged(self) -> None:
        frame = clamped_frame(lead=5, trail=0).iloc[:5]
        out = extend_axis_ends(frame, LINE_LENGTH_M)
        assert (out["distance_aligned"] == 0.0).all()

    def test_input_not_mutated(self) -> None:
        frame = clamped_frame()
        before = frame.copy()
        extend_axis_ends(frame, LINE_LENGTH_M)
        pd.testing.assert_frame_equal(frame, before)


class TestProjectLapIntegration:
    def test_a_lap_overrunning_the_line_extends_past_it(self) -> None:
        radius = 300.0
        theta = np.linspace(0.0, 2 * np.pi, 801)
        line = build_reference_line(np.column_stack([radius * np.cos(theta), radius * np.sin(theta)]))
        # The lap opens 0.05 rad before the line's start and closes 0.05 rad after its end.
        theta_lap = np.linspace(-0.05, 2 * np.pi + 0.05, 830)
        xy = np.column_stack([radius * np.cos(theta_lap), radius * np.sin(theta_lap)])
        frame = pd.DataFrame(
            {
                "driver": "SYN", "lap_number": 1, "session_time": np.linspace(0.0, 90.0, len(xy)),
                "speed": 200.0, "x": xy[:, 0], "y": xy[:, 1], "distance_raw": radius * (theta_lap - theta_lap[0]),
            }
        )
        out = project_lap(frame, line)
        d = out["distance_aligned"].to_numpy()
        assert d[0] == pytest.approx(-radius * 0.05, abs=1.0)
        assert d[-1] == pytest.approx(line.total_length_m + radius * 0.05, abs=1.0)
        assert (np.diff(d) >= 0).all()
        assert (d == 0.0).sum() <= 1 and (d == line.total_length_m).sum() <= 1
