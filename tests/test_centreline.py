"""Reference-line projection, including the start/finish wrap it must survive."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.align.centreline import (
    build_reference_line,
    enforce_monotonic,
    project_lap,
    project_onto_line,
)

RADIUS_M = 300.0
CIRCUMFERENCE_M = 2 * np.pi * RADIUS_M


def _ring(n: int = 800, radius: float = RADIUS_M, start: float = 0.0) -> np.ndarray:
    """A closed lap: the last point returns to the first.

    Closing it matters -- an open ring never brings the end back alongside the
    start, so it cannot exercise the start/finish wrap these tests exist to
    check.
    """
    theta = np.linspace(0.0, 2 * np.pi, n + 1, endpoint=True) + start
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def _lap_frame(xy: np.ndarray, driver: str = "SYN") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver": [driver] * len(xy),
            "lap_number": [1] * len(xy),
            "session_time": np.linspace(0.0, 90.0, len(xy)),
            "speed": np.full(len(xy), 200.0),
            "x": xy[:, 0],
            "y": xy[:, 1],
        }
    )


class TestReferenceLine:
    def test_length_matches_the_geometry(self) -> None:
        line = build_reference_line(_ring())
        assert line.total_length_m == pytest.approx(CIRCUMFERENCE_M, rel=1e-3)

    def test_resamples_to_roughly_uniform_spacing(self) -> None:
        line = build_reference_line(_ring(), spacing_m=2.0)
        steps = np.hypot(*np.diff(line.xy, axis=0).T)
        assert steps.std() < 0.05
        assert steps.mean() == pytest.approx(2.0, abs=0.1)

    def test_arc_zero_is_the_seed_lap_s_own_start(self) -> None:
        """The lead-in carries negative arc, so zero stays where the seed lap's
        telemetry began -- which is what the resampler grids from."""
        line = build_reference_line(_ring())
        assert line.arc_length_m[0] == pytest.approx(-line.lead_in_m)
        assert line.start_m < 0.0
        assert np.all(np.diff(line.arc_length_m) > 0)
        origin = int(np.argmin(np.abs(line.arc_length_m)))
        assert line.arc_length_m[origin] == pytest.approx(0.0, abs=1.0)

    def test_without_a_lead_in_arc_starts_at_zero(self) -> None:
        line = build_reference_line(_ring(), lead_m=0.0)
        assert line.arc_length_m[0] == 0.0
        assert line.lead_in_m == 0.0

    def test_rejects_a_degenerate_path(self) -> None:
        with pytest.raises(ValueError, match=">=10 points"):
            build_reference_line(_ring(5))


class TestProjection:
    def test_a_lap_projects_onto_its_own_line_as_arc_length(self) -> None:
        xy = _ring()
        line = build_reference_line(xy)
        distance, offset = project_onto_line(xy, line)

        assert offset.max() < 0.5
        assert distance[0] == pytest.approx(0.0, abs=2.0)
        assert distance[-1] == pytest.approx(CIRCUMFERENCE_M, rel=5e-3)

    def test_an_offset_lap_keeps_the_same_arc_coordinate(self) -> None:
        """A wider line is a different path, but the same place along the track."""
        line = build_reference_line(_ring())
        wide = _ring(radius=RADIUS_M + 8.0)
        distance, offset = project_onto_line(wide, line)

        assert offset.mean() == pytest.approx(8.0, abs=0.5)
        # Same angular position => same arc length, within a sample step.
        expected = np.linspace(0.0, CIRCUMFERENCE_M, len(wide), endpoint=False)
        assert np.abs(distance - expected).max() < 5.0

    def test_two_laps_register_together(self) -> None:
        line = build_reference_line(_ring())
        a, _ = project_onto_line(_ring(n=800), line)
        b, _ = project_onto_line(_ring(n=650, radius=RADIUS_M + 5.0), line)

        # Compare at matching angular positions.
        for fraction in (0.25, 0.5, 0.75):
            ia, ib = int(fraction * len(a)), int(fraction * len(b))
            assert abs(a[ia] - b[ib]) < 10.0

    def test_start_finish_wrap_does_not_split_the_lap(self) -> None:
        """The line's first and last vertices are adjacent in space, a lap apart
        in arc. A global nearest search assigns some samples ~0 and others ~L."""
        line = build_reference_line(_ring())
        distance, _ = project_onto_line(_ring(), line, ordered=True)

        jumps = np.abs(np.diff(distance))
        assert jumps.max() < CIRCUMFERENCE_M / 4

    def test_lap_starting_just_before_the_line_does_not_read_a_full_lap(self) -> None:
        """The real failure mode: a lap's telemetry window opens a few metres
        before the start/finish line, so its first samples sit nearest the far
        END of the reference line. Seeded globally, such a lap reported ~5722 m
        for its whole length and destroyed the axis."""
        line = build_reference_line(_ring())
        early = _ring(start=-0.02)  # begins ~6 m before the line's own start

        ordered, _ = project_onto_line(early, line, ordered=True)
        unordered, _ = project_onto_line(early, line, ordered=False)

        assert ordered[0] < 50.0, "ordered search must not start a lap near the finish"
        assert unordered[0] > line.total_length_m - 50.0, (
            "an unordered search is expected to mis-seed here; if it no longer "
            "does, this guard is testing nothing"
        )


class TestMonotonic:
    def test_clamps_backward_steps_and_counts_them(self) -> None:
        corrected, fixed = enforce_monotonic(np.array([0.0, 10.0, 8.0, 20.0]))
        assert corrected.tolist() == [0.0, 10.0, 10.0, 20.0]
        assert fixed == 1

    def test_leaves_a_clean_axis_untouched(self) -> None:
        values = np.array([0.0, 5.0, 10.0])
        corrected, fixed = enforce_monotonic(values)
        assert corrected.tolist() == values.tolist()
        assert fixed == 0


class TestProjectLap:
    def test_adds_the_expected_columns_without_mutating_input(self) -> None:
        line = build_reference_line(_ring())
        frame = _lap_frame(_ring())
        out = project_lap(frame, line)

        assert "distance_aligned" in out.columns
        assert "line_offset_m" in out.columns
        assert "distance_aligned" not in frame.columns

    def test_distance_is_non_decreasing(self) -> None:
        line = build_reference_line(_ring())
        out = project_lap(_lap_frame(_ring()), line)
        assert np.all(np.diff(out["distance_aligned"].to_numpy()) >= 0)
