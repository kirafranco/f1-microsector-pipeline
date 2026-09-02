"""Corner-anchored alignment, against a synthetic circuit with known geometry.

The synthetic lap carries a deliberate 3% distance distortion — the kind a
speed-integrated axis really has. Alignment must recover the true distances at
corners it never used as anchors. If it cannot do that here, it will not do it
at Suzuka.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.align.track_reference import (
    AlignedLap,
    TrackReference,
    align_lap,
    locate_anchors,
    split_anchor_and_holdout,
)

CIRCUMFERENCE_M = 1000.0
RADIUS_M = CIRCUMFERENCE_M / (2 * np.pi)
N_CORNERS = 12
DISTORTION = 1.03  # speed reads 3% high, so the raw axis runs long


def _xy(arc_length: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = arc_length / RADIUS_M
    return RADIUS_M * np.cos(theta), RADIUS_M * np.sin(theta)


def _reference(n_corners: int = N_CORNERS) -> TrackReference:
    distances = np.linspace(0.0, CIRCUMFERENCE_M, n_corners, endpoint=False) + 40.0
    xs, ys = _xy(distances)
    corners = pd.DataFrame(
        {
            "number": np.arange(1, n_corners + 1),
            "letter": [""] * n_corners,
            "x": xs,
            "y": ys,
            "distance": distances,
        }
    )
    return TrackReference(circuit="Synthetic", corners=corners, lap_length_m=CIRCUMFERENCE_M)


def _lap(distortion: float = DISTORTION, n_samples: int = 400, offset_m: float = 0.0) -> pd.DataFrame:
    """A lap whose raw distance axis is stretched by `distortion`."""
    true_arc = np.linspace(0.0, CIRCUMFERENCE_M, n_samples)
    xs, ys = _xy(true_arc + offset_m)
    return pd.DataFrame(
        {
            "driver": ["SYN"] * n_samples,
            "lap_number": [1] * n_samples,
            "session_time": np.linspace(0.0, 60.0, n_samples),
            "speed": np.full(n_samples, 200.0),
            "x": xs,
            "y": ys,
            "distance_raw": true_arc * distortion,
            "true_arc": true_arc,
        }
    )


class TestAnchorLocation:
    def test_finds_every_corner_in_track_order(self) -> None:
        anchors = locate_anchors(_lap(), _reference().subset(None))
        assert len(anchors) == N_CORNERS
        assert anchors["corner_number"].tolist() == list(range(1, N_CORNERS + 1))
        assert anchors["d_raw"].is_monotonic_increasing

    def test_residuals_are_small_on_a_clean_lap(self) -> None:
        anchors = locate_anchors(_lap(), _reference().subset(None))
        # Sample spacing is 1000/399 ~ 2.5 m, so the nearest sample is within half that.
        assert anchors["residual_m"].max() < 2.0

    def test_corner_far_from_the_track_is_dropped(self) -> None:
        reference = _reference()
        corners = reference.subset(None).copy()
        corners.loc[3, "x"] = 10_000.0  # nowhere near the circuit
        anchors = locate_anchors(_lap(), corners, max_residual_m=50.0)
        assert 4 not in anchors["corner_number"].tolist()
        assert len(anchors) == N_CORNERS - 1

    def test_search_window_only_moves_forward(self) -> None:
        """Two corners cannot match the same sample; the later one is dropped."""
        reference = _reference()
        corners = reference.subset(None).copy()
        corners.loc[5, ["x", "y"]] = corners.loc[4, ["x", "y"]].to_numpy()
        anchors = locate_anchors(_lap(), corners)
        assert anchors["d_raw"].is_monotonic_increasing
        assert anchors["d_raw"].is_unique


class TestAlignment:
    def test_recovers_true_distance_at_held_out_corners(self) -> None:
        """The criterion that matters: corners the algorithm never saw."""
        reference = _reference()
        anchor_numbers, holdout_numbers = split_anchor_and_holdout(reference)
        lap = _lap()

        result = align_lap(lap, reference, min_anchors=4, anchor_corner_numbers=anchor_numbers)
        assert not result.rejected

        aligned = result.telemetry
        held_out = reference.corners[reference.corners["number"].isin(holdout_numbers)]

        errors = []
        for corner in held_out.itertuples(index=False):
            dx = aligned["x"].to_numpy() - corner.x
            dy = aligned["y"].to_numpy() - corner.y
            index = int(np.argmin(dx * dx + dy * dy))
            errors.append(abs(aligned["distance_aligned"].iloc[index] - corner.distance))

        assert max(errors) < 5.0, f"held-out registration error {max(errors):.2f} m"

    def test_raw_axis_is_materially_worse_than_the_aligned_one(self) -> None:
        """Evidence that alignment did something, not just that it ran."""
        reference = _reference()
        anchor_numbers, holdout_numbers = split_anchor_and_holdout(reference)
        lap = _lap()
        aligned = align_lap(
            lap, reference, min_anchors=4, anchor_corner_numbers=anchor_numbers
        ).telemetry

        held_out = reference.corners[reference.corners["number"].isin(holdout_numbers)]
        raw_errors, aligned_errors = [], []
        for corner in held_out.itertuples(index=False):
            dx = aligned["x"].to_numpy() - corner.x
            dy = aligned["y"].to_numpy() - corner.y
            index = int(np.argmin(dx * dx + dy * dy))
            raw_errors.append(abs(aligned["distance_raw"].iloc[index] - corner.distance))
            aligned_errors.append(abs(aligned["distance_aligned"].iloc[index] - corner.distance))

        assert max(aligned_errors) < max(raw_errors) / 5

    def test_lap_length_is_measured_not_imposed(self) -> None:
        """No endpoint anchoring: total distance must fall out within 0.5%."""
        reference = _reference()
        anchor_numbers, _ = split_anchor_and_holdout(reference)
        result = align_lap(_lap(), reference, min_anchors=4, anchor_corner_numbers=anchor_numbers)

        total = float(result.telemetry["distance_aligned"].iloc[-1])
        error_pct = 100 * abs(total - CIRCUMFERENCE_M) / CIRCUMFERENCE_M
        assert error_pct < 0.5, f"lap length {total:.1f} m, {error_pct:.3f}% off"

    def test_aligned_distance_is_strictly_increasing(self) -> None:
        result = align_lap(_lap(), _reference(), min_anchors=4)
        assert np.all(np.diff(result.telemetry["distance_aligned"].to_numpy()) > 0)

    def test_start_of_lap_extrapolates_back_towards_zero(self) -> None:
        """np.interp would clamp here, flattening the run to turn one."""
        result = align_lap(_lap(), _reference(), min_anchors=4)
        assert result.telemetry["distance_aligned"].iloc[0] == pytest.approx(0.0, abs=5.0)

    def test_two_drivers_on_different_lines_register_together(self) -> None:
        reference = _reference()
        anchor_numbers, holdout_numbers = split_anchor_and_holdout(reference)

        a = align_lap(_lap(distortion=1.03), reference, min_anchors=4,
                      anchor_corner_numbers=anchor_numbers).telemetry
        b = align_lap(_lap(distortion=0.97, n_samples=350), reference, min_anchors=4,
                      anchor_corner_numbers=anchor_numbers).telemetry

        held_out = reference.corners[reference.corners["number"].isin(holdout_numbers)]
        for corner in held_out.itertuples(index=False):
            positions = []
            for frame in (a, b):
                dx = frame["x"].to_numpy() - corner.x
                dy = frame["y"].to_numpy() - corner.y
                index = int(np.argmin(dx * dx + dy * dy))
                positions.append(frame["distance_aligned"].iloc[index])
            assert abs(positions[0] - positions[1]) < 10.0


class TestRejection:
    def test_too_few_anchors_is_rejected_not_silently_rescaled(self) -> None:
        reference = _reference(n_corners=4)
        result = align_lap(_lap(), reference, min_anchors=8)
        assert result.rejected
        assert "minimum is 8" in result.reject_reason
        assert result.telemetry.empty

    def test_rejection_still_reports_the_anchors_it_found(self) -> None:
        result = align_lap(_lap(), _reference(n_corners=4), min_anchors=8)
        assert len(result.anchors) == 4


class TestPurity:
    def test_is_deterministic(self) -> None:
        lap, reference = _lap(), _reference()
        first = align_lap(lap, reference, min_anchors=4)
        second = align_lap(lap, reference, min_anchors=4)
        pd.testing.assert_frame_equal(first.telemetry, second.telemetry)
        pd.testing.assert_frame_equal(first.anchors, second.anchors)

    def test_does_not_mutate_the_input_frame(self) -> None:
        lap = _lap()
        before = lap.copy()
        align_lap(lap, _reference(), min_anchors=4)
        assert "distance_aligned" not in lap.columns
        pd.testing.assert_frame_equal(lap, before)


class TestSplit:
    def test_odd_corners_anchor_even_corners_validate(self) -> None:
        anchors, holdout = split_anchor_and_holdout(_reference(n_corners=18))
        assert anchors == list(range(1, 19, 2))
        assert holdout == list(range(2, 19, 2))
        assert not set(anchors) & set(holdout)

    def test_suzuka_sized_circuit_clears_the_anchor_floor(self) -> None:
        anchors, holdout = split_anchor_and_holdout(_reference(n_corners=18))
        assert len(anchors) >= 8
        assert len(holdout) >= 8


class TestTrackReference:
    def test_rejects_corners_without_required_columns(self) -> None:
        corners = pd.DataFrame({"number": [1, 2], "x": [0.0, 1.0]})
        with pytest.raises(ValueError, match="missing columns"):
            TrackReference(circuit="X", corners=corners, lap_length_m=1000.0)

    def test_subset_orders_by_reference_distance(self) -> None:
        reference = _reference()
        subset = reference.subset([5, 1, 3])
        assert subset["distance"].is_monotonic_increasing
        assert subset["number"].tolist() == [1, 3, 5]
