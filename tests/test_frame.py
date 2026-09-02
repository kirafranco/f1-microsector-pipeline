"""Frame calibration between corner points and position telemetry."""

from __future__ import annotations

import numpy as np
import pytest

from src.align.frame import FrameFitError, RigidTransform, fit_corner_frame

RADIUS_M = 500.0


def _ring(n: int = 4000) -> np.ndarray:
    """Dense enough that vertex spacing (~0.8 m) is not the measurement floor.

    fit_corner_frame matches against vertices, so residuals cannot go below half
    the vertex spacing. On real data the cloud is many laps stacked together and
    is correspondingly dense.
    """
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([RADIUS_M * np.cos(theta), RADIUS_M * np.sin(theta)])


def _markers(n: int = 9) -> np.ndarray:
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([RADIUS_M * np.cos(theta), RADIUS_M * np.sin(theta)])


def _shift(points: np.ndarray, dx: float, dy: float, deg: float = 0.0) -> np.ndarray:
    a = np.deg2rad(deg)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return points @ rotation.T + np.array([dx, dy])


class TestFit:
    def test_recovers_a_pure_translation(self) -> None:
        """The Suzuka case: corners offset from telemetry by a constant vector."""
        path = _ring()
        corners = _shift(_markers(), -8.7, 29.5)

        transform = fit_corner_frame(corners, path)

        assert transform.median_residual_m < 0.5
        assert transform.translation[0] == pytest.approx(8.7, abs=0.5)
        assert transform.translation[1] == pytest.approx(-29.5, abs=0.5)
        assert abs(transform.rotation_deg) < 1.0

    def test_recovers_a_rotation(self) -> None:
        corners = _shift(_markers(), 0.0, 0.0, deg=5.0)
        transform = fit_corner_frame(corners, _ring())
        assert transform.median_residual_m < 1.0

    def test_already_aligned_input_is_left_alone(self) -> None:
        transform = fit_corner_frame(_markers(), _ring())
        assert transform.median_residual_m < 0.5
        assert transform.translation_norm_m < 1.0

    def test_apply_moves_points_onto_the_path(self) -> None:
        path = _ring()
        corners = _shift(_markers(), -8.7, 29.5)
        moved = fit_corner_frame(corners, path).apply(corners)

        distances = np.sqrt(((moved[:, None, :] - path[None, :, :]) ** 2).sum(-1)).min(axis=1)
        assert distances.max() < 1.0


class TestGuards:
    def test_rejects_too_few_corners(self) -> None:
        with pytest.raises(FrameFitError, match=">=3 corners"):
            fit_corner_frame(_markers(2), _ring())

    def test_rejects_a_sparse_path(self) -> None:
        with pytest.raises(FrameFitError, match="too sparse"):
            fit_corner_frame(_markers(), _ring(50))

    def test_refuses_when_no_transform_fits(self) -> None:
        """Corners from a different circuit must not be silently 'calibrated'."""
        corners = np.column_stack(
            [np.linspace(-2000, 2000, 9), np.full(9, 1500.0)]  # a straight line
        )
        with pytest.raises(FrameFitError, match="no rigid transform"):
            fit_corner_frame(corners, _ring(), max_residual_m=8.0)

    def test_does_not_mirror(self) -> None:
        """A reflected fit would score well and be geometrically nonsense."""
        corners = _markers().copy()
        corners[:, 1] *= -1
        transform = fit_corner_frame(corners, _ring())
        cos_a, sin_a = np.cos(transform.rotation_rad), np.sin(transform.rotation_rad)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        assert np.linalg.det(rotation) == pytest.approx(1.0)


class TestTransform:
    def test_is_frozen_and_reports_in_readable_units(self) -> None:
        transform = RigidTransform(
            rotation_rad=np.deg2rad(90.0),
            translation=np.array([3.0, 4.0]),
            median_residual_m=0.1,
            iterations=3,
        )
        assert transform.rotation_deg == pytest.approx(90.0)
        assert transform.translation_norm_m == pytest.approx(5.0)
        with pytest.raises(Exception):
            transform.rotation_rad = 0.0  # type: ignore[misc]

    def test_apply_is_a_rotation_then_translation(self) -> None:
        transform = RigidTransform(
            rotation_rad=np.deg2rad(90.0),
            translation=np.array([1.0, 0.0]),
            median_residual_m=0.0,
            iterations=1,
        )
        out = transform.apply(np.array([[1.0, 0.0]]))
        assert out[0] == pytest.approx([1.0, 1.0], abs=1e-9)
