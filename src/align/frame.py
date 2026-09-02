"""Frame calibration between corner reference points and position telemetry.

FastF1's circuit-info corner coordinates and the car position stream do not
share an origin. Measured on Suzuka 2024 Q, corner points sit a median of
24.5 m from the driven line -- wider than the track -- purely because of a
constant frame offset. Fitting and removing it brings the median to 0.56 m.

The transform is fitted per session from the data itself rather than stored as
a per-circuit constant: the offset is a property of how a given session's
streams were produced, and a hardcoded table would silently rot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ITERATIONS = 30
DEFAULT_TOLERANCE_M = 1e-3

#: Above this the fit has not found a sensible frame and anchoring should not
#: proceed on the assumption that it has.
MAX_ACCEPTABLE_RESIDUAL_M = 8.0


class FrameFitError(RuntimeError):
    """No rigid transform brings corner points onto the driven line."""


@dataclass(frozen=True)
class RigidTransform:
    """A 2D rotation about the origin followed by a translation."""

    rotation_rad: float
    translation: np.ndarray
    median_residual_m: float
    iterations: int

    @property
    def rotation_deg(self) -> float:
        return float(np.rad2deg(self.rotation_rad))

    @property
    def translation_norm_m(self) -> float:
        return float(np.hypot(*self.translation))

    def apply(self, xy: np.ndarray) -> np.ndarray:
        cos_a, sin_a = np.cos(self.rotation_rad), np.sin(self.rotation_rad)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return np.asarray(xy, dtype=float) @ rotation.T + self.translation


def _nearest(points: np.ndarray, cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance and index of the closest cloud point for each input point."""
    squared = ((points[:, None, :] - cloud[None, :, :]) ** 2).sum(-1)
    index = squared.argmin(axis=1)
    return np.sqrt(squared[np.arange(len(points)), index]), index


def fit_corner_frame(
    corners_xy: np.ndarray,
    path_xy: np.ndarray,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    max_residual_m: float = MAX_ACCEPTABLE_RESIDUAL_M,
) -> RigidTransform:
    """Fit the rigid transform putting corner points onto the driven line.

    Iterative closest point: no correspondences are assumed between corners and
    path samples. Uses X/Y only -- never the corner distance channel -- so the
    fit cannot leak into the distance registration it later enables.
    """
    corners_xy = np.asarray(corners_xy, dtype=float)
    path_xy = np.asarray(path_xy, dtype=float)

    if len(corners_xy) < 3:
        raise FrameFitError(f"need >=3 corners to fit a frame, got {len(corners_xy)}")
    if len(path_xy) < 100:
        raise FrameFitError(f"path has only {len(path_xy)} points; too sparse to fit against")

    current = corners_xy.copy()
    rotation_total = np.eye(2)
    translation_total = np.zeros(2)
    previous_residual = np.inf
    used = 0

    for used in range(1, iterations + 1):
        distances, index = _nearest(current, path_xy)
        target = path_xy[index]

        source_centre = current.mean(axis=0)
        target_centre = target.mean(axis=0)
        covariance = (current - source_centre).T @ (target - target_centre)
        u, _, vt = np.linalg.svd(covariance)
        # Reflection guard: a mirrored "fit" is never a valid circuit frame.
        correction = np.diag([1.0, float(np.linalg.det(vt.T @ u.T))])
        rotation = vt.T @ correction @ u.T
        translation = target_centre - rotation @ source_centre

        current = current @ rotation.T + translation
        rotation_total = rotation @ rotation_total
        translation_total = rotation @ translation_total + translation

        residual = float(np.median(_nearest(current, path_xy)[0]))
        if abs(previous_residual - residual) < tolerance_m:
            break
        previous_residual = residual

    residual = float(np.median(_nearest(current, path_xy)[0]))
    if residual > max_residual_m:
        raise FrameFitError(
            f"no rigid transform aligns corners to the driven line "
            f"(median residual {residual:.2f} m > {max_residual_m} m). The corner "
            f"reference data does not describe this circuit's telemetry frame."
        )

    transform = RigidTransform(
        rotation_rad=float(np.arctan2(rotation_total[1, 0], rotation_total[0, 0])),
        translation=translation_total,
        median_residual_m=residual,
        iterations=used,
    )
    logger.info(
        "frame_fitted rotation_deg=%.4f translation_m=(%.2f, %.2f) residual_m=%.2f iterations=%d",
        transform.rotation_deg,
        transform.translation[0],
        transform.translation[1],
        residual,
        used,
    )
    return transform
