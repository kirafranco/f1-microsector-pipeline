"""Corner-anchored distance alignment (D5, method (c)).

FastF1's per-lap distance is the time-integral of speed, so it carries both the
line the driver took and the speed sensor's error. Two drivers' lap totals
differ by tens of metres, which at 300 km/h is a few tenths of a second of
registration error — larger than the effects this project exists to measure.

The fix: locate each lap's closest approach to known circuit corners, then
rubber-band that lap's distance axis onto the reference distances of those
corners. The map is built from corner anchors only. The lap's endpoint is
deliberately *not* anchored to the official lap length, so total distance
remains a measurement rather than an assumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MIN_ANCHORS = 8
#: An anchor further than this from its corner means the car was not there:
#: off track, a wildly different line, or bad position data. Dropped, not used.
DEFAULT_MAX_RESIDUAL_M = 50.0


@dataclass(frozen=True)
class TrackReference:
    """The circuit's reference frame: where the corners are, and how far in."""

    circuit: str
    corners: pd.DataFrame  # number, letter, x, y, distance
    lap_length_m: float

    def __post_init__(self) -> None:
        required = {"number", "x", "y", "distance"}
        missing = required - set(self.corners.columns)
        if missing:
            raise ValueError(f"track reference corners missing columns: {sorted(missing)}")
        if len(self.corners) < 2:
            raise ValueError("track reference needs at least 2 corners")

    def subset(self, corner_numbers: "list[int] | None") -> pd.DataFrame:
        """Corners ordered by reference distance, optionally filtered."""
        corners = self.corners.sort_values("distance").reset_index(drop=True)
        if corner_numbers is None:
            return corners
        keep = corners["number"].isin(corner_numbers)
        return corners.loc[keep].reset_index(drop=True)


@dataclass(frozen=True)
class AlignedLap:
    """One lap's alignment result."""

    telemetry: pd.DataFrame
    anchors: pd.DataFrame
    rejected: bool
    reject_reason: str | None = None


def split_anchor_and_holdout(reference: TrackReference) -> tuple[list[int], list[int]]:
    """Odd corner numbers anchor; even corner numbers validate.

    The held-out corners are never seen by the alignment, which is what makes
    registration error measured at them a real test rather than a restatement
    of the algorithm's own arithmetic.
    """
    numbers = sorted(int(n) for n in reference.corners["number"].dropna().unique())
    anchors = [n for n in numbers if n % 2 == 1]
    holdout = [n for n in numbers if n % 2 == 0]
    return anchors, holdout


def _project_onto_path(
    px: float,
    py: float,
    xs: np.ndarray,
    ys: np.ndarray,
    distance: np.ndarray,
    index: int,
) -> tuple[float, float]:
    """Project a corner onto the path segments adjacent to `index`.

    Snapping an anchor to the nearest *sample* quantises its longitudinal
    position by the sample spacing -- about 20 m at racing speed, which is
    larger than the registration tolerance the anchors exist to achieve.
    Projecting onto the segment recovers sub-sample precision, and the residual
    becomes the perpendicular offset between the corner reference point and the
    driven line rather than a mix of that and sampling error.

    Returns (perpendicular_residual_m, distance_at_projection).
    """
    best_residual = float("inf")
    best_distance = float(distance[index])

    for a, b in ((index - 1, index), (index, index + 1)):
        if a < 0 or b >= len(xs):
            continue
        ax, ay = float(xs[a]), float(ys[a])
        vx, vy = float(xs[b]) - ax, float(ys[b]) - ay
        length_sq = vx * vx + vy * vy
        if length_sq == 0.0:
            continue

        t = ((px - ax) * vx + (py - ay) * vy) / length_sq
        t = min(1.0, max(0.0, t))
        cx, cy = ax + t * vx, ay + t * vy
        residual = float(np.hypot(px - cx, py - cy))

        if residual < best_residual:
            best_residual = residual
            best_distance = float(distance[a] + t * (distance[b] - distance[a]))

    if not np.isfinite(best_residual):
        return float("inf"), best_distance
    return best_residual, best_distance


def locate_anchors(
    lap: pd.DataFrame,
    corners: pd.DataFrame,
    *,
    max_residual_m: float = DEFAULT_MAX_RESIDUAL_M,
) -> pd.DataFrame:
    """Closest approach to each corner, searched in track order.

    The search window only ever moves forward, so a corner cannot match a
    sample earlier than the previous corner's match. That is what keeps tightly
    spaced sequences — Suzuka's esses — from cross-matching.
    """
    xs = lap["x"].to_numpy(dtype=float)
    ys = lap["y"].to_numpy(dtype=float)
    d_raw = lap["distance_raw"].to_numpy(dtype=float)
    n = len(xs)

    rows: list[dict] = []
    cursor = 0
    previous_d: float | None = None

    for corner in corners.itertuples(index=False):
        if cursor >= n:
            break

        dx = xs[cursor:] - float(corner.x)
        dy = ys[cursor:] - float(corner.y)
        squared = dx * dx + dy * dy
        offset = int(np.argmin(squared))
        index = cursor + offset
        residual, d_here = _project_onto_path(
            float(corner.x), float(corner.y), xs, ys, d_raw, index
        )

        # The cursor advances past accepted anchors only. Advancing on a drop
        # would consume the rest of the lap and starve every later corner of
        # search window -- one bad corner would take the whole lap with it.
        if residual > max_residual_m:
            logger.debug(
                "anchor_dropped corner=%s reason=residual residual_m=%.1f", corner.number, residual
            )
            continue
        if previous_d is not None and d_here <= previous_d:
            logger.debug("anchor_dropped corner=%s reason=non_monotonic", corner.number)
            continue

        cursor = index + 1
        previous_d = d_here
        rows.append(
            {
                "corner_number": int(corner.number),
                "sample_index": index,
                "d_raw": d_here,
                "d_ref": float(corner.distance),
                "residual_m": residual,
            }
        )

    return pd.DataFrame(
        rows, columns=["corner_number", "sample_index", "d_raw", "d_ref", "residual_m"]
    )


def _map_distance(d_raw: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Piecewise-linear map, extrapolating beyond the anchors by local scale.

    np.interp clamps outside its range, which would flatten the start and end
    of every lap into a constant distance. Extrapolating with the local scale
    factor keeps the axis monotonic and leaves total lap length free to be
    measured rather than imposed.
    """
    out = np.interp(d_raw, xp, fp)

    before = d_raw < xp[0]
    if before.any():
        scale = (fp[1] - fp[0]) / (xp[1] - xp[0])
        out[before] = fp[0] + (d_raw[before] - xp[0]) * scale

    after = d_raw > xp[-1]
    if after.any():
        scale = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        out[after] = fp[-1] + (d_raw[after] - xp[-1]) * scale

    return out


def align_lap(
    lap_telemetry: pd.DataFrame,
    reference: TrackReference,
    min_anchors: int = DEFAULT_MIN_ANCHORS,
    anchor_corner_numbers: "list[int] | None" = None,
    max_residual_m: float = DEFAULT_MAX_RESIDUAL_M,
) -> AlignedLap:
    """Give one lap a track-referenced distance axis.

    Pure: no I/O, no global state, no mutation of the input frame.
    """
    corners = reference.subset(anchor_corner_numbers)
    anchors = locate_anchors(lap_telemetry, corners, max_residual_m=max_residual_m)

    if len(anchors) < min_anchors:
        return AlignedLap(
            telemetry=lap_telemetry.iloc[0:0].assign(distance_aligned=pd.Series(dtype=float)),
            anchors=anchors,
            rejected=True,
            reject_reason=(
                f"only {len(anchors)} usable anchor(s) of {len(corners)} corners, "
                f"minimum is {min_anchors}"
            ),
        )

    d_raw = lap_telemetry["distance_raw"].to_numpy(dtype=float)
    aligned = _map_distance(d_raw, anchors["d_raw"].to_numpy(), anchors["d_ref"].to_numpy())

    if not np.all(np.diff(aligned) > 0):
        return AlignedLap(
            telemetry=lap_telemetry.iloc[0:0].assign(distance_aligned=pd.Series(dtype=float)),
            anchors=anchors,
            rejected=True,
            reject_reason="aligned distance is not strictly increasing",
        )

    out = lap_telemetry.copy()
    out["distance_aligned"] = aligned
    return AlignedLap(telemetry=out, anchors=anchors, rejected=False)
