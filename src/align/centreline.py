"""Reference-line projection (D5 fallback, method (d)).

Corner anchoring maps each lap onto FastF1's corner `Distance` values, so it
inherits whatever error those 18 numbers carry. Projection instead defines the
distance axis as arc length along a single reference line shared by every lap:
two laps at the same physical point project to the same arc length by
construction, regardless of how good FastF1's corner distances are.

The reference line is one clean lap, resampled to uniform spacing. It is a
*reference*, not a claim about the geometric centre of the track -- what the
project needs is that every lap uses the same one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SPACING_M = 2.0
#: Chunk size for the nearest-vertex scan, to bound peak memory.
_CHUNK = 512


@dataclass(frozen=True)
class ReferenceLine:
    """An ordered polyline with cumulative arc length at each vertex."""

    xy: np.ndarray  # (n, 2), metres
    arc_length_m: np.ndarray  # (n,), metres from the line's own start

    @property
    def total_length_m(self) -> float:
        return float(self.arc_length_m[-1])


def _arc_length(xy: np.ndarray) -> np.ndarray:
    steps = np.hypot(*np.diff(xy, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(steps)])


def build_reference_line(
    xy: np.ndarray, spacing_m: float = DEFAULT_SPACING_M
) -> ReferenceLine:
    """Resample one lap's path to uniform spacing along its own arc length."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 10:
        raise ValueError(f"need >=10 points to build a reference line, got {len(xy)}")

    source_arc = _arc_length(xy)
    total = float(source_arc[-1])
    n_points = max(10, int(round(total / spacing_m)) + 1)
    target_arc = np.linspace(0.0, total, n_points)

    resampled = np.column_stack(
        [np.interp(target_arc, source_arc, xy[:, 0]), np.interp(target_arc, source_arc, xy[:, 1])]
    )
    line = ReferenceLine(xy=resampled, arc_length_m=_arc_length(resampled))
    logger.info(
        "reference_line_built points=%d length_m=%.1f spacing_m=%.2f",
        len(resampled),
        line.total_length_m,
        spacing_m,
    )
    return line


def _nearest_vertex(points: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    """Index of the closest cloud vertex per point, scanned in chunks."""
    out = np.empty(len(points), dtype=int)
    for start in range(0, len(points), _CHUNK):
        block = points[start : start + _CHUNK]
        squared = ((block[:, None, :] - cloud[None, :, :]) ** 2).sum(-1)
        out[start : start + _CHUNK] = squared.argmin(axis=1)
    return out


def _ordered_nearest(
    xy: np.ndarray, vertices: np.ndarray, arc: np.ndarray, window_m: float
) -> np.ndarray:
    """Nearest vertex per point, searched in a window that only moves forward."""
    spacing = float(np.median(np.diff(arc))) or 1.0
    span = max(4, int(round(window_m / spacing)))
    back = max(2, span // 8)

    # Seed at arc zero, not at the globally nearest vertex. Both the lap and the
    # reference line begin at the start/finish line, where the line's first and
    # last vertices are metres apart in space but a full lap apart in arc. A
    # global seed picks whichever is marginally closer, so laps seeded at the
    # far end read ~5722 m for their whole length.
    cursor = 0
    out = np.empty(len(xy), dtype=int)
    out[0] = cursor

    for i in range(1, len(xy)):
        lo = max(0, cursor - back)
        hi = min(len(vertices), cursor + span)
        block = vertices[lo:hi]
        squared = ((block - xy[i]) ** 2).sum(axis=1)
        cursor = lo + int(squared.argmin())
        out[i] = cursor

    return out


def project_onto_line(
    xy: np.ndarray, line: ReferenceLine, *, ordered: bool = True, window_m: float = 150.0
) -> tuple[np.ndarray, np.ndarray]:
    """Arc length and perpendicular offset of each point against the line.

    Projection is refined onto the segments adjacent to the nearest vertex, so
    precision is not limited by the line's vertex spacing.

    With `ordered`, the search advances a cursor along the line instead of
    scanning it globally. That is not an optimisation: at the start/finish line
    the first and last vertices are metres apart in space but a full lap apart
    in arc length, so a global nearest-vertex search assigns some samples ~0 and
    others ~5722 m, and the axis is destroyed. A lap traverses the line once, in
    order, so the cursor encodes exactly that.
    """
    xy = np.asarray(xy, dtype=float)
    vertices = line.xy
    arc = line.arc_length_m

    if ordered:
        nearest = _ordered_nearest(xy, vertices, arc, window_m)
    else:
        nearest = _nearest_vertex(xy, vertices)

    distance = np.empty(len(xy))
    offset = np.empty(len(xy))

    for i, (point, index) in enumerate(zip(xy, nearest)):
        best_offset = np.inf
        best_distance = float(arc[index])
        for a, b in ((index - 1, index), (index, index + 1)):
            if a < 0 or b >= len(vertices):
                continue
            start = vertices[a]
            direction = vertices[b] - start
            length_sq = float(direction @ direction)
            if length_sq == 0.0:
                continue
            t = float((point - start) @ direction / length_sq)
            t = min(1.0, max(0.0, t))
            foot = start + t * direction
            perpendicular = float(np.hypot(*(point - foot)))
            if perpendicular < best_offset:
                best_offset = perpendicular
                best_distance = float(arc[a] + t * (arc[b] - arc[a]))
        distance[i] = best_distance
        offset[i] = best_offset

    return distance, offset


def enforce_monotonic(distance: np.ndarray) -> tuple[np.ndarray, int]:
    """Force a non-decreasing axis, reporting how many samples were corrected.

    Projection is pointwise, so a sample near the start/finish line or a noisy
    position fix can land slightly behind its predecessor. A distance axis that
    goes backwards is meaningless downstream, so it is clamped -- and the count
    is surfaced rather than swallowed, because a large one means the projection
    is failing, not merely jittering.
    """
    corrected = np.maximum.accumulate(distance)
    return corrected, int((corrected != distance).sum())


def project_lap(frame: pd.DataFrame, line: ReferenceLine) -> pd.DataFrame:
    """Add `distance_aligned` and `line_offset_m` to one merged lap frame."""
    distance, offset = project_onto_line(frame[["x", "y"]].to_numpy(dtype=float), line)
    distance, _ = enforce_monotonic(distance)
    out = frame.copy()
    out["distance_aligned"] = distance
    out["line_offset_m"] = offset
    return out
