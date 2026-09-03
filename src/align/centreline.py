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


def extend_axis_ends(frame: pd.DataFrame, line_length_m: float) -> pd.DataFrame:
    """Replace clamped runs at both ends of the axis with raw-distance extension.

    Samples that lie before the reference line's first vertex or after its last
    one all project onto that vertex, so they share one distance (0 or the line
    length). Measured on Suzuka 2024 Q (F010): 72 of 74 laps carry such a run
    at the start and 62 at the end, and downstream resampling then places the
    lap's time origin up to one sample interval early and its end one late --
    a systematic +0.12 s on every reconstructed lap time.

    The fix continues the axis beyond the line with `distance_raw`, the
    speed-integrated distance the alignment already trusts:
    ``d = d_anchor -/+ (raw_anchor - raw)`` from the nearest unclamped sample.
    Interior flat runs left by the monotonic guard are not touched. Frames
    without `distance_raw` are returned unchanged.
    """
    if "distance_raw" not in frame.columns or len(frame) < 2:
        return frame
    distance = frame["distance_aligned"].to_numpy(dtype=float).copy()
    raw = frame["distance_raw"].to_numpy(dtype=float)
    n = len(distance)
    leading = trailing = 0

    if distance[0] <= 1e-9:
        k = int(np.argmax(distance > distance[0]))
        if 0 < k < n:
            distance[:k] = distance[k] - (raw[k] - raw[:k])
            leading = k

    if distance[-1] >= line_length_m - 1e-6:
        reversed_ = distance[::-1]
        k = int(np.argmax(reversed_ < reversed_[0]))
        if 0 < k < n:
            j = n - 1 - k
            distance[j + 1 :] = distance[j] + (raw[j + 1 :] - raw[j])
            trailing = k

    if leading or trailing:
        logger.debug("axis_extended leading=%d trailing=%d", leading, trailing)
    out = frame.copy()
    out["distance_aligned"] = distance
    return out


def project_lap(frame: pd.DataFrame, line: ReferenceLine) -> pd.DataFrame:
    """Add `distance_aligned` and `line_offset_m` to one merged lap frame.

    Projection, then the monotonic guard, then the axis is extended past the
    reference line's ends with raw distance instead of being clamped there.
    """
    distance, offset = project_onto_line(frame[["x", "y"]].to_numpy(dtype=float), line)
    distance, _ = enforce_monotonic(distance)
    out = frame.copy()
    out["distance_aligned"] = distance
    out["line_offset_m"] = offset
    return extend_axis_ends(out, line.total_length_m)
