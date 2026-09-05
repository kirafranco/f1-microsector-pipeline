"""Reference-line projection (D5 fallback, method (d)).

Corner anchoring maps each lap onto FastF1's corner `Distance` values, so it
inherits whatever error those 18 numbers carry. Projection instead defines the
distance axis as arc length along a single reference line shared by every lap:
two laps at the same physical point project to the same arc length by
construction, regardless of how good FastF1's corner distances are.

The reference line is one clean lap, resampled to uniform spacing. It is a
*reference*, not a claim about the geometric centre of the track -- what the
project needs is that every lap uses the same one.

That lap's telemetry does not begin at the timing line: it opens some tenths
after the car crosses, and closes some tenths before the next crossing. The
seed path is therefore an *open* curve whose two ends are metres apart in
space -- 30.2 m at Bahrain, measured in F015 -- and a lap whose own telemetry
opens inside that gap has nowhere real to project to. It lands on the first
vertex with a 20 m perpendicular offset, its arc reads ~0 instead of ~-25, and
the lap's start is registered that much late. Measured on Bahrain 2024 Q, that
put a -0.187 s bias on every reconstructed lap time, all of it in sector 1.

So the line is given a lead-in before it is used: its start is continued
backwards along its own heading, which gives a sample arriving before the seed
lap's telemetry began a real segment to project onto instead of the nearest
vertex several metres away. Suzuka never showed this because its seed lap
opened 30 m *past* the line, so its leading samples fell before vertex 0 and
were caught by `extend_axis_ends` instead -- the right fix for a precondition
that turned out to be circuit-dependent.

The *end* of the line is deliberately left alone. Samples past the last vertex
clamp to it cleanly and `extend_axis_ends` continues them with speed-integrated
distance, which measured better than extrapolating the line: a lead-out was
tried and cost Suzuka 0.044 s of delta-t closure at p95 (F015).
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

#: How far the seed path is continued backwards from its own start. The
#: largest telemetry-opening gap observed is 0.40 s, which at 81 m/s is 32 m;
#: 60 m covers it with room, and 90 m measured identically.
LEAD_M = 60.0

#: A sample further than this from the line has not been located on it -- it
#: has been snapped to the closest thing available. Normal offsets are ~0 m;
#: the mis-projected leading samples at Bahrain measured 20-25 m. Used to
#: report the condition, not to correct it: the lead-in is the correction.
MAX_PROJECTION_OFFSET_M = 10.0


@dataclass(frozen=True)
class ReferenceLine:
    """An ordered polyline with cumulative arc length at each vertex."""

    xy: np.ndarray  # (n, 2), metres
    arc_length_m: np.ndarray  # (n,), metres from the line's own start
    #: Metres of lead-in prepended to the seed path when the line was closed.
    #: Arc zero remains the seed lap's own first sample, so the lead-in spans
    #: [-lead_in_m, 0) and a negative distance is a sample from before the
    #: seed lap's telemetry opened.
    lead_in_m: float = 0.0

    @property
    def total_length_m(self) -> float:
        return float(self.arc_length_m[-1])

    @property
    def start_m(self) -> float:
        """Arc length of the line's first vertex: negative by the lead-in."""
        return float(self.arc_length_m[0])


def _arc_length(xy: np.ndarray) -> np.ndarray:
    steps = np.hypot(*np.diff(xy, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _heading(xy: np.ndarray, at_start: bool, span_m: float = 20.0) -> np.ndarray:
    """Unit direction of travel at one end of a path.

    Taken over `span_m` rather than between two adjacent samples: position data
    is noisy at 10 Hz, and one noisy pair would aim the extension off track.
    """
    arc = _arc_length(xy)
    if at_start:
        far = int(np.searchsorted(arc, span_m))
        a, b = xy[min(far, len(xy) - 1)], xy[0]
    else:
        far = int(np.searchsorted(arc, arc[-1] - span_m))
        a, b = xy[-1], xy[max(far, 0)]
    direction = a - b if at_start else a - b
    norm = float(np.hypot(*direction))
    if norm == 0.0:
        raise ValueError("cannot take a heading from a zero-length span")
    return direction / norm


def add_lead_in(xy: np.ndarray, lead_m: float = LEAD_M,
                spacing_m: float = DEFAULT_SPACING_M) -> tuple[np.ndarray, float]:
    """Continue a seed lap's path backwards from its own first sample.

    The seed lap's telemetry opens after the car crosses the timing line, so
    the line's first vertex is already inside the lap. Another lap whose
    telemetry opened earlier has samples before that vertex with nowhere to
    project; extending the path backwards gives them somewhere real.

    The extension is straight. Over the tens of metres involved that is the
    right approximation: start/finish is on a straight at every circuit on the
    calendar, because that is where the pits are.

    Returns the extended path and the metres prepended to its start.
    """
    xy = np.asarray(xy, dtype=float)
    if lead_m <= 0:
        return xy, 0.0

    steps = max(2, int(round(lead_m / spacing_m)))
    offsets = np.linspace(lead_m, 0.0, steps, endpoint=False)[:, None]
    into = _heading(xy, at_start=True)
    return np.vstack([xy[0] - offsets * into, xy]), float(lead_m)


def build_reference_line(
    xy: np.ndarray, spacing_m: float = DEFAULT_SPACING_M, lead_m: float = LEAD_M
) -> ReferenceLine:
    """Resample one lap's path to uniform spacing, with the ends continued.

    `lead_m` is the lead-in described in the module docstring. Pass 0 to build
    the bare seed path, which is what the unit tests use to show what the
    lead-in is for.
    """
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 10:
        raise ValueError(f"need >=10 points to build a reference line, got {len(xy)}")

    extended, lead_in = add_lead_in(xy, lead_m=lead_m, spacing_m=spacing_m)

    source_arc = _arc_length(extended)
    total = float(source_arc[-1])
    n_points = max(10, int(round(total / spacing_m)) + 1)
    target_arc = np.linspace(0.0, total, n_points)

    resampled = np.column_stack(
        [np.interp(target_arc, source_arc, extended[:, 0]),
         np.interp(target_arc, source_arc, extended[:, 1])]
    )
    # Arc zero stays where the seed lap's own telemetry began, so the lead-in
    # carries *negative* arc. Shifting the origin forward instead would leave
    # every lap's first grid points beyond the start of its data, and the
    # resampler would extrapolate the time channel across them -- which is the
    # -0.19 s bias this fix exists to remove, reintroduced by the fix itself.
    # Measured both ways on Suzuka and Bahrain (F015).
    arc = _arc_length(resampled) - lead_in
    line = ReferenceLine(xy=resampled, arc_length_m=arc, lead_in_m=lead_in)
    logger.info(
        "reference_line_built points=%d length_m=%.1f spacing_m=%.2f lead_in_m=%.1f",
        len(resampled),
        line.total_length_m,
        spacing_m,
        lead_in,
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
    xy: np.ndarray, vertices: np.ndarray, arc: np.ndarray, window_m: float,
    seed_span_m: float = 0.0,
) -> np.ndarray:
    """Nearest vertex per point, searched in a window that only moves forward."""
    spacing = float(np.median(np.diff(arc))) or 1.0
    span = max(4, int(round(window_m / spacing)))
    back = max(2, span // 8)

    # The seed is searched near the start of the line, never globally. Both the
    # lap and the line begin at the start/finish area, where the line's first
    # and last vertices are metres apart in space but a full lap apart in arc;
    # a global seed picks whichever is marginally closer, so laps seeded at the
    # far end read ~5722 m for their whole length.
    #
    # It is searched rather than fixed at vertex zero because the line now
    # carries a lead-in: vertex zero is tens of metres before the seed lap's
    # first sample, and pinning the first point there would put every lap's
    # opening sample at -lead_in whatever it actually was.
    seed_to = max(4, int(round((seed_span_m + window_m) / spacing)))
    seed_block = vertices[:min(seed_to, len(vertices))]
    cursor = int(((seed_block - xy[0]) ** 2).sum(axis=1).argmin())
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
        nearest = _ordered_nearest(xy, vertices, arc, window_m, seed_span_m=line.lead_in_m)
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


def misprojected(offset: np.ndarray, max_offset_m: float = MAX_PROJECTION_OFFSET_M) -> int:
    """How many samples did not land on the line.

    A sample whose nearest point on the line is 20 m away has not been located
    on it. With the lead-in this should be zero at the start of a lap; at the
    end, samples past the last vertex do clamp to it, and `extend_axis_ends`
    corrects their distance while leaving this offset visible -- so a trailing
    count is expected and a leading one is a defect.
    """
    return int((np.asarray(offset, dtype=float) > max_offset_m).sum())


def project_lap(frame: pd.DataFrame, line: ReferenceLine) -> pd.DataFrame:
    """Add `distance_aligned` and `line_offset_m` to one merged lap frame.

    Projection, then the monotonic guard, then the axis extended past the
    line's own end with raw distance instead of being clamped there. The start
    needs no such treatment: that is what the line's lead-in is for.
    """
    distance, offset = project_onto_line(frame[["x", "y"]].to_numpy(dtype=float), line)
    distance, _ = enforce_monotonic(distance)
    out = frame.copy()
    out["distance_aligned"] = distance
    out["line_offset_m"] = offset
    return extend_axis_ends(out, line.total_length_m)
