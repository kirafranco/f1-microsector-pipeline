"""F015: the lead-in, and the defect it exists to remove.

The synthetic circuit here is the Bahrain shape: a seed lap whose telemetry
opens just after the timing line, so the reference line starts inside the lap
and another lap that opened earlier has samples with nowhere to project. Every
test states what happens *without* the lead-in as well as with it, because the
whole point is that the bare line looked fine on one circuit and was wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.align.centreline import (
    LEAD_M,
    MAX_PROJECTION_OFFSET_M,
    add_lead_in,
    build_reference_line,
    misprojected,
    project_lap,
    project_onto_line,
)

RADIUS_M = 300.0
CIRCUMFERENCE_M = 2 * np.pi * RADIUS_M
#: Metres per radian on the ring, for turning an opening gap into an angle.
PER_RADIAN = RADIUS_M


def ring(n: int = 900, radius: float = RADIUS_M, opens_m: float = 0.0,
         closes_m: float = 0.0) -> np.ndarray:
    """One lap of a circular circuit, as telemetry that misses the ends.

    `opens_m` is how far past the timing line the telemetry begins and
    `closes_m` how far before the next crossing it stops -- the gap that makes
    a real seed lap an open curve.
    """
    start = opens_m / PER_RADIAN
    end = 2 * np.pi - closes_m / PER_RADIAN
    theta = np.linspace(start, end, n)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def lap_frame(xy: np.ndarray, speed_ms: float = 80.0) -> pd.DataFrame:
    """A merged lap frame with the raw distance the alignment trusts."""
    steps = np.hypot(*np.diff(xy, axis=0).T)
    raw = np.concatenate([[0.0], np.cumsum(steps)])
    return pd.DataFrame({
        "x": xy[:, 0], "y": xy[:, 1], "distance_raw": raw,
        "session_time": raw / speed_ms, "speed": speed_ms * 3.6,
    })


#: The seed lap Bahrain's reference line was built from: opens 6 m past the
#: line, stops 24 m before it. The two ends are 30 m apart, as measured.
SEED = ring(opens_m=6.0, closes_m=24.0)


class TestTheLineIsOpenWithoutALeadIn:
    def test_the_seed_path_does_not_close(self) -> None:
        gap = float(np.hypot(*(SEED[0] - SEED[-1])))
        assert gap == pytest.approx(30.0, abs=1.0), "the measured Bahrain gap"

    def test_a_lap_opening_before_the_seed_did_has_nowhere_to_project(self) -> None:
        """Without a lead-in the line simply does not reach back that far, so
        the sample lands on the line's own first vertex, metres away."""
        bare = build_reference_line(SEED, lead_m=0.0)
        early = ring(opens_m=0.0, closes_m=24.0)  # crossed the line 6 m before the seed opened
        distance, offset = project_onto_line(early, bare)
        assert distance[0] >= 0.0, "clamped forward onto the line's own start"
        assert offset[0] > 1.0, "and it is not actually there"

    def test_with_a_lead_in_it_lands_where_it_belongs(self) -> None:
        line = build_reference_line(SEED)
        early = ring(opens_m=0.0, closes_m=24.0)
        distance, offset = project_onto_line(early, line)
        assert offset[0] < MAX_PROJECTION_OFFSET_M
        assert distance[0] == pytest.approx(-6.0, abs=3.0), "6 m before the seed's own start"
        assert misprojected(offset) == 0


class TestLeadIn:
    def test_it_extends_backwards_only(self) -> None:
        extended, lead = add_lead_in(SEED, lead_m=60.0)
        assert lead == 60.0
        assert len(extended) > len(SEED)
        assert np.allclose(extended[-1], SEED[-1]), "the end of the path is untouched"

    def test_the_extension_continues_the_direction_of_travel(self) -> None:
        extended, _ = add_lead_in(SEED, lead_m=60.0)
        back = float(np.hypot(*(extended[0] - SEED[0])))
        assert back == pytest.approx(60.0, rel=0.05)
        # It goes backwards along the lap, not forwards into it.
        assert np.hypot(*(extended[0] - SEED[10])) > np.hypot(*(SEED[0] - SEED[10]))

    def test_zero_is_a_no_op(self) -> None:
        extended, lead = add_lead_in(SEED, lead_m=0.0)
        assert lead == 0.0 and len(extended) == len(SEED)

    def test_the_default_covers_the_worst_opening_seen(self) -> None:
        """0.40 s at 81 m/s is 32 m; the default leaves room for it."""
        assert LEAD_M >= 32.0


class TestArcZeroStaysPut:
    def test_the_lead_in_carries_negative_arc(self) -> None:
        """Moving the origin forward instead would push every lap's first grid
        points beyond the start of its data, and the resampler would
        extrapolate the time channel across them -- the very bias being fixed."""
        line = build_reference_line(SEED)
        assert line.arc_length_m[0] == pytest.approx(-line.lead_in_m)
        assert line.start_m < 0

    def test_the_seed_lap_still_starts_at_zero(self) -> None:
        line = build_reference_line(SEED)
        distance, _ = project_onto_line(SEED, line)
        assert distance[0] == pytest.approx(0.0, abs=2.0)

    def test_the_lap_length_is_unchanged_by_the_lead_in(self) -> None:
        bare = build_reference_line(SEED, lead_m=0.0)
        led = build_reference_line(SEED)
        assert led.total_length_m == pytest.approx(bare.total_length_m, rel=1e-3)
        assert led.total_length_m - led.start_m == pytest.approx(
            bare.total_length_m + led.lead_in_m, rel=1e-3)


class TestProjectLap:
    def test_an_early_opening_lap_gets_a_negative_start(self) -> None:
        """It began before the seed lap did, so it is before distance zero."""
        line = build_reference_line(SEED)
        frame = lap_frame(ring(opens_m=0.0, closes_m=24.0))
        out = project_lap(frame, line)
        assert out["distance_aligned"].iloc[0] < 0.0
        assert misprojected(out["line_offset_m"].to_numpy()) == 0

    def test_the_axis_never_goes_backwards(self) -> None:
        line = build_reference_line(SEED)
        out = project_lap(lap_frame(ring(opens_m=20.0, closes_m=24.0)), line)
        assert np.all(np.diff(out["distance_aligned"].to_numpy()) >= 0)

    def test_two_laps_that_opened_differently_still_register(self) -> None:
        """The premise of the whole project: the same physical point is the
        same distance, whatever each lap's telemetry happened to do."""
        line = build_reference_line(SEED)
        early = project_lap(lap_frame(ring(opens_m=2.0, closes_m=24.0)), line)
        late = project_lap(lap_frame(ring(opens_m=30.0, closes_m=24.0)), line)
        # A quarter of the way round is a quarter of the way round.
        quarter = CIRCUMFERENCE_M / 4
        for frame in (early, late):
            d = frame["distance_aligned"].to_numpy()
            index = int(np.argmin(np.abs(d - quarter)))
            angle = np.arctan2(frame["y"].iloc[index], frame["x"].iloc[index]) % (2 * np.pi)
            assert angle == pytest.approx(np.pi / 2, abs=0.05)


class TestMisprojected:
    def test_it_counts_what_did_not_land_on_the_line(self) -> None:
        assert misprojected(np.array([0.1, 0.2, 25.0, 0.3])) == 1
        assert misprojected(np.array([0.1, 0.2])) == 0

    def test_the_threshold_is_far_above_a_normal_offset(self) -> None:
        """Normal offsets are ~0 m; the Bahrain failures measured 20-25 m."""
        line = build_reference_line(SEED)
        _, offset = project_onto_line(ring(opens_m=6.0, closes_m=24.0), line)
        assert offset.max() < 1.0
        assert MAX_PROJECTION_OFFSET_M > 10 * offset.max()
