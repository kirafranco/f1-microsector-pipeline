"""F009 trough search and event detection on the designed session."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.segment.corners import corner_positions, load_frame
from src.segment.events import (
    EVENT_SCHEMA,
    EventError,
    EventParams,
    detect_events,
    find_troughs,
    median_traces,
)
from src.segment.phases import build_corner_phases
from tests import synthetic_session as syn


def placed_corners() -> pd.DataFrame:
    return corner_positions(syn.grid(), syn.raw_corners(), load_frame(syn.frame_meta()), syn.REFERENCE_LAP)


class TestFindTroughs:
    def test_finds_the_two_events_and_ignores_the_shallow_dip(self) -> None:
        found = find_troughs(syn.speed_profile(), 8.0)
        assert [t.index for t in found] == [80, 180]
        assert [t.speed for t in found] == [100.0, 286.0]
        assert [round(t.prominence, 6) for t in found] == [200.0, 14.0]

    def test_threshold_is_respected(self) -> None:
        assert len(find_troughs(syn.speed_profile(), 4.0)) == 3
        assert len(find_troughs(syn.speed_profile(), 50.0)) == 1

    def test_flat_floor_reports_its_first_point(self) -> None:
        found = find_troughs(syn.speed_profile(), 8.0)
        assert found[0].index == 80  # floor spans 800..830 m

    def test_circular_wrap_finds_a_trough_across_the_line_once(self) -> None:
        rolled = np.roll(syn.speed_profile(), -78)  # trough A now at index 2
        found = find_troughs(rolled, 8.0)
        assert [t.index for t in found] == [2, 102]
        assert found[0].prominence == pytest.approx(200.0)

    def test_constant_and_degenerate_traces_have_no_troughs(self) -> None:
        assert find_troughs(np.full(50, 250.0), 8.0) == []
        assert find_troughs(np.array([1.0, 2.0]), 0.0) == []
        assert find_troughs(np.array([1.0, np.nan, 3.0, 2.0]), 0.0) == []

    def test_single_sinusoid_prominence_is_peak_to_trough(self) -> None:
        x = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        found = find_troughs(200 + 50 * np.cos(x), 8.0)
        assert len(found) == 1
        assert found[0].prominence == pytest.approx(100.0, abs=0.1)


class TestMedianTraces:
    def test_medians_match_the_design(self) -> None:
        t = median_traces(syn.grid())
        assert len(t) == syn.N_POINTS
        np.testing.assert_allclose(t["throttle"], syn.throttle_profile())
        np.testing.assert_allclose(t["speed"], syn.speed_profile(), atol=0.6)
        # Two of four laps brake one bin later: the 640 m point is braked by half the field.
        assert t["brake"].loc[64] == pytest.approx(0.5)
        assert t["brake"].loc[65] == pytest.approx(1.0)

    def test_short_laps_do_not_extend_the_trace(self) -> None:
        g = syn.grid()
        long_lap = (g["driver"] == "AAA") & (g["lap_number"] == 1)
        extra = g[long_lap].tail(3).assign(grid_index=[300, 301, 302], distance_m=[3000.0, 3010.0, 3020.0])
        t = median_traces(pd.concat([g, extra], ignore_index=True))
        assert len(t) == syn.N_POINTS

    def test_missing_index_zero_is_an_error(self) -> None:
        g = syn.grid()
        with pytest.raises(EventError, match="index 0"):
            median_traces(g[g["grid_index"] > 0])

    def test_empty_grid_is_an_error(self) -> None:
        with pytest.raises(EventError):
            median_traces(syn.grid().iloc[:0])


class TestDetectEvents:
    @pytest.fixture()
    def events(self) -> pd.DataFrame:
        return detect_events(syn.traces(), placed_corners())

    def test_two_events_with_the_designed_boundaries(self, events: pd.DataFrame) -> None:
        assert len(events) == 2
        a, b = events.iloc[0], events.iloc[1]
        assert (a["apex_m"], a["v_min_kmh"]) == (800.0, 100.0)
        assert a["prominence_kmh"] == pytest.approx(200.0)
        assert (a["brake_on_m"], a["brake_off_m"]) == (640.0, 790.0)
        assert a["lift_m"] == 640.0
        assert (a["apex_start_m"], a["apex_end_m"], a["exit_end_m"]) == (800.0, 840.0, 920.0)
        assert bool(a["has_braking"]) and not bool(a["marginal"])

        assert (b["apex_m"], b["v_min_kmh"]) == (1800.0, 286.0)
        assert b["prominence_kmh"] == pytest.approx(14.0)
        assert pd.isna(b["brake_on_m"]) and pd.isna(b["brake_off_m"])
        assert b["lift_m"] == 1720.0
        assert (b["apex_start_m"], b["apex_end_m"], b["exit_end_m"]) == (1800.0, 1810.0, 1870.0)
        assert not bool(b["has_braking"]) and bool(b["marginal"])

    def test_left_max_is_the_plateau_after_the_previous_event(self, events: pd.DataFrame) -> None:
        assert events["left_max_m"].tolist() == [1930.0, 1000.0]

    def test_corner_labels(self, events: pd.DataFrame) -> None:
        assert events["corners"].tolist() == ["T1", "T2"]

    def test_no_corners_means_null_labels(self) -> None:
        events = detect_events(syn.traces(), None)
        assert events["corners"].isna().all()
        assert len(events) == 2

    def test_schema(self, events: pd.DataFrame) -> None:
        assert list(events.columns) == list(EVENT_SCHEMA)
        for column, dtype in EVENT_SCHEMA.items():
            assert str(events[column].dtype) == dtype, column
        assert events["event_id"].tolist() == [0, 1]

    def test_wrapped_session_gives_the_same_events_modulo_lap_length(self) -> None:
        shift = -78  # trough A lands at 20 m, braking starts before the line
        events = detect_events(syn.rolled_traces(shift), None)
        a = events.iloc[0]
        L = syn.LAP_LENGTH_M
        assert a["apex_m"] == 20.0
        assert a["brake_on_m"] == (640.0 + shift * 10) % L == 2860.0
        assert a["brake_off_m"] == 10.0
        assert (a["apex_start_m"], a["apex_end_m"], a["exit_end_m"]) == (20.0, 60.0, 140.0)
        assert a["prominence_kmh"] == pytest.approx(200.0)

    def test_parameters_change_the_outcome(self) -> None:
        loose = detect_events(syn.traces(), None, EventParams(min_prominence_kmh=4.0))
        assert len(loose) == 3
        strict = detect_events(syn.traces(), None, EventParams(min_prominence_kmh=50.0))
        assert len(strict) == 1
        wide = detect_events(syn.traces(), None, EventParams(apex_fraction=0.10))
        assert wide["apex_end_m"].iloc[0] - wide["apex_start_m"].iloc[0] >= 40.0

    def test_no_troughs_gives_an_empty_typed_table(self) -> None:
        flat = syn.traces().assign(speed=250.0)
        events = detect_events(flat, None)
        assert events.empty and list(events.columns) == list(EVENT_SCHEMA)

    def test_missing_channel_is_an_error(self) -> None:
        with pytest.raises(EventError):
            detect_events(syn.traces().drop(columns=["brake"]), None)

    def test_deterministic(self, events: pd.DataFrame) -> None:
        pd.testing.assert_frame_equal(events, detect_events(syn.traces(), placed_corners()))


class TestChainedEvents:
    """Exit that never recovers half its speed loss before the next trough."""

    @pytest.fixture()
    def traces(self) -> pd.DataFrame:
        d = syn.distance()
        v = np.full(len(d), 300.0)
        fall = (d >= 640) & (d <= 800)
        v[fall] = 300.0 - 200.0 * (d[fall] - 640.0) / 160.0
        rise1 = (d > 800) & (d <= 900)
        v[rise1] = 100.0 + 50.0 * (d[rise1] - 800.0) / 100.0
        # Second floor sits 4 km/h above the first so its prominence walk stops there.
        fall2 = (d > 900) & (d <= 1000)
        v[fall2] = 150.0 - 46.0 * (d[fall2] - 900.0) / 100.0
        rise2 = (d > 1000) & (d <= 1200)
        v[rise2] = 104.0 + 196.0 * (d[rise2] - 1000.0) / 200.0
        brake = ((d >= 640) & (d <= 780)).astype(float)
        throttle = np.where((d >= 640) & (d < 1100), 0.0, 100.0)
        return pd.DataFrame({"distance_m": d, "speed": v, "brake": brake, "throttle": throttle, "n_laps": 4},
                            index=pd.Index(np.arange(len(d)), name="grid_index"))

    def test_exit_is_cut_at_the_next_event_start(self, traces: pd.DataFrame) -> None:
        events = detect_events(traces, None)
        assert len(events) == 2
        first, second = events.iloc[0], events.iloc[1]
        assert first["exit_end_m"] == 900.0  # the speed maximum between the troughs
        assert second["lift_m"] == 900.0 and not bool(second["has_braking"])
        assert second["prominence_kmh"] == pytest.approx(46.0)
        assert first["exit_end_m"] <= min(second["lift_m"], second["apex_start_m"])


# --- F018: corners taken flat ------------------------------------------------

FLAT_N = 200
FLAT_GRID_M = 10.0
FLAT_LAP_M = FLAT_N * FLAT_GRID_M


def _well(centre: int, half: int, depth: float) -> np.ndarray:
    """A cosine speed well: ``depth`` km/h deep at the centre, zero at both edges."""
    out = np.zeros(FLAT_N)
    k = np.arange(centre - half, centre + half + 1)
    x = (k - centre) / half
    out[k % FLAT_N] = np.maximum(out[k % FLAT_N], depth * 0.5 * (1 + np.cos(np.pi * x)))
    return out


def _flat_traces(speed: np.ndarray, brake: np.ndarray, throttle: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"distance_m": np.arange(FLAT_N) * FLAT_GRID_M, "speed": speed, "brake": brake,
         "throttle": throttle, "n_laps": 4},
        index=pd.Index(np.arange(FLAT_N), name="grid_index"),
    )


def flat_out_traces(lift_before_band: bool = False) -> pd.DataFrame:
    """A 12 km/h kink at 1,520 m taken flat; its apex band is 1,510-1,540 m.

    The median driver is still at 95 % throttle *at the trough*, which is what
    used to put the lift inside the band. With ``lift_before_band`` the throttle
    drops to 85 % for the 40 m in front of the band and is re-applied inside it:
    a real lift, which a plain clamp to ``apex_start_m`` would hide.
    """
    throttle = np.full(FLAT_N, 95.0)
    if lift_before_band:
        throttle[147:152] = 85.0
    return _flat_traces(300.0 - _well(152, 10, 12.0), np.zeros(FLAT_N), throttle)


def hairpin_then_kink() -> pd.DataFrame:
    """A braked hairpin whose label window ends at 1,450 m, then the same kink."""
    speed = 300.0 - np.maximum(_well(137, 9, 30.0), _well(152, 10, 12.0))
    brake = np.zeros(FLAT_N)
    brake[129:137] = 1.0
    throttle = np.full(FLAT_N, 95.0)
    throttle[129:140] = 20.0
    return _flat_traces(speed, brake, throttle)


def markers(*distances: float) -> pd.DataFrame:
    """Circuit-info corner markers at the given aligned distances, T1 upward."""
    return pd.DataFrame({"number": np.arange(1, len(distances) + 1, dtype="int16"),
                         "letter": [""] * len(distances), "distance_m": list(distances)})


class TestFlatOutCorner:
    """The lift is sought before the apex band, not before the trough."""

    def test_the_lift_stops_at_the_apex_band(self) -> None:
        event = detect_events(flat_out_traces(), None).iloc[0]
        assert (event["apex_start_m"], event["apex_end_m"]) == (1510.0, 1540.0)
        assert event["lift_m"] == 1510.0
        assert not bool(event["has_braking"])

    def test_the_ordering_invariant_holds(self) -> None:
        events = detect_events(flat_out_traces(), None)
        assert (events["lift_m"] <= events["apex_start_m"]).all()
        assert (events["apex_start_m"] < events["apex_end_m"]).all()
        assert (events["apex_end_m"] <= events["exit_end_m"]).all()

    def test_the_throttle_is_still_open_at_the_trough(self) -> None:
        """Why searching to the trough failed: full throttle past the band."""
        traces = flat_out_traces()
        trough = int(detect_events(traces, None).iloc[0]["apex_m"] / FLAT_GRID_M)
        assert traces["throttle"].iloc[trough] >= EventParams().throttle_lift_pct

    def test_it_yields_no_entry_phase(self) -> None:
        events = detect_events(flat_out_traces(), None)
        phases = build_corner_phases(events, FLAT_LAP_M, FLAT_GRID_M)
        assert set(phases.loc[phases["event_id"].notna(), "phase"]) == {"apex", "exit"}

    def test_a_real_lift_before_the_band_is_kept(self) -> None:
        event = detect_events(flat_out_traces(lift_before_band=True), None).iloc[0]
        assert event["lift_m"] == 1470.0
        assert event["lift_m"] < event["apex_start_m"]

    def test_and_becomes_an_entry_phase(self) -> None:
        events = detect_events(flat_out_traces(lift_before_band=True), None)
        phases = build_corner_phases(events, FLAT_LAP_M, FLAT_GRID_M)
        entry = phases[phases["phase"] == "entry"]
        assert len(entry) == 1
        assert (float(entry.iloc[0]["start_m"]), float(entry.iloc[0]["end_m"])) == (1470.0, 1510.0)


class TestLabelFallback:
    """A flat-out event's window collapses onto the band, so reach behind it."""

    def test_the_default_reach_is_pinned(self) -> None:
        assert EventParams().fallback_margin_m == 60.0

    def test_a_marker_ahead_of_the_band_is_taken(self) -> None:
        assert detect_events(flat_out_traces(), markers(1460.0))["corners"].tolist() == ["T1"]

    def test_the_far_edge_of_the_reach_is_included(self) -> None:
        assert detect_events(flat_out_traces(), markers(1450.0))["corners"].tolist() == ["T1"]

    def test_a_marker_beyond_the_reach_is_left_alone(self) -> None:
        assert detect_events(flat_out_traces(), markers(1440.0))["corners"].isna().all()

    def test_it_does_not_fire_when_the_window_already_labels(self) -> None:
        events = detect_events(flat_out_traces(), markers(1490.0, 1460.0))
        assert events["corners"].tolist() == ["T1"]

    def test_a_marker_another_event_claimed_is_not_stolen(self) -> None:
        events = detect_events(hairpin_then_kink(), markers(1450.0))
        assert events["corners"].iloc[0] == "T1"
        assert pd.isna(events["corners"].iloc[1])

    def test_it_takes_the_unclaimed_marker_instead(self) -> None:
        events = detect_events(hairpin_then_kink(), markers(1450.0, 1470.0))
        assert events["corners"].tolist() == ["T1", "T2"]

    def test_the_reach_does_not_run_forward(self) -> None:
        events = detect_events(hairpin_then_kink(), markers(1470.0))
        assert pd.isna(events["corners"].iloc[0])
        assert events["corners"].iloc[1] == "T1"
