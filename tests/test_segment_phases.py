"""F009 micro-sector tables: partition, wrap handling, fixed bins."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.segment.events import detect_events
from src.segment.phases import (
    GRAIN_CORNER_PHASE,
    GRAIN_FIXED_100M,
    MICROSECTOR_SCHEMA,
    SegmentationError,
    build_corner_phases,
    build_fixed_bins,
    check_partition,
    event_intervals,
)
from tests import synthetic_session as syn


@pytest.fixture()
def events() -> pd.DataFrame:
    return detect_events(syn.traces(), None)


@pytest.fixture()
def phases(events: pd.DataFrame) -> pd.DataFrame:
    return build_corner_phases(events, syn.LAP_LENGTH_M)


class TestCornerPhases:
    def test_exact_designed_table(self, phases: pd.DataFrame) -> None:
        got = [
            (r.phase, float(r.start_m), float(r.end_m), None if pd.isna(r.event_id) else int(r.event_id))
            for r in phases.itertuples()
        ]
        assert got == syn.EXPECTED_PHASES

    def test_is_an_exact_partition(self, phases: pd.DataFrame) -> None:
        assert check_partition(phases, syn.LAP_LENGTH_M)
        assert phases["microsector_id"].tolist() == list(range(len(phases)))
        assert (phases["grain"] == GRAIN_CORNER_PHASE).all()

    def test_indices_match_distances(self, phases: pd.DataFrame) -> None:
        np.testing.assert_array_equal(phases["start_index"], phases["start_m"] / 10)
        np.testing.assert_array_equal(phases["end_index"], phases["end_m"] / 10)

    def test_schema(self, phases: pd.DataFrame) -> None:
        assert list(phases.columns) == list(MICROSECTOR_SCHEMA)
        for column, dtype in MICROSECTOR_SCHEMA.items():
            assert str(phases[column].dtype) == dtype, column

    def test_marginal_and_labels_travel_with_the_event(self) -> None:
        events = detect_events(syn.traces(), None).assign(corners=["T1", "T2"])
        phases = build_corner_phases(events, syn.LAP_LENGTH_M)
        b = phases[phases["event_id"] == 1]
        assert b["marginal"].all() and (b["corners"] == "T2").all()
        straights = phases[phases["phase"] == "straight"]
        assert straights["corners"].isna().all() and not straights["marginal"].any()

    def test_no_events_is_one_straight(self) -> None:
        empty = detect_events(syn.traces().assign(speed=250.0), None)
        phases = build_corner_phases(empty, syn.LAP_LENGTH_M)
        assert len(phases) == 1 and phases["phase"].iloc[0] == "straight"
        assert check_partition(phases, syn.LAP_LENGTH_M)

    def test_event_across_the_line_is_split_into_two_sectors(self) -> None:
        events = detect_events(syn.rolled_traces(-78), None)
        phases = build_corner_phases(events, syn.LAP_LENGTH_M)
        braking = phases[(phases["phase"] == "braking") & (phases["event_id"] == 0)]
        assert [(float(r.start_m), float(r.end_m)) for r in braking.itertuples()] == [(0.0, 10.0), (2860.0, 3000.0)]
        assert check_partition(phases, syn.LAP_LENGTH_M)
        assert len(phases) == len(syn.EXPECTED_PHASES) + 1 - 1  # one straight fewer, one braking piece more

    def test_overlapping_events_raise(self, events: pd.DataFrame) -> None:
        broken = events.copy()
        broken.loc[0, "exit_end_m"] = 1750.0  # runs into event 1's entry at 1720
        with pytest.raises(SegmentationError, match="overlap"):
            build_corner_phases(broken, syn.LAP_LENGTH_M)

    def test_trail_braking_to_the_apex_has_no_entry(self, events: pd.DataFrame) -> None:
        trail = events.copy()
        trail.loc[0, "brake_off_m"] = 820.0  # released inside the apex window
        intervals = event_intervals(trail.iloc[0], syn.LAP_LENGTH_M)
        assert [p for p, *_ in intervals] == ["braking", "apex", "exit"]
        assert intervals[0][1:] == (640.0, 800.0)


class TestFixedBins:
    def test_designed_session_tiles_exactly(self) -> None:
        bins = build_fixed_bins(syn.LAP_LENGTH_M)
        assert len(bins) == 30
        assert check_partition(bins, syn.LAP_LENGTH_M)
        assert (bins["grain"] == GRAIN_FIXED_100M).all() and (bins["phase"] == "bin").all()

    def test_last_bin_is_truncated(self) -> None:
        bins = build_fixed_bins(5730.0)
        assert len(bins) == 58
        assert float(bins["end_m"].iloc[-1] - bins["start_m"].iloc[-1]) == pytest.approx(30.0)
        assert check_partition(bins, 5730.0)

    def test_exact_multiple_has_no_stub(self) -> None:
        assert len(build_fixed_bins(5700.0)) == 57

    def test_bad_inputs(self) -> None:
        with pytest.raises(SegmentationError):
            build_fixed_bins(0.0)
        with pytest.raises(SegmentationError):
            build_fixed_bins(1000.0, bin_m=0.0)


class TestCheckPartition:
    def test_detects_a_gap_and_an_overlap(self, phases: pd.DataFrame) -> None:
        gap = phases.copy()
        gap.loc[1, "start_m"] = 650.0
        assert not check_partition(gap, syn.LAP_LENGTH_M)
        overlap = phases.copy()
        overlap.loc[1, "start_m"] = 630.0
        assert not check_partition(overlap, syn.LAP_LENGTH_M)

    def test_detects_wrong_extent_and_empty_sector(self, phases: pd.DataFrame) -> None:
        assert not check_partition(phases, syn.LAP_LENGTH_M + 10)
        assert not check_partition(phases.iloc[:0], syn.LAP_LENGTH_M)
        tiny = phases.copy()
        tiny.loc[2, "end_m"] = tiny.loc[2, "start_m"] + 1.0
        tiny.loc[3, "start_m"] = tiny.loc[2, "end_m"]
        assert not check_partition(tiny, syn.LAP_LENGTH_M)
