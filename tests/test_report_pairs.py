"""F016 pair decomposition, on frames small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.report import pairs
from src.report.pairs import ReportError

#: Four micro-sectors: a straight, then a corner event of three phases, then a
#: second event of one. Boundaries are explicit rather than cumulative so the
#: test would notice if the code went back to inferring them.
MICROSECTORS = pd.DataFrame([
    {"grain": "corner_phase", "microsector_id": 0, "phase": "straight", "event_id": None,
     "corners": None, "start_m": 0.0, "end_m": 300.0},
    {"grain": "corner_phase", "microsector_id": 1, "phase": "braking", "event_id": 0,
     "corners": "T1", "start_m": 300.0, "end_m": 400.0},
    {"grain": "corner_phase", "microsector_id": 2, "phase": "apex", "event_id": 0,
     "corners": "T1", "start_m": 400.0, "end_m": 450.0},
    {"grain": "corner_phase", "microsector_id": 3, "phase": "exit", "event_id": 0,
     "corners": "T1", "start_m": 450.0, "end_m": 550.0},
    {"grain": "corner_phase", "microsector_id": 4, "phase": "braking", "event_id": 1,
     "corners": "T2", "start_m": 550.0, "end_m": 700.0},
    {"grain": "fixed_100m", "microsector_id": 0, "phase": "bin", "event_id": None,
     "corners": None, "start_m": 0.0, "end_m": 100.0},
])

SUMMARY = pd.DataFrame([
    {"grain": "corner_phase", "microsector_id": 0, "within_driver_std_s": 0.03},
    {"grain": "corner_phase", "microsector_id": 1, "within_driver_std_s": 0.04},
    {"grain": "corner_phase", "microsector_id": 2, "within_driver_std_s": 0.03},
    {"grain": "corner_phase", "microsector_id": 3, "within_driver_std_s": 0.12},
    {"grain": "corner_phase", "microsector_id": 4, "within_driver_std_s": 0.05},
    {"grain": "fixed_100m", "microsector_id": 0, "within_driver_std_s": 0.01},
])


def lap_times(driver: str, lap: int, times: list[float], partial: list[bool] | None = None) -> pd.DataFrame:
    partial = partial or [False] * len(times)
    return pd.DataFrame({
        "driver": driver, "lap_number": lap, "grain": "corner_phase",
        "microsector_id": range(len(times)), "time_s": times, "partial": partial,
    })


#: A is flat; B loses 0.10 s at the T1 apex and gains 0.05 s on the straight.
#: Everything else is identical, so every total is known in advance.
TIMES = pd.concat([
    lap_times("AAA", 1, [4.00, 1.50, 0.80, 1.90, 2.20]),
    lap_times("BBB", 1, [3.95, 1.50, 0.90, 1.90, 2.20]),
    lap_times("AAA", 2, [4.10, 1.55, 0.82, 1.92, 2.25]),
    lap_times("BBB", 2, [4.00, 1.52, 0.95, 1.90, 2.30]),
], ignore_index=True)

LAP_SUMMARY = pd.DataFrame([
    {"driver": "AAA", "lap_number": 1, "lap_time_s": 10.40},
    {"driver": "AAA", "lap_number": 2, "lap_time_s": 10.64},
    {"driver": "BBB", "lap_number": 1, "lap_time_s": 10.45},
    {"driver": "BBB", "lap_number": 2, "lap_time_s": 10.67},
    {"driver": "AAA", "lap_number": 3, "lap_time_s": None},
])


class TestQuadrature:
    def test_independent_errors_add_in_quadrature(self) -> None:
        assert pairs.quadrature(pd.Series([0.03, 0.04])) == pytest.approx(0.05)

    def test_it_is_not_a_sum(self) -> None:
        """Adding sigmas linearly would inflate every error bar in the report."""
        assert pairs.quadrature(pd.Series([0.1, 0.1])) == pytest.approx(0.1414, abs=1e-4)

    def test_missing_values_are_skipped_not_counted_as_zero(self) -> None:
        assert pairs.quadrature(pd.Series([0.03, np.nan, 0.04])) == pytest.approx(0.05)

    def test_nothing_finite_is_not_a_number(self) -> None:
        assert np.isnan(pairs.quadrature(pd.Series([np.nan])))


class TestSectorFrame:
    def test_it_takes_boundaries_from_the_segmentation_not_from_lengths(self) -> None:
        frame = pairs.sector_frame(MICROSECTORS, SUMMARY)
        assert frame.loc[1, "start_m"] == 300.0 and frame.loc[1, "end_m"] == 400.0
        assert frame.loc[1, "length_m"] == 100.0

    def test_it_keeps_only_the_grain_asked_for(self) -> None:
        assert len(pairs.sector_frame(MICROSECTORS, SUMMARY, "corner_phase")) == 5
        assert len(pairs.sector_frame(MICROSECTORS, SUMMARY, "fixed_100m")) == 1

    def test_an_absent_grain_is_an_error(self) -> None:
        with pytest.raises(ReportError, match="no micro-sectors"):
            pairs.sector_frame(MICROSECTORS, SUMMARY, "no_such_grain")


class TestDecompose:
    def test_designed_differences_come_back_exactly(self) -> None:
        pair = pairs.decompose(TIMES, MICROSECTORS, SUMMARY, ("AAA", 1), ("BBB", 1))
        assert pair.sectors.loc[0, "delta_s"] == pytest.approx(-0.05)
        assert pair.sectors.loc[2, "delta_s"] == pytest.approx(+0.10)
        assert pair.total_s == pytest.approx(0.05)

    def test_events_and_phases_partition_the_same_total(self) -> None:
        pair = pairs.decompose(TIMES, MICROSECTORS, SUMMARY, ("AAA", 1), ("BBB", 1))
        assert pair.phases["delta_s"].sum() == pytest.approx(pair.total_s)
        straights = pair.phases.loc["straight", "delta_s"]
        assert pair.events["delta_s"].sum() + straights == pytest.approx(pair.total_s)

    def test_an_event_carries_the_quadrature_of_its_phases(self) -> None:
        pair = pairs.decompose(TIMES, MICROSECTORS, SUMMARY, ("AAA", 1), ("BBB", 1))
        assert pair.events.loc[0, "sigma_s"] == pytest.approx(np.sqrt(0.04**2 + 0.03**2 + 0.12**2))
        assert pair.events.loc[0, "delta_s"] == pytest.approx(0.10)

    def test_a_lap_against_itself_is_zero_everywhere(self) -> None:
        """The most basic thing a decomposition must not get wrong."""
        pair = pairs.decompose(TIMES, MICROSECTORS, SUMMARY, ("AAA", 1), ("AAA", 1))
        assert pair.total_s == pytest.approx(0.0)
        assert (pair.sectors["delta_s"].abs() < 1e-12).all()
        assert (pair.events["delta_s"].abs() < 1e-12).all()

    def test_a_sector_either_lap_did_not_finish_is_left_out_of_every_total(self) -> None:
        """Subtracting a whole sector from part of one compares two distances."""
        truncated = TIMES.copy()
        mask = (truncated.driver == "BBB") & (truncated.lap_number == 1) & (truncated.microsector_id == 4)
        truncated.loc[mask, "partial"] = True
        pair = pairs.decompose(truncated, MICROSECTORS, SUMMARY, ("AAA", 1), ("BBB", 1))
        assert pair.complete == 4 and pair.of == 5
        assert 1 not in pair.events.index, "the partial sector's event drops out with it"
        assert pair.total_s == pytest.approx(0.05)

    def test_the_ratio_is_the_delta_against_its_own_spread(self) -> None:
        pair = pairs.decompose(TIMES, MICROSECTORS, SUMMARY, ("AAA", 1), ("BBB", 1))
        assert pair.sectors.loc[2, "ratio"] == pytest.approx(0.10 / 0.03)
        assert pair.sectors.loc[3, "ratio"] == pytest.approx(0.0)

    def test_above_noise_is_empty_when_nothing_clears_two_sigma(self) -> None:
        pair = pairs.decompose(TIMES, MICROSECTORS, SUMMARY, ("AAA", 1), ("BBB", 1))
        assert pair.above_noise.empty, "0.10 s against 0.13 s is not a finding"

    def test_a_missing_lap_is_named_in_the_error(self) -> None:
        with pytest.raises(ReportError, match="CCC lap 9"):
            pairs.decompose(TIMES, MICROSECTORS, SUMMARY, ("AAA", 1), ("CCC", 9))

    def test_the_label_reads_like_the_document(self) -> None:
        pair = pairs.decompose(TIMES, MICROSECTORS, SUMMARY, ("AAA", 1), ("BBB", 2))
        assert pair.label == "AAA L1 vs BBB L2"


class TestTimedLaps:
    def test_laps_without_a_time_are_not_pace(self) -> None:
        assert pairs.timed_laps(LAP_SUMMARY, "AAA") == [1, 2]

    def test_they_come_back_quickest_first(self) -> None:
        assert pairs.timed_laps(LAP_SUMMARY, "BBB") == [1, 2]


class TestPairings:
    def test_every_lap_meets_every_lap(self) -> None:
        frame = pairs.pairings(TIMES, MICROSECTORS, SUMMARY, LAP_SUMMARY, "AAA", "BBB")
        assert len(frame) == 4
        assert set(frame.index) == {"AAA L1 vs BBB L1", "AAA L1 vs BBB L2",
                                    "AAA L2 vs BBB L1", "AAA L2 vs BBB L2"}

    def test_each_row_carries_the_official_gap_beside_the_reconstruction(self) -> None:
        frame = pairs.pairings(TIMES, MICROSECTORS, SUMMARY, LAP_SUMMARY, "AAA", "BBB")
        assert frame.loc["AAA L1 vs BBB L1", "gap_s"] == pytest.approx(0.05)
        assert frame.loc["AAA L1 vs BBB L1", "total_s"] == pytest.approx(0.05)

    def test_corner_events_become_columns(self) -> None:
        frame = pairs.pairings(TIMES, MICROSECTORS, SUMMARY, LAP_SUMMARY, "AAA", "BBB")
        assert {"T1", "T2", "straights_s"} <= set(frame.columns)

    def test_a_driver_with_no_timed_lap_is_an_error(self) -> None:
        with pytest.raises(ReportError, match="no timed laps"):
            pairs.pairings(TIMES, MICROSECTORS, SUMMARY, LAP_SUMMARY, "AAA", "ZZZ")


class TestConsistency:
    def test_a_corner_that_always_costs_the_same_driver_is_flagged(self) -> None:
        """T1 is designed to be positive on every pairing; T2 is not designed at all."""
        frame = pairs.pairings(TIMES, MICROSECTORS, SUMMARY, LAP_SUMMARY, "AAA", "BBB")
        noise = pairs.event_noise(MICROSECTORS, SUMMARY)
        agreement = pairs.consistency(frame, noise)
        assert agreement.loc["T1", "positive"] == 4
        assert agreement.loc["T1", "pairings"] == 4
        assert bool(agreement.loc["T1", "always_same_sign"]) is True

    def test_sign_consistency_counts_a_negative_corner_too(self) -> None:
        """Always-negative is as consistent as always-positive."""
        flipped = TIMES.copy()
        mask = (flipped.driver == "BBB") & (flipped.microsector_id == 2)
        flipped.loc[mask, "time_s"] = flipped.loc[mask, "time_s"] - 0.30
        frame = pairs.pairings(flipped, MICROSECTORS, SUMMARY, LAP_SUMMARY, "AAA", "BBB")
        agreement = pairs.consistency(frame, pairs.event_noise(MICROSECTORS, SUMMARY))
        assert agreement.loc["T1", "positive"] == 0
        assert bool(agreement.loc["T1", "always_same_sign"]) is True

    def test_it_counts_pairings_large_enough_to_notice_alone(self) -> None:
        frame = pairs.pairings(TIMES, MICROSECTORS, SUMMARY, LAP_SUMMARY, "AAA", "BBB")
        noise = pairs.event_noise(MICROSECTORS, SUMMARY)
        agreement = pairs.consistency(frame, noise)
        assert agreement.loc["T1", "above_2_sigma"] == 0, "0.1 s against 2x0.13 s clears nothing"

    def test_event_noise_is_labelled_the_way_the_columns_are(self) -> None:
        noise = pairs.event_noise(MICROSECTORS, SUMMARY)
        assert set(noise.index) == {"T1", "T2"}
        assert noise["T2"] == pytest.approx(0.05)


class TestControl:
    def test_a_driver_against_his_own_lap_is_the_floor(self) -> None:
        own = pairs.control(TIMES, MICROSECTORS, SUMMARY, "AAA", 1, 2)
        assert own.label == "AAA L1 vs AAA L2"
        assert own.events.loc[0, "delta_s"] == pytest.approx(0.05 + 0.02 + 0.02)
