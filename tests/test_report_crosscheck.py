"""F016 cross-checks: counting brake applications, and reconciling with official timing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.report import crosscheck
from src.report.pairs import ReportError

SPACING_M = 10.0

#: Two corner events on a 1,000 m stretch. T1 is a single-stage braking zone;
#: T2 has a kink before it, which is where an extra application can appear.
EVENTS = pd.DataFrame([
    {"event_id": 0, "corners": "T1", "has_braking": True, "left_max_m": 100.0,
     "apex_m": 300.0, "exit_end_m": 400.0},
    {"event_id": 1, "corners": "T2", "has_braking": True, "left_max_m": 500.0,
     "apex_m": 800.0, "exit_end_m": 900.0},
])


def lap_grid(driver: str, lap: int, brake_ranges: list[tuple[float, float]],
             length_m: float = 1000.0) -> pd.DataFrame:
    """One lap's grid, with the brake on over the given metre ranges."""
    distance = np.arange(0.0, length_m + SPACING_M, SPACING_M)
    brake = np.zeros(len(distance), dtype=bool)
    for low, high in brake_ranges:
        brake |= (distance >= low) & (distance <= high)
    return pd.DataFrame({
        "driver": driver, "lap_number": lap, "grid_index": np.arange(len(distance)),
        "distance_m": distance, "brake": brake,
        "speed": np.full(len(distance), 200.0), "throttle": np.where(brake, 0.0, 100.0),
        "n_gear": 6, "source_gap_m": 8.0,
        "elapsed_time": distance / 55.0,
    })


def corner_metric_rows(driver: str, lap: int, brake_on: dict[int, float]) -> pd.DataFrame:
    return pd.DataFrame([
        {"driver": driver, "lap_number": lap, "event_id": event_id, "corners": "T",
         "brake_on_m": metres, "brake_gap_m": 9.0, "v_min_kmh": 120.0}
        for event_id, metres in brake_on.items()
    ])


class TestBrakeRuns:
    def test_one_application_is_one_run(self) -> None:
        grid = lap_grid("AAA", 1, [(200.0, 260.0)])
        assert crosscheck.brake_runs(grid, ("AAA", 1), 100.0, 300.0) == [(200.0, 260.0)]

    def test_a_gap_between_applications_makes_two(self) -> None:
        """The dab and the braking zone are separate events, not one long one."""
        grid = lap_grid("AAA", 1, [(560.0, 580.0), (700.0, 790.0)])
        assert crosscheck.brake_runs(grid, ("AAA", 1), 500.0, 800.0) == [(560.0, 580.0), (700.0, 790.0)]

    def test_no_braking_is_an_empty_list_not_an_error(self) -> None:
        assert crosscheck.brake_runs(lap_grid("AAA", 1, []), ("AAA", 1), 100.0, 300.0) == []

    def test_it_stops_at_the_window(self) -> None:
        grid = lap_grid("AAA", 1, [(200.0, 400.0)])
        assert crosscheck.brake_runs(grid, ("AAA", 1), 100.0, 300.0) == [(200.0, 300.0)]

    def test_an_unknown_lap_is_an_error(self) -> None:
        with pytest.raises(ReportError, match="ZZZ lap 4"):
            crosscheck.brake_runs(lap_grid("AAA", 1, []), ("ZZZ", 4), 0.0, 100.0)


class TestApplications:
    def test_each_application_is_a_row_and_knows_how_many_there_were(self) -> None:
        grid = lap_grid("AAA", 1, [(200.0, 260.0), (560.0, 580.0), (700.0, 790.0)])
        applications = crosscheck.brake_applications(grid, EVENTS, ("AAA", 1))
        assert len(applications) == 3
        assert list(applications[applications.event_id == 1]["of"]) == [2, 2]

    def test_the_last_one_before_the_apex_is_the_one_that_matters(self) -> None:
        grid = lap_grid("AAA", 1, [(200.0, 260.0), (560.0, 580.0), (700.0, 790.0)])
        leading = crosscheck.leading_application(crosscheck.brake_applications(grid, EVENTS, ("AAA", 1)))
        assert leading.loc[1, "on_m"] == 700.0, "the dab at 560 m is not the braking point"
        assert leading.loc[0, "on_m"] == 200.0

    def test_a_release_shorter_than_the_grid_is_not_a_second_application(self) -> None:
        """A limitation, stated rather than papered over: two applications
        separated by less than one grid step read as one. At 10 m and racing
        speed that is under a tenth of a second off the brake."""
        merged = crosscheck.brake_runs(lap_grid("AAA", 1, [(610.0, 650.0), (660.0, 780.0)]),
                                       ("AAA", 1), 500.0, 800.0)
        assert merged == [(610.0, 780.0)]
        separate = crosscheck.brake_runs(lap_grid("AAA", 1, [(610.0, 650.0), (670.0, 780.0)]),
                                         ("AAA", 1), 500.0, 800.0)
        assert separate == [(610.0, 650.0), (670.0, 780.0)]

    def test_an_event_with_no_braking_still_gets_a_row(self) -> None:
        """A missing row would silently drop the event from every comparison."""
        applications = crosscheck.brake_applications(lap_grid("AAA", 1, []), EVENTS, ("AAA", 1))
        assert len(applications) == 2
        assert applications["of"].tolist() == [0, 0]


class TestBrakingComparison:
    def build(self, ranges_a, ranges_b, metric_a, metric_b) -> pd.DataFrame:
        grid = pd.concat([lap_grid("AAA", 1, ranges_a), lap_grid("BBB", 1, ranges_b)], ignore_index=True)
        metrics = pd.concat([corner_metric_rows("AAA", 1, metric_a),
                             corner_metric_rows("BBB", 1, metric_b)], ignore_index=True)
        return crosscheck.braking_comparison(grid, EVENTS, metrics, ("AAA", 1), ("BBB", 1))

    def test_same_braking_point_agrees(self) -> None:
        frame = self.build([(200.0, 260.0)], [(200.0, 260.0)], {0: 200.0}, {0: 200.0})
        assert frame.loc[0, "verdict"] == "agree"

    def test_an_extra_dab_makes_the_metric_report_a_difference_that_is_not_one(self) -> None:
        """A lap that dabs at the kink reports braking 140 m earlier than one
        that does not, while both brake for the corner in the same place."""
        frame = self.build(ranges_a=[(560.0, 580.0), (700.0, 790.0)], ranges_b=[(700.0, 790.0)],
                           metric_a={1: 560.0}, metric_b={1: 700.0})
        assert frame.loc[1, "metric_delta_m"] == pytest.approx(140.0)
        assert frame.loc[1, "leading_delta_m"] == pytest.approx(0.0)
        assert frame.loc[1, "verdict"] == "metric artefact"
        assert bool(frame.loc[1, "extra_dab"]) is True

    def test_a_real_difference_survives_both_readings(self) -> None:
        frame = self.build([(200.0, 290.0)], [(150.0, 290.0)], {0: 200.0}, {0: 150.0})
        assert frame.loc[0, "verdict"] == "confirmed"
        assert frame.loc[0, "leading_delta_m"] == pytest.approx(-50.0)

    def test_two_stage_braking_on_both_laps_is_a_question_of_definition(self) -> None:
        """Both laps brake twice, and the two readings disagree about which
        application is the braking point. Neither is wrong; the single number is."""
        frame = self.build(ranges_a=[(600.0, 640.0), (700.0, 790.0)],
                           ranges_b=[(610.0, 650.0), (670.0, 780.0)],
                           metric_a={1: 600.0}, metric_b={1: 610.0})
        assert frame.loc[1, "applications_a"] == frame.loc[1, "applications_b"] == 2
        assert frame.loc[1, "metric_delta_m"] == pytest.approx(10.0), "first applications agree"
        assert frame.loc[1, "leading_delta_m"] == pytest.approx(-30.0), "second ones do not"
        assert frame.loc[1, "verdict"] == "definition-sensitive"

    def test_the_window_is_the_d6_one(self) -> None:
        assert crosscheck.BRAKE_WINDOW_M == 20.0
        frame = self.build([(200.0, 290.0)], [(190.0, 290.0)], {0: 200.0}, {0: 190.0})
        assert frame.loc[0, "verdict"] == "agree", "10 m is inside the window"


class TestBrakingSummary:
    def test_an_outlier_is_explained_when_it_is_an_artefact(self) -> None:
        grid = pd.concat([lap_grid("AAA", 1, [(560.0, 580.0), (700.0, 790.0)]),
                          lap_grid("BBB", 1, [(700.0, 790.0)])], ignore_index=True)
        metrics = pd.concat([corner_metric_rows("AAA", 1, {0: 200.0, 1: 560.0}),
                             corner_metric_rows("BBB", 1, {0: 200.0, 1: 700.0})], ignore_index=True)
        frame = crosscheck.braking_comparison(grid, EVENTS, metrics, ("AAA", 1), ("BBB", 1))
        assert frame.loc[0, "applications_a"] == 0, "an event never braked for counts zero, not missing"
        summary = crosscheck.braking_summary(frame)
        assert summary["metric_artefact"] == ["T2"]
        assert summary["metric_outliers_explained"] is True
        assert summary["multi_application_events"] == ["T2"]

    def test_an_unexplained_outlier_fails_the_check(self) -> None:
        """A metric outlier that is neither confirmed nor caused by an extra
        application means the two readings disagree for a reason nobody knows."""
        frame = pd.DataFrame([{
            "corners": "T1", "has_braking": True, "applications_a": 1, "applications_b": 1,
            "metric_delta_m": 100.0, "leading_delta_m": 0.0,
            "metric_outside_window": True, "leading_outside_window": False,
            "extra_dab": False, "multi_application": False,
        }])
        frame["verdict"] = [crosscheck._verdict(row, 20.0) for _, row in frame.iterrows()]
        assert frame.loc[0, "verdict"] == "definition-sensitive"
        assert crosscheck.braking_summary(frame)["metric_outliers_explained"] is False


class TestDabPrevalence:
    def test_it_counts_the_laps_that_brake_in_the_stretch(self) -> None:
        grid = pd.concat([
            lap_grid("AAA", 1, [(560.0, 580.0), (700.0, 790.0)]),
            lap_grid("AAA", 2, [(560.0, 580.0), (700.0, 790.0)]),
            lap_grid("BBB", 1, [(700.0, 790.0)]),
        ], ignore_index=True)
        prevalence = crosscheck.dab_prevalence(grid, EVENTS, 1, (500.0, 690.0))
        assert int(prevalence["dabbed"].sum()) == 2
        assert prevalence[~prevalence["dabbed"]]["driver"].tolist() == ["BBB"]


class TestSectorReconciliation:
    """The identity the check rests on: the disagreement between a grid sector
    gap and an official one should be the difference of the two laps' own
    registration residuals, which F010 measured per lap."""

    def frames(self, s_official_a, s_official_b, residual_a, residual_b, curve_b_offsets):
        lap_summary = pd.DataFrame([
            {"driver": "AAA", "lap_number": 1, "lap_time_s": 60.0,
             "s1_official_s": s_official_a[0], "s2_official_s": s_official_a[1], "s3_official_s": s_official_a[2]},
            {"driver": "BBB", "lap_number": 1, "lap_time_s": 60.2,
             "s1_official_s": s_official_b[0], "s2_official_s": s_official_b[1], "s3_official_s": s_official_b[2]},
        ])
        ground_truth = pd.DataFrame([
            {"driver": "AAA", "lap_number": 1, "s1_residual_s": residual_a[0],
             "s2_residual_s": residual_a[1], "s3_residual_s": residual_a[2]},
            {"driver": "BBB", "lap_number": 1, "s1_residual_s": residual_b[0],
             "s2_residual_s": residual_b[1], "s3_residual_s": residual_b[2]},
        ])
        indices = np.arange(0, 101)
        curve_a = indices * 0.2
        curve_b = curve_a.copy()
        for index, offset in curve_b_offsets.items():
            curve_b[index:] += offset
        delta_t = pd.concat([
            pd.DataFrame({"driver": "AAA", "lap_number": 1, "grid_index": indices, "t_s": curve_a}),
            pd.DataFrame({"driver": "BBB", "lap_number": 1, "grid_index": indices, "t_s": curve_b}),
        ], ignore_index=True)
        return lap_summary, ground_truth, delta_t

    def test_a_perfectly_registered_pair_leaves_nothing_unexplained(self) -> None:
        lap_summary, ground_truth, delta_t = self.frames(
            s_official_a=(20.0, 20.0, 20.0), s_official_b=(20.1, 20.0, 20.1),
            residual_a=(0.0, 0.0, 0.0), residual_b=(0.0, 0.0, 0.0),
            curve_b_offsets={1: 0.1, 51: 0.0, 76: 0.1})
        frame = crosscheck.sector_reconciliation(lap_summary, ground_truth, delta_t,
                                                 (300.0, 750.0), ("AAA", 1), ("BBB", 1))
        assert frame["unexplained_s"].abs().max() < 1e-9

    def test_a_registration_shift_lands_in_the_residual_column_not_the_remainder(self) -> None:
        """This is the whole point: the grid disagrees with the timing feed, and
        the amount is already known per lap."""
        lap_summary, ground_truth, delta_t = self.frames(
            s_official_a=(20.0, 20.0, 20.0), s_official_b=(20.1, 20.0, 20.1),
            residual_a=(0.0, 0.0, 0.0), residual_b=(-0.05, 0.05, 0.0),
            curve_b_offsets={1: 0.05, 51: 0.05, 76: 0.1})
        frame = crosscheck.sector_reconciliation(lap_summary, ground_truth, delta_t,
                                                 (300.0, 750.0), ("AAA", 1), ("BBB", 1))
        assert frame.loc["s1", "difference_s"] == pytest.approx(-0.05, abs=1e-9)
        assert frame.loc["s1", "f010_residual_difference_s"] == pytest.approx(-0.05)
        assert frame.loc["s1", "unexplained_s"] == pytest.approx(0.0, abs=1e-9)

    def test_laps_that_share_no_grid_are_an_error_not_a_silent_nan(self) -> None:
        lap_summary, ground_truth, delta_t = self.frames(
            (20.0, 20.0, 20.0), (20.0, 20.0, 20.0), (0, 0, 0), (0, 0, 0), {})
        delta_t = delta_t[~((delta_t.driver == "BBB") & (delta_t.grid_index > 10))]
        with pytest.raises(ReportError, match="grid index"):
            crosscheck.sector_reconciliation(lap_summary, ground_truth, delta_t,
                                             (300.0, 750.0), ("AAA", 1), ("BBB", 1))
