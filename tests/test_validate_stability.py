"""F019: V_min stability is measured on push laps, not on every accurate lap.

FastF1's `is_accurate`, which the pipeline applies upstream, drops out-laps and
pit laps but keeps a qualifying cool-down lap: complete, timed, green-flag, and
tens of km/h slower through every corner. Leaving those in a group measures the
run plan rather than the driver's repeatability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validate.stability import (
    DEFAULT_MIN_LAPS,
    DEFAULT_PUSH_FRACTION,
    push_laps,
    v_min_stability,
)

PUSH_V_MIN = [120.0, 121.0, 119.0, 120.5]
COOLDOWN_V_MIN = [80.0, 78.0]


def session(push_time: float = 90.0, cooldown_time: float = 120.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One driver: four push laps at a consistent V_min, two cool-down laps 40 km/h slower."""
    n = len(PUSH_V_MIN) + len(COOLDOWN_V_MIN)
    metrics = pd.DataFrame(
        {
            "driver": ["AAA"] * n,
            "lap_number": list(range(1, n + 1)),
            "event_id": [0] * n,
            "corners": ["T1"] * n,
            "v_min_kmh": PUSH_V_MIN + COOLDOWN_V_MIN,
        }
    )
    laps = pd.DataFrame(
        {
            "driver": ["AAA"] * n,
            "lap_number": list(range(1, n + 1)),
            "compound": ["SOFT"] * n,
            "lap_time": [push_time] * len(PUSH_V_MIN) + [cooldown_time] * len(COOLDOWN_V_MIN),
        }
    )
    return metrics, laps


class TestPushLaps:
    def test_the_default_fraction_is_pinned(self) -> None:
        assert DEFAULT_PUSH_FRACTION == 1.07

    def test_a_cool_down_lap_is_not_a_push_lap(self) -> None:
        _, laps = session()
        assert push_laps(laps)["lap_number"].tolist() == [1, 2, 3, 4]

    def test_the_cut_is_per_driver(self) -> None:
        """A slower driver's own best sets their cut, not the session's best."""
        laps = pd.DataFrame(
            {
                "driver": ["AAA", "AAA", "BBB", "BBB"],
                "lap_number": [1, 2, 1, 2],
                "compound": ["SOFT"] * 4,
                "lap_time": [90.0, 130.0, 100.0, 104.0],
            }
        )
        assert push_laps(laps)["driver"].tolist() == ["AAA", "BBB", "BBB"]

    def test_a_lap_with_no_time_is_dropped(self) -> None:
        _, laps = session()
        laps.loc[0, "lap_time"] = np.nan
        assert push_laps(laps)["lap_number"].tolist() == [2, 3, 4]

    def test_a_lap_frame_without_times_is_an_error(self) -> None:
        _, laps = session()
        with pytest.raises(ValueError, match="lap_time"):
            push_laps(laps.drop(columns=["lap_time"]))

    def test_a_looser_fraction_admits_the_cool_down_laps(self) -> None:
        _, laps = session()
        assert len(push_laps(laps, push_fraction=1.5)) == 6


class TestStabilityUsesPushLapsOnly:
    def test_the_cool_down_laps_are_what_moved_the_spread(self) -> None:
        metrics, laps = session()
        every = np.std(PUSH_V_MIN + COOLDOWN_V_MIN, ddof=1)
        push_only = np.std(PUSH_V_MIN, ddof=1)
        out = v_min_stability(metrics, laps, min_laps=DEFAULT_MIN_LAPS)
        assert len(out) == 1
        assert out["n_laps"].iloc[0] == len(PUSH_V_MIN)
        assert out["v_min_std_kmh"].iloc[0] == pytest.approx(push_only, abs=1e-4)
        assert push_only < 1.0 < every  # the two cool-down laps dominated it

    def test_a_group_without_enough_push_laps_drops_out(self) -> None:
        metrics, laps = session()
        laps.loc[[1, 2], "lap_time"] = 130.0  # only two push laps left
        assert v_min_stability(metrics, laps, min_laps=DEFAULT_MIN_LAPS).empty

    def test_a_looser_fraction_puts_the_spread_back(self) -> None:
        metrics, laps = session()
        out = v_min_stability(metrics, laps, min_laps=DEFAULT_MIN_LAPS, push_fraction=1.5)
        assert out["n_laps"].iloc[0] == 6
        assert out["v_min_std_kmh"].iloc[0] == pytest.approx(np.std(PUSH_V_MIN + COOLDOWN_V_MIN, ddof=1), abs=1e-4)

    def test_a_corner_metric_with_no_matching_lap_is_dropped(self) -> None:
        metrics, laps = session()
        metrics.loc[len(metrics)] = ["AAA", 99, 0, "T1", 118.0]
        out = v_min_stability(metrics, laps, min_laps=DEFAULT_MIN_LAPS)
        assert out["n_laps"].iloc[0] == len(PUSH_V_MIN)
