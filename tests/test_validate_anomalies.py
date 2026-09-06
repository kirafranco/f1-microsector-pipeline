"""F022: per-lap timing anomalies, flagged rather than averaged into a session."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validate.anomalies import (
    ANOMALY_S,
    CANCEL_FRACTION,
    lap_outliers,
    sector_split_outliers,
)


def laps(*rows: tuple[float, float, float, float]) -> pd.DataFrame:
    """One row per lap: (s1, s2, s3, lap) residuals in seconds."""
    return pd.DataFrame(rows, columns=["s1_residual_s", "s2_residual_s", "s3_residual_s", "lap_residual_s"])


CLEAN = (0.02, -0.03, 0.01, 0.00)


class TestSectorSplitOutliers:
    def test_the_defaults_are_pinned(self) -> None:
        assert ANOMALY_S == 0.30 and CANCEL_FRACTION == 0.5

    def test_a_cancelling_adjacent_pair_is_flagged(self) -> None:
        """Sainz at Austin: S1 +1.098, S2 -1.311, lap -0.401."""
        assert sector_split_outliers(laps(CLEAN, (1.098, -1.311, -0.189, -0.401))).tolist() == [False, True]

    def test_it_works_in_either_direction(self) -> None:
        assert sector_split_outliers(laps((-1.311, 1.098, 0.0, 0.0))).tolist() == [True]

    def test_the_second_adjacent_pair_is_checked_too(self) -> None:
        """Sainz at Baku: S2 +0.785, S3 -1.149."""
        assert sector_split_outliers(laps((0.304, 0.785, -1.149, -0.060))).tolist() == [True]

    def test_two_genuinely_slow_sectors_are_not_flagged(self) -> None:
        """They add up instead of cancelling: a real, slow lap."""
        assert sector_split_outliers(laps((0.5, 0.5, 0.0, 1.0))).tolist() == [False]

    def test_a_pair_that_only_half_cancels_is_not_flagged(self) -> None:
        assert sector_split_outliers(laps((1.0, -0.4, 0.0, 0.6))).tolist() == [False]

    def test_a_small_cancelling_pair_is_left_alone(self) -> None:
        """Registration noise cancels too; below the threshold it is not an anomaly."""
        assert sector_split_outliers(laps((0.10, -0.10, 0.0, 0.0))).tolist() == [False]

    def test_s1_and_s3_are_not_treated_as_adjacent(self) -> None:
        """They meet only across the timing line, where the lap residual catches it."""
        assert sector_split_outliers(laps((0.9, 0.0, -0.9, 0.0))).tolist() == [False]

    def test_a_clean_session_flags_nothing(self) -> None:
        assert not sector_split_outliers(laps(CLEAN, CLEAN, CLEAN)).any()

    def test_nulls_are_not_anomalies(self) -> None:
        assert sector_split_outliers(laps((np.nan, np.nan, np.nan, np.nan))).tolist() == [False]


class TestLapOutliers:
    def test_a_far_out_lap_is_flagged(self) -> None:
        assert lap_outliers(laps(CLEAN, (0.0, 0.0, 0.0, 0.51))).tolist() == [False, True]

    def test_the_threshold_is_inclusive_below(self) -> None:
        assert lap_outliers(laps((0.0, 0.0, 0.0, ANOMALY_S))).tolist() == [False]
        assert lap_outliers(laps((0.0, 0.0, 0.0, -ANOMALY_S - 1e-6))).tolist() == [True]

    def test_nulls_are_not_outliers(self) -> None:
        assert lap_outliers(laps((0.0, 0.0, 0.0, np.nan))).tolist() == [False]

    def test_a_lap_can_be_both(self) -> None:
        frame = laps((1.2, -1.25, 0.0, 0.42))
        assert sector_split_outliers(frame).iloc[0] and lap_outliers(frame).iloc[0]

    def test_a_split_lap_need_not_be_a_lap_outlier(self) -> None:
        """The point of the split flag: the lap is right, the split is not."""
        frame = laps((1.098, -1.311, -0.189, -0.05))
        assert sector_split_outliers(frame).iloc[0] and not lap_outliers(frame).iloc[0]
