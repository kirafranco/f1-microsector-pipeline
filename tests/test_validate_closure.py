"""F010: lap and sector reconstruction from the grid, and delta-t closure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics.delta import time_curves
from src.metrics.reference import lap_index
from src.validate.closure import closure_residuals, reconstruct_laps
from src.validate.stability import v_min_stability
from tests.test_metrics_delta import constant_speed_grid

LINE_START_M = -30.0
LINE_END_M = 1000.0 + 20.0
S1_M, S2_M = 300.0, 700.0


def curves_and_speeds(specs):
    grid = constant_speed_grid(specs)
    curves = time_curves(grid)
    speeds = grid.pivot_table(index=["driver", "lap_number"], columns="grid_index", values="speed", aggfunc="first")
    speeds.index = curves.index
    return curves, speeds


def official(specs, extra: dict | None = None) -> pd.DataFrame:
    """Official timing consistent with constant-speed laps over the line-to-line distance."""
    rows = []
    for driver, lap, speed, n, _ in specs:
        v = speed / 3.6
        rows.append(
            {
                "driver": driver, "lap_number": lap,
                "lap_time": (LINE_END_M - LINE_START_M) / v,
                "sector1_time": (S1_M - LINE_START_M) / v,
                "sector2_time": (S2_M - S1_M) / v,
                "sector3_time": (LINE_END_M - S2_M) / v,
                "is_accurate": True,
            }
        )
    frame = pd.DataFrame(rows)
    if extra:
        frame = frame.assign(**extra)
    return frame


SPECS = [("AAA", 1, 100.0, 110, 0.0), ("BBB", 1, 90.0, 110, 0.0)]


class TestReconstruction:
    def test_constant_speed_laps_reproduce_official_times_exactly(self) -> None:
        curves, speeds = curves_and_speeds(SPECS)
        out = reconstruct_laps(curves, speeds, official(SPECS), LINE_START_M, LINE_END_M, S1_M, S2_M)
        assert len(out) == 2
        # float4 telemetry: exactness is at float32 precision, not float64.
        np.testing.assert_allclose(out["lap_residual_s"], 0.0, atol=1e-4)
        for name in ("s1", "s2", "s3"):
            np.testing.assert_allclose(out[f"{name}_residual_s"], 0.0, atol=1e-4)

    def test_sector_times_sum_to_the_lap(self) -> None:
        curves, speeds = curves_and_speeds(SPECS)
        out = reconstruct_laps(curves, speeds, official(SPECS), LINE_START_M, LINE_END_M, S1_M, S2_M)
        total = out["s1_grid_s"] + out["s2_grid_s"] + out["s3_grid_s"]
        np.testing.assert_allclose(total, out["lap_grid_s"], atol=1e-9)

    def test_extrapolation_extents_are_reported(self) -> None:
        curves, speeds = curves_and_speeds(SPECS)
        out = reconstruct_laps(curves, speeds, official(SPECS), LINE_START_M, LINE_END_M, S1_M, S2_M)
        # Grid runs 0..1090 m; the line sits at -30 and 1020, so only the start needs extending.
        assert (out["start_extrap_m"] == 30.0).all()
        assert (out["end_extrap_m"] == 0.0).all()

    def test_a_wrong_lap_time_shows_up_as_a_residual(self) -> None:
        curves, speeds = curves_and_speeds(SPECS)
        laps = official(SPECS)
        laps.loc[0, "lap_time"] += 0.5
        out = reconstruct_laps(curves, speeds, laps, LINE_START_M, LINE_END_M, S1_M, S2_M)
        assert out.loc[0, "lap_residual_s"] == pytest.approx(-0.5, abs=1e-4)
        assert out.loc[1, "lap_residual_s"] == pytest.approx(0.0, abs=1e-4)

    def test_shorter_lap_is_still_reconstructed(self) -> None:
        specs = [("AAA", 1, 100.0, 110, 0.0), ("BBB", 1, 100.0, 100, 0.0)]
        curves, speeds = curves_and_speeds(specs)
        out = reconstruct_laps(curves, speeds, official(specs), LINE_START_M, LINE_END_M, S1_M, S2_M)
        assert out["n_points"].tolist() == [110, 100]
        assert out["end_extrap_m"].iloc[1] > out["end_extrap_m"].iloc[0]
        # float4 telemetry: exactness is at float32 precision, not float64.
        np.testing.assert_allclose(out["lap_residual_s"], 0.0, atol=1e-4)


class TestClosure:
    def test_reference_lap_is_nan_and_others_are_measured(self) -> None:
        curves, speeds = curves_and_speeds(SPECS)
        out = reconstruct_laps(curves, speeds, official(SPECS), LINE_START_M, LINE_END_M, S1_M, S2_M)
        reference = pd.Series([("AAA", 1)] * 2, index=lap_index([("AAA", 1), ("BBB", 1)]))
        closed = closure_residuals(out, reference)
        assert closed["is_reference"].tolist() == [True, False]
        assert np.isnan(closed.loc[0, "closure_residual_s"])
        assert closed.loc[1, "closure_residual_s"] == pytest.approx(0.0, abs=1e-4)

    def test_a_grid_error_appears_in_the_closure(self) -> None:
        curves, speeds = curves_and_speeds(SPECS)
        out = reconstruct_laps(curves, speeds, official(SPECS), LINE_START_M, LINE_END_M, S1_M, S2_M)
        out.loc[1, "lap_grid_s"] += 0.3
        reference = pd.Series([("AAA", 1)] * 2, index=lap_index([("AAA", 1), ("BBB", 1)]))
        closed = closure_residuals(out, reference)
        assert closed.loc[1, "closure_residual_s"] == pytest.approx(0.3, abs=1e-4)


class TestStability:
    def test_groups_below_the_minimum_are_dropped(self) -> None:
        metrics = pd.DataFrame(
            {
                "driver": ["AAA"] * 4 + ["BBB"] * 2,
                "lap_number": [1, 2, 3, 4, 1, 2],
                "event_id": [0] * 6,
                "corners": ["T1"] * 6,
                "v_min_kmh": [100.0, 102.0, 101.0, 99.0, 120.0, 121.0],
            }
        )
        laps = pd.DataFrame(
            {"driver": ["AAA"] * 4 + ["BBB"] * 2, "lap_number": [1, 2, 3, 4, 1, 2], "compound": ["SOFT"] * 6}
        )
        out = v_min_stability(metrics, laps, min_laps=3)
        assert out["driver"].tolist() == ["AAA"]
        assert out["n_laps"].iloc[0] == 4
        assert out["v_min_std_kmh"].iloc[0] == pytest.approx(np.std([100.0, 102.0, 101.0, 99.0], ddof=1), abs=1e-4)
        assert out["corners"].iloc[0] == "T1"

    def test_compounds_split_the_groups(self) -> None:
        metrics = pd.DataFrame(
            {
                "driver": ["AAA"] * 6, "lap_number": [1, 2, 3, 4, 5, 6], "event_id": [0] * 6,
                "corners": ["T1"] * 6, "v_min_kmh": [100.0, 101.0, 99.0, 130.0, 131.0, 129.0],
            }
        )
        laps = pd.DataFrame(
            {"driver": ["AAA"] * 6, "lap_number": [1, 2, 3, 4, 5, 6], "compound": ["SOFT"] * 3 + ["HARD"] * 3}
        )
        out = v_min_stability(metrics, laps, min_laps=3)
        assert sorted(out["compound"].tolist()) == ["HARD", "SOFT"]
        assert out["v_min_mean_kmh"].round(0).tolist() in ([130.0, 100.0], [100.0, 130.0])
