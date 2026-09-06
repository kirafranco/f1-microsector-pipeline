"""F010 session runner on the designed session, and the acceptance table on Suzuka 2024 Q."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import DATA_ROOT, INTERIM_ROOT, PROCESSED_ROOT
from src.metrics.session import compute_metrics
from src.validate import session as mod
from src.validate.session import GROUND_TRUTH_COLUMNS, validate_session
from tests import synthetic_session as syn

SUZUKA = {
    "snapshot_root": DATA_ROOT / "raw" / "fastf1" / "2026-09-05" / "2024_Japanese-Grand-Prix_Q",
    "aligned_root": INTERIM_ROOT / "aligned" / "2024_Japanese-Grand-Prix_Q_projection",
    "grid_root": INTERIM_ROOT / "grid" / "2024_Japanese-Grand-Prix_Q_projection",
    "processed_root": PROCESSED_ROOT / "2024_Japanese-Grand-Prix_Q_projection",
}
SUZUKA_PRESENT = all(
    (SUZUKA[k] / f).exists()
    for k, f in (
        ("snapshot_root", "laps.parquet"),
        ("aligned_root", "telemetry_aligned.parquet"),
        ("grid_root", "grid.parquet"),
        ("processed_root", "corner_metrics.parquet"),
    )
)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("validate")
    roots = syn.write_full_session(root)
    processed = root / "processed" / "synthetic"
    compute_metrics(
        roots["grid_root"], roots["microsector_root"], roots["snapshot_root"], roots["aligned_root"], out_root=processed
    )
    return {
        "snapshot_root": roots["snapshot_root"],
        "aligned_root": roots["aligned_root"],
        "grid_root": roots["grid_root"],
        "processed_root": processed,
    }


@pytest.fixture(scope="module")
def suzuka(tmp_path_factory: pytest.TempPathFactory):
    if not SUZUKA_PRESENT:
        pytest.skip("Suzuka 2024 Q data not present under data/")
    return validate_session(**SUZUKA, out_root=tmp_path_factory.mktemp("ground_truth"))


class TestSyntheticSession:
    def test_writes_the_artefacts(self, synthetic, tmp_path: Path) -> None:
        result = validate_session(**synthetic, out_root=tmp_path / "out")
        for name in ("ground_truth.parquet", "v_min_stability.parquet", "ground_truth_report.json"):
            assert (result.root / name).exists(), name
        assert list(result.ground_truth.columns) == list(GROUND_TRUTH_COLUMNS)

    def test_recovers_the_designed_timing_line(self, synthetic, tmp_path: Path) -> None:
        """The synthetic official lap time is grid time plus a constant offset,
        so the line sits symmetrically outside the axis."""
        report = validate_session(**synthetic, out_root=tmp_path / "out").report
        assert report.line_start_m < 0.0
        assert report.line_end_m > syn.LAP_LENGTH_M - syn.GRID_M
        span = report.line_end_m - report.line_start_m
        assert span == pytest.approx(syn.LAP_LENGTH_M + syn.LAP_TIME_OFFSET_S * 300.0 / 3.6, rel=0.05)

    def test_lap_and_sector_residuals_vanish(self, synthetic, tmp_path: Path) -> None:
        report = validate_session(**synthetic, out_root=tmp_path / "out").report
        assert abs(report.lap_residual_median_s) < 1e-3
        assert report.lap_residual.max < 1e-3
        for name in ("s1", "s2", "s3"):
            assert abs(report.sector_median_s[name]) < 1e-3, name

    def test_closure_is_zero_and_every_check_passes(self, synthetic, tmp_path: Path) -> None:
        report = validate_session(**synthetic, out_root=tmp_path / "out", min_laps=2).report
        assert report.closure.max < 1e-3
        assert report.ok

    def test_measuring_from_the_line_removes_the_offset_f004_reports(self, synthetic, tmp_path: Path) -> None:
        """The designed S1 offset is an artefact of where the clock starts.

        F004 zeroes its curves at grid 0 and therefore sees -0.1 s on S1; F010
        starts at the timing line itself and sees nothing. That difference is
        the whole point of locating the line.
        """
        report = validate_session(**synthetic, out_root=tmp_path / "out").report
        metrics = json.loads((synthetic["processed_root"] / "metrics_meta.json").read_text(encoding="utf-8"))
        assert metrics["acceptance"]["s1_median_s"] == pytest.approx(-syn.S1_OFFSET_S, abs=1e-3)
        assert report.sector_median_s["s1"] == pytest.approx(0.0, abs=1e-3)

    def test_a_late_window_is_flagged_and_excluded(self, synthetic, tmp_path: Path) -> None:
        """A lap missing its first samples must not pollute the gated statistics."""
        aligned_root = tmp_path / "aligned"
        aligned_root.mkdir(parents=True, exist_ok=True)
        for name in ("alignment_meta.json",):
            (aligned_root / name).write_text((synthetic["aligned_root"] / name).read_text(encoding="utf-8"), encoding="utf-8")
        telemetry = pd.read_parquet(synthetic["aligned_root"] / "telemetry_aligned.parquet")
        victim = (telemetry["driver"] == "BBB") & (telemetry["lap_number"] == 1)
        keep = ~(victim & (telemetry.groupby(["driver", "lap_number"]).cumcount() < 40))
        telemetry[keep].to_parquet(aligned_root / "telemetry_aligned.parquet", index=False)

        result = validate_session(**{**synthetic, "aligned_root": aligned_root}, out_root=tmp_path / "out")
        assert "BBB L1" in result.report.flagged
        assert result.report.laps_gated == result.report.laps - 1

    def test_stability_table(self, synthetic, tmp_path: Path) -> None:
        result = validate_session(**synthetic, out_root=tmp_path / "out", min_laps=2)
        assert len(result.stability) > 0
        assert (result.stability["compound"] == "SOFT").all()
        assert result.report.v_min_groups == len(result.stability)

    def test_rerun_is_idempotent(self, synthetic, tmp_path: Path) -> None:
        out = tmp_path / "out"
        first = validate_session(**synthetic, out_root=out)
        second = validate_session(**synthetic, out_root=out)
        pd.testing.assert_frame_equal(first.ground_truth, second.ground_truth)
        pd.testing.assert_frame_equal(pd.read_parquet(out / "v_min_stability.parquet"), second.stability)

    def test_default_output_root(self, synthetic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "PROCESSED_ROOT", tmp_path / "processed")
        result = validate_session(**synthetic)
        assert result.root == tmp_path / "processed" / "synthetic"


@pytest.mark.skipif(not SUZUKA_PRESENT, reason="Suzuka 2024 Q data not present under data/")
class TestSuzukaAcceptance:
    """The spec's acceptance table, on the real session, network-free."""

    def test_criterion_1_lap_reconstruction(self, suzuka) -> None:
        assert suzuka.report.lap_residual_std_s <= mod.LAP_STD_MAX_S
        assert suzuka.report.lap_residual.p95 <= mod.LAP_P95_MAX_S

    def test_criterion_a_bias_removed(self, suzuka) -> None:
        """Part A: the +0.119 s systematic bias is gone."""
        assert abs(suzuka.report.lap_residual_median_s) <= 0.03

    def test_criterion_2_delta_closure(self, suzuka) -> None:
        assert suzuka.report.closure.p95 <= mod.CLOSURE_P95_MAX_S
        assert abs(suzuka.report.reference_offset_s) <= mod.LAP_P95_MAX_S

    def test_criterion_3_sector_times(self, suzuka) -> None:
        for name in ("s1", "s2", "s3"):
            assert abs(suzuka.report.sector_median_s[name]) <= mod.SECTOR_MEDIAN_MAX_S, name
            assert suzuka.report.sector_std_s[name] <= mod.SECTOR_STD_MAX_S[name], name

    def test_criterion_4_driven_distance(self, suzuka) -> None:
        assert suzuka.report.distance_ok
        assert suzuka.report.driven_std_pct <= mod.DISTANCE_STD_MAX_PCT

    def test_criterion_5_v_min_stability(self, suzuka) -> None:
        assert suzuka.report.v_min_groups >= 100
        assert suzuka.report.v_min_std_median_kmh <= mod.V_MIN_STD_MEDIAN_MAX_KMH
        assert suzuka.report.v_min_std_p95_kmh <= mod.V_MIN_STD_P95_MAX_KMH

    def test_criterion_6_timing_line_spread(self, suzuka) -> None:
        assert suzuka.report.line_start_std_m <= mod.LINE_POSITION_STD_MAX_M
        assert suzuka.report.line_end_std_m <= mod.LINE_POSITION_STD_MAX_M

    def test_criterion_8_deterministic(self, suzuka, tmp_path: Path) -> None:
        again = validate_session(**SUZUKA, out_root=tmp_path / "again")
        pd.testing.assert_frame_equal(again.ground_truth, suzuka.ground_truth)

    def test_reported_not_gated(self, suzuka) -> None:
        """The window offsets and the flagged lap are reported, not silently dropped."""
        assert suzuka.report.window_open_median_s > 0 and suzuka.report.window_close_median_s < 0
        assert suzuka.report.laps_gated <= suzuka.report.laps
        meta = json.loads((suzuka.root / "ground_truth_report.json").read_text(encoding="utf-8"))
        assert meta["acceptance"]["flagged"] == suzuka.report.flagged
        assert "registration" in meta["limitation"]

    def test_everything_passes(self, suzuka) -> None:
        assert suzuka.report.ok


class TestDistanceExcursions:
    """F019: a lap that drove somewhere else is reported and excluded, not gated on."""

    @staticmethod
    def report(synthetic, out: Path, *, alter=None):
        """Validate the designed session, optionally after moving one lap's distance."""
        if alter is None:
            return validate_session(**synthetic, out_root=out).report
        aligned = out / "aligned"
        aligned.mkdir(parents=True, exist_ok=True)
        for name in ("telemetry_aligned.parquet", "alignment_meta.json"):
            (aligned / name).write_bytes((synthetic["aligned_root"] / name).read_bytes())
        meta = json.loads((aligned / "alignment_meta.json").read_text(encoding="utf-8"))
        meta["official_lap_length_m"] = alter
        (aligned / "alignment_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return validate_session(**{**synthetic, "aligned_root": aligned}, out_root=out / "o").report

    def test_a_clean_session_has_no_excursions(self, synthetic, tmp_path: Path) -> None:
        report = self.report(synthetic, tmp_path / "clean")
        assert report.excursions == []
        assert report.distance_ok

    def test_the_spread_rule_is_a_fraction_of_lap_length(self, synthetic, tmp_path: Path) -> None:
        """The same metres of spread pass on a long circuit and fail on a short one."""
        report = self.report(synthetic, tmp_path / "clean")
        assert report.driven_std_pct == pytest.approx(
            100.0 * report.driven_std_m / syn.OFFICIAL_LENGTH_M, rel=1e-6
        )

    def test_the_ground_truth_carries_the_flag(self, synthetic, tmp_path: Path) -> None:
        result = validate_session(**synthetic, out_root=tmp_path / "flagged")
        assert "distance_excursion" in result.ground_truth.columns
        assert not result.ground_truth["distance_excursion"].any()

    def test_the_report_counts_push_laps(self, synthetic, tmp_path: Path) -> None:
        report = self.report(synthetic, tmp_path / "push")
        assert 0 < report.push_laps <= report.laps

    def test_out_of_band_laps_are_named_with_their_percentage(self, synthetic, tmp_path: Path) -> None:
        """Shortening the official length pushes every lap above the +0.2 % edge.

        The designed session's four laps drive within a centimetre of each
        other, so it can only put all of them in the band or none. That the
        excluded laps are actually left out of the statistics is checked on the
        real season, where Silverstone R has exactly one excursion in 850 laps.
        """
        report = self.report(synthetic, tmp_path / "short", alter=syn.OFFICIAL_LENGTH_M * 0.9)
        assert len(report.excursions) == report.laps
        assert all("%" in entry and " L" in entry for entry in report.excursions)


class TestDistanceExcursionRule:
    """The band rule itself, where a partial split can be constructed."""

    def series(self, *values: float) -> pd.Series:
        return pd.Series(list(values), dtype="float64")

    def test_only_the_laps_outside_the_band_are_marked(self) -> None:
        low, high = mod.OFFICIAL_LENGTH_BAND_PCT
        flags = mod.distance_excursions(self.series(0.0, low - 0.01, high + 0.01, high - 0.01, low + 0.01))
        assert flags.tolist() == [False, True, True, False, False]

    def test_the_edges_are_inside(self) -> None:
        low, high = mod.OFFICIAL_LENGTH_BAND_PCT
        assert mod.distance_excursions(self.series(low, high)).tolist() == [False, False]

    def test_a_lap_with_no_distance_is_not_an_excursion(self) -> None:
        assert mod.distance_excursions(self.series(np.nan, 0.0)).tolist() == [False, False]

    def test_the_measured_season_extremes_are_caught(self) -> None:
        assert mod.distance_excursions(self.series(-5.05, 4.64, -0.44)).tolist() == [True, True, False]


class TestF022Ceilings:
    """The two checks F022 re-formed, on the designed session."""

    def test_the_sector_ceilings_are_the_lap_ceiling(self) -> None:
        """Derived, not typed: a sector is bounded by two loops exactly as a lap is."""
        assert set(mod.SECTOR_STD_MAX_S) == {"s1", "s2", "s3"}
        assert set(mod.SECTOR_STD_MAX_S.values()) == {mod.LAP_STD_MAX_S}

    def test_the_p50_rule_is_gone(self) -> None:
        assert not hasattr(mod, "CLOSURE_P50_MAX_S")

    def test_a_clean_session_has_no_anomalies(self, synthetic, tmp_path: Path) -> None:
        report = validate_session(**synthetic, out_root=tmp_path / "clean").report
        assert report.anomalies == []
        assert report.anomaly_fraction == 0.0
        assert report.anomaly_fraction_ok

    def test_the_ground_truth_carries_both_flags(self, synthetic, tmp_path: Path) -> None:
        truth = validate_session(**synthetic, out_root=tmp_path / "flags").ground_truth
        for column in ("sector_split_outlier", "lap_outlier"):
            assert column in truth.columns and not truth[column].any()

    def test_the_reference_offset_is_reported(self, synthetic, tmp_path: Path) -> None:
        report = validate_session(**synthetic, out_root=tmp_path / "ref").report
        assert not report.reference_flagged
        assert np.isfinite(report.reference_offset_s)

    def test_registration_metres_are_reported_for_every_sector(self, synthetic, tmp_path: Path) -> None:
        """The designed session has essentially no sector spread, so ~0 m is right."""
        report = validate_session(**synthetic, out_root=tmp_path / "reg").report
        assert set(report.registration_m) == {"s1", "s2", "s3"}
        assert all(np.isfinite(v) and v >= 0 for v in report.registration_m.values())
        assert max(report.registration_m.values()) < 0.01


class TestSectorRegistrationMetres:
    """The conversion itself, where the answer can be worked out by hand."""

    def grid(self, kmh: float, points: int = 300) -> pd.DataFrame:
        return pd.DataFrame({"grid_index": np.arange(points), "speed": np.full(points, kmh)})

    def test_equal_loop_speeds_give_std_times_v_over_root_two(self) -> None:
        v_kmh, std = 360.0, 0.10                      # 100 m/s
        out = mod.sector_registration_m(self.grid(v_kmh), {"s1": std, "s2": std, "s3": std}, 1000.0, 2000.0, 10.0)
        expected = std * (v_kmh / 3.6) / np.sqrt(2)   # 7.07 m
        assert all(out[name] == pytest.approx(expected, rel=1e-6) for name in ("s1", "s2", "s3"))

    def test_a_slower_loop_needs_fewer_metres_for_the_same_seconds(self) -> None:
        """Why Baku and Monaco look worst in seconds: 4 m at 150 km/h is 0.1 s."""
        fast = mod.sector_registration_m(self.grid(360.0), {"s1": 0.10, "s2": 0.10, "s3": 0.10}, 1000.0, 2000.0, 10.0)
        slow = mod.sector_registration_m(self.grid(150.0), {"s1": 0.10, "s2": 0.10, "s3": 0.10}, 1000.0, 2000.0, 10.0)
        assert slow["s2"] < fast["s2"]
        assert slow["s2"] == pytest.approx(0.10 * (150.0 / 3.6) / np.sqrt(2), rel=1e-6)

    def test_zero_spread_is_zero_metres(self) -> None:
        out = mod.sector_registration_m(self.grid(300.0), {"s1": 0.0, "s2": 0.0, "s3": 0.0}, 1000.0, 2000.0, 10.0)
        assert set(out.values()) == {0.0}

    def test_a_stopped_loop_is_not_a_division_by_zero(self) -> None:
        out = mod.sector_registration_m(self.grid(0.0), {"s1": 0.1, "s2": 0.1, "s3": 0.1}, 1000.0, 2000.0, 10.0)
        assert all(np.isnan(value) for value in out.values())


class TestClosureIsAboutTheReferenceLap:
    """`closure_ok` on constructed reports, where the reference offset is known."""

    def report(self, **overrides):
        from dataclasses import replace
        base = dict(closure_p95=0.10, reference_offset_s=0.0, reference_flagged=False)
        base.update(overrides)

        class Stub:
            closure = type("S", (), {"p50": 0.0, "p95": base["closure_p95"]})()
            reference_offset_s = base["reference_offset_s"]
            reference_flagged = base["reference_flagged"]
            closure_ok = mod.ValidationReport.closure_ok
        return Stub()

    def test_a_centred_reference_passes(self) -> None:
        assert type(self.report()).closure_ok.fget(self.report())

    def test_a_reference_lap_that_is_itself_an_outlier_fails(self) -> None:
        stub = self.report(reference_offset_s=mod.LAP_P95_MAX_S + 0.05)
        assert not type(stub).closure_ok.fget(stub)

    def test_a_wide_spread_still_fails(self) -> None:
        stub = self.report(closure_p95=mod.CLOSURE_P95_MAX_S + 0.01)
        assert not type(stub).closure_ok.fget(stub)

    def test_a_flagged_reference_is_not_held_against_the_session(self) -> None:
        stub = self.report(reference_offset_s=float("nan"), reference_flagged=True)
        assert type(stub).closure_ok.fget(stub)
