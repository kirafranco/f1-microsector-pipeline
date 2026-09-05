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
        assert suzuka.report.closure.p50 <= mod.CLOSURE_P50_MAX_S
        assert suzuka.report.closure.p95 <= mod.CLOSURE_P95_MAX_S

    def test_criterion_3_sector_times(self, suzuka) -> None:
        for name in ("s1", "s2", "s3"):
            assert abs(suzuka.report.sector_median_s[name]) <= mod.SECTOR_MEDIAN_MAX_S, name
            assert suzuka.report.sector_std_s[name] <= mod.SECTOR_STD_MAX_S[name], name

    def test_criterion_4_driven_distance(self, suzuka) -> None:
        assert suzuka.report.distance_ok
        assert suzuka.report.driven_std_m <= mod.DISTANCE_STD_MAX_M

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
