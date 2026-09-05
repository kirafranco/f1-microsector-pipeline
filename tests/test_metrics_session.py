"""F004 session runner on the designed session, and the acceptance table on Suzuka 2024 Q."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import DATA_ROOT, INTERIM_ROOT
from src.metrics import validation
from src.metrics.reference import ReferenceError, ReferenceSpec
from src.metrics.session import LAP_SUMMARY_SCHEMA, compute_metrics
from tests import synthetic_session as syn

SUZUKA = {
    "grid_root": INTERIM_ROOT / "grid" / "2024_Japanese-Grand-Prix_Q_projection",
    "microsector_root": INTERIM_ROOT / "microsectors" / "2024_Japanese-Grand-Prix_Q_projection",
    "snapshot_root": DATA_ROOT / "raw" / "fastf1" / "2026-09-05" / "2024_Japanese-Grand-Prix_Q",
    "aligned_root": INTERIM_ROOT / "aligned" / "2024_Japanese-Grand-Prix_Q_projection",
}
SUZUKA_PRESENT = all(
    (SUZUKA[k] / f).exists()
    for k, f in (
        ("grid_root", "grid.parquet"),
        ("microsector_root", "microsectors.parquet"),
        ("snapshot_root", "laps.parquet"),
        ("aligned_root", "alignment_meta.json"),
    )
)
ARTEFACTS = (
    "delta_t.parquet",
    "microsector_times.parquet",
    "microsector_summary.parquet",
    "corner_metrics.parquet",
    "lap_summary.parquet",
    "metrics_meta.json",
)


@pytest.fixture(scope="module")
def roots(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return syn.write_full_session(tmp_path_factory.mktemp("session"))


@pytest.fixture(scope="module")
def suzuka(tmp_path_factory: pytest.TempPathFactory):
    if not SUZUKA_PRESENT:
        pytest.skip("Suzuka 2024 Q data not present under data/")
    return compute_metrics(**SUZUKA, out_root=tmp_path_factory.mktemp("metrics"))


class TestSyntheticSession:
    def test_writes_every_artefact(self, roots, tmp_path: Path) -> None:
        result = compute_metrics(**roots, out_root=tmp_path / "out")
        for name in ARTEFACTS:
            assert (result.root / name).exists(), name

    def test_report_passes_with_the_designed_numbers(self, roots, tmp_path: Path) -> None:
        report = compute_metrics(**roots, out_root=tmp_path / "out").report
        assert report.ok
        assert report.delta_zero_ok and report.reference_zero_ok
        assert report.closure_max_err_s <= validation.CLOSURE_MAX_S
        # sector2_time was built from the grid itself: the interior check is exact.
        assert abs(report.s2_median_s) < 1e-3 and report.s2_std_s < 1e-3
        # The constant window offset cancels in every difference.
        assert report.endpoint.max < 1e-3
        # S1 carries the designed extra offset, reported not gated.
        assert report.s1_median_s == pytest.approx(-syn.S1_OFFSET_S, abs=1e-3)
        assert report.vmin_coverage == 1.0 and report.brake_coverage == 1.0
        assert report.brake_gap_m.max == 8.0

    def test_lap_summary_contract(self, roots, tmp_path: Path) -> None:
        result = compute_metrics(**roots, out_root=tmp_path / "out")
        summary = result.lap_summary
        assert list(summary.columns) == list(LAP_SUMMARY_SCHEMA)
        for column, dtype in LAP_SUMMARY_SCHEMA.items():
            assert str(summary[column].dtype) == dtype, column
        assert summary["is_reference"].sum() == 1
        assert (summary["compound"] == "SOFT").all()
        np.testing.assert_allclose(summary["lap_time_s"] - summary["grid_time_s"], syn.LAP_TIME_OFFSET_S, atol=1e-3)

    def test_meta_carries_the_acceptance_table(self, roots, tmp_path: Path) -> None:
        result = compute_metrics(**roots, out_root=tmp_path / "out")
        meta = json.loads((result.root / "metrics_meta.json").read_text(encoding="utf-8"))
        assert meta["acceptance"]["checks"]["all"] is True
        assert meta["reference"]["kind"] == "session_fastest"
        assert meta["rows"]["delta_t"] == 4 * syn.N_POINTS
        assert meta["sector_boundaries_m"] == [syn.S1_M, syn.S2_M]

    def test_driver_best_reference(self, roots, tmp_path: Path) -> None:
        result = compute_metrics(**roots, out_root=tmp_path / "out", reference=ReferenceSpec("driver_best"))
        assert result.report.ok
        assert result.lap_summary["is_reference"].sum() == 2
        for driver in ("AAA", "BBB"):
            best = result.lap_summary[(result.lap_summary["driver"] == driver) & result.lap_summary["is_reference"].astype(bool)]
            assert len(best) == 1
            rows = result.delta[(result.delta["driver"] == driver) & (result.delta["lap_number"] == best["lap_number"].iloc[0])]
            assert (rows["delta_t_s"] == 0.0).all()
        assert result.summary["ref_s"].notna().sum() > 0  # session fastest stands in for ref_s

    def test_nominated_reference(self, roots, tmp_path: Path) -> None:
        result = compute_metrics(**roots, out_root=tmp_path / "out", reference=ReferenceSpec("lap", "BBB", 2))
        assert (result.delta["reference"] == "BBB L2").all()
        assert result.report.ok
        with pytest.raises(ReferenceError):
            compute_metrics(**roots, out_root=tmp_path / "out", reference=ReferenceSpec("lap", "ZZZ", 1))

    def test_rerun_is_idempotent(self, roots, tmp_path: Path) -> None:
        out = tmp_path / "out"
        first = compute_metrics(**roots, out_root=out)
        second = compute_metrics(**roots, out_root=out)
        pd.testing.assert_frame_equal(first.delta, second.delta)
        pd.testing.assert_frame_equal(first.times, second.times)
        pd.testing.assert_frame_equal(first.corners, second.corners)
        pd.testing.assert_frame_equal(pd.read_parquet(out / "lap_summary.parquet"), second.lap_summary)

    def test_default_output_root_is_processed(self, roots, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.metrics.session as mod

        monkeypatch.setattr(mod, "PROCESSED_ROOT", tmp_path / "processed")
        result = compute_metrics(**roots)
        assert result.root == tmp_path / "processed" / "synthetic"


@pytest.mark.skipif(not SUZUKA_PRESENT, reason="Suzuka 2024 Q data not present under data/")
class TestSuzukaAcceptance:
    """The spec's acceptance table, on the real session, network-free."""

    def test_reference_is_pole(self, suzuka) -> None:
        assert suzuka.report.reference_label == "VER L11"
        assert suzuka.report.laps == 74

    def test_criterion_1_delta_zero(self, suzuka) -> None:
        assert suzuka.report.delta_zero_ok and suzuka.report.reference_zero_ok

    def test_criterion_2_closure(self, suzuka) -> None:
        assert suzuka.report.closure_max_err_s <= validation.CLOSURE_MAX_S

    def test_criterion_3_interior_accuracy(self, suzuka) -> None:
        assert abs(suzuka.report.s2_median_s) <= validation.S2_MEDIAN_MAX_S
        assert suzuka.report.s2_std_s <= validation.S2_STD_MAX_S

    def test_criterion_4_endpoint_vs_official(self, suzuka) -> None:
        assert suzuka.report.endpoint.p50 <= validation.ENDPOINT_P50_MAX_S
        assert suzuka.report.endpoint.p95 <= validation.ENDPOINT_P95_MAX_S

    def test_criterion_5_coverage(self, suzuka) -> None:
        assert suzuka.report.vmin_coverage == 1.0
        assert suzuka.report.brake_coverage >= validation.BRAKE_COVERAGE_MIN

    def test_criterion_6_sector_times_complete(self, suzuka) -> None:
        assert suzuka.report.unflagged_nan_sector_times == 0
        assert len(suzuka.times) == 74 * 92  # 92 micro-sectors per lap since F015 placed the origin: 57 fixed bins (was 58 / 93)

    def test_criterion_7_all_reference_kinds(self, suzuka, tmp_path: Path) -> None:
        best = compute_metrics(**SUZUKA, out_root=tmp_path / "best", reference=ReferenceSpec("driver_best"))
        assert best.report.reference_zero_ok and best.lap_summary["is_reference"].sum() == 20
        nominated = compute_metrics(**SUZUKA, out_root=tmp_path / "nom", reference=ReferenceSpec("lap", "PER", 11))
        assert (nominated.delta["reference"] == "PER L11").all()

    def test_criterion_8_deterministic(self, suzuka, tmp_path: Path) -> None:
        again = compute_metrics(**SUZUKA, out_root=tmp_path / "again")
        pd.testing.assert_frame_equal(again.delta, suzuka.delta)
        pd.testing.assert_frame_equal(again.lap_summary, suzuka.lap_summary)

    def test_reported_window_offsets_are_negative(self, suzuka) -> None:
        """Not gated: both lap ends sit inside the official lap (F010's problem)."""
        assert suzuka.report.s1_median_s < 0 and suzuka.report.s3_median_s < 0

    def test_everything_passes(self, suzuka) -> None:
        assert suzuka.report.ok
