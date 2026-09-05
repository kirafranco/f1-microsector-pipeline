"""F011 session runner, and the acceptance table on Suzuka 2024 Q."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import DATA_ROOT, INTERIM_ROOT, PROCESSED_ROOT
from src.metrics.session import compute_metrics
from src.quality import session as mod
from src.quality.contracts import CONTRACTS, SESSION_TABLES
from src.quality.engine import QualityGateError, require
from src.quality.session import ARTEFACTS, check_session, load_artefacts, observed_ranges
from src.validate.session import validate_session
from tests import synthetic_session as syn

SUZUKA = {
    "snapshot_root": DATA_ROOT / "raw" / "fastf1" / "2026-09-05" / "2024_Japanese-Grand-Prix_Q",
    "aligned_root": INTERIM_ROOT / "aligned" / "2024_Japanese-Grand-Prix_Q_projection",
    "grid_root": INTERIM_ROOT / "grid" / "2024_Japanese-Grand-Prix_Q_projection",
    "microsector_root": INTERIM_ROOT / "microsectors" / "2024_Japanese-Grand-Prix_Q_projection",
    "processed_root": PROCESSED_ROOT / "2024_Japanese-Grand-Prix_Q_projection",
}
SUZUKA_PRESENT = all(
    (SUZUKA[key] / name).exists()
    for key, name in (
        ("snapshot_root", "laps.parquet"),
        ("grid_root", "grid.parquet"),
        ("microsector_root", "microsectors.parquet"),
        ("processed_root", "corner_metrics.parquet"),
    )
)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("quality_session")
    roots = syn.write_full_session(root)
    processed = root / "processed" / "synthetic"
    compute_metrics(roots["grid_root"], roots["microsector_root"], roots["snapshot_root"], roots["aligned_root"],
                    out_root=processed)
    validate_session(roots["snapshot_root"], roots["aligned_root"], roots["grid_root"], processed,
                     out_root=processed, min_laps=2)
    return {
        "snapshot_root": roots["snapshot_root"],
        "aligned_root": roots["aligned_root"],
        "grid_root": roots["grid_root"],
        "microsector_root": roots["microsector_root"],
        "processed_root": processed,
    }


@pytest.fixture(scope="module")
def suzuka(tmp_path_factory: pytest.TempPathFactory):
    if not SUZUKA_PRESENT:
        pytest.skip("Suzuka 2024 Q artefacts not present under data/")
    return check_session(**SUZUKA, out_root=tmp_path_factory.mktemp("quality"))


class TestSyntheticSession:
    def test_writes_both_artefacts(self, synthetic, tmp_path: Path) -> None:
        result = check_session(**synthetic, out_root=tmp_path / "out")
        assert (result.root / "quality_findings.parquet").exists()
        assert (result.root / "quality_report.json").exists()

    def test_the_designed_session_is_clean(self, synthetic, tmp_path: Path) -> None:
        result = check_session(**synthetic, out_root=tmp_path / "out")
        assert result.ok, [str(f) for f in result.report.errors]
        require(result.report)

    def test_every_artefact_is_loaded(self, synthetic) -> None:
        frames = load_artefacts(**synthetic)
        assert set(frames) == set(ARTEFACTS) == set(SESSION_TABLES)

    def test_report_json_carries_findings_and_ranges(self, synthetic, tmp_path: Path) -> None:
        result = check_session(**synthetic, out_root=tmp_path / "out")
        payload = json.loads((result.root / "quality_report.json").read_text(encoding="utf-8"))
        assert payload["ok"] is True
        assert len(payload["tables"]) == len(SESSION_TABLES)
        assert payload["observed_ranges"]["grid"]["speed"][1] > 0
        assert "envelopes" in payload["limitation"]
        assert payload["contract_version"] == result.report.contract_version

    def test_a_missing_optional_artefact_is_only_a_warning(self, synthetic, tmp_path: Path) -> None:
        stripped = dict(synthetic)
        processed = tmp_path / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        for name in ("delta_t", "microsector_times", "microsector_summary", "corner_metrics", "lap_summary"):
            pd.read_parquet(synthetic["processed_root"] / f"{name}.parquet").to_parquet(processed / f"{name}.parquet")
        stripped["processed_root"] = processed  # ground_truth and v_min_stability left out: both optional
        result = check_session(**stripped, out_root=tmp_path / "out")
        missing = {f.table for f in result.report.warnings if f.rule == "MissingTable"}
        assert {"ground_truth", "v_min_stability"} <= missing
        assert result.ok

    def test_a_corrupted_artefact_blocks_the_gate(self, synthetic, tmp_path: Path) -> None:
        damaged = tmp_path / "grid_damaged"
        damaged.mkdir(parents=True, exist_ok=True)
        grid = pd.read_parquet(synthetic["grid_root"] / "grid.parquet")
        grid.loc[0, "speed"] = None
        grid.to_parquet(damaged / "grid.parquet", index=False)
        result = check_session(**{**synthetic, "grid_root": damaged}, out_root=tmp_path / "out")
        assert not result.ok
        with pytest.raises(QualityGateError, match="grid"):
            require(result.report)

    def test_observed_ranges_report_real_extremes(self, synthetic) -> None:
        frames = load_artefacts(**synthetic)
        ranges = observed_ranges(frames)
        low, high = ranges["grid"]["speed"]
        assert low < high and high <= 400.0

    def test_rerun_is_idempotent(self, synthetic, tmp_path: Path) -> None:
        out = tmp_path / "out"
        first = check_session(**synthetic, out_root=out)
        second = check_session(**synthetic, out_root=out)
        pd.testing.assert_frame_equal(first.findings, second.findings)
        assert first.report.contract_version == second.report.contract_version

    def test_default_output_root(self, synthetic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "PROCESSED_ROOT", tmp_path / "processed")
        result = check_session(**synthetic)
        assert result.root == tmp_path / "processed" / "synthetic"

    def test_no_artefacts_at_all_is_an_error(self, tmp_path: Path) -> None:
        empty = tmp_path / "nothing"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            check_session(empty, empty, empty, empty, empty, out_root=tmp_path / "out")


@pytest.mark.skipif(not SUZUKA_PRESENT, reason="Suzuka 2024 Q artefacts not present under data/")
class TestSuzukaAcceptance:
    """The spec's acceptance table, on the real session, network-free."""

    def test_criterion_1_every_artefact_passes_with_no_error(self, suzuka) -> None:
        assert len(suzuka.report.tables) == 17
        assert suzuka.ok, [str(f) for f in suzuka.report.errors]

    def test_criterion_2_only_the_expected_warning_stands(self, suzuka) -> None:
        warnings = suzuka.report.warnings
        assert len(warnings) == 1
        finding = warnings[0]
        assert finding.table == "corner_metrics"
        assert finding.columns == ("brake_on_m", "brake_dev_m")
        assert finding.count == 7, "F004 measured 7 braked lap-events with no brake sample in the window"
        assert len(finding.samples) == 7

    def test_criterion_5_findings_are_written_with_their_keys(self, suzuka) -> None:
        findings = pd.read_parquet(suzuka.root / "quality_findings.parquet")
        assert len(findings) == 1
        assert json.loads(findings["samples"].iloc[0])[0][0] == "NOR"

    def test_criterion_6_a_whole_session_is_checked_quickly(self, suzuka) -> None:
        assert suzuka.elapsed_s <= 5.0

    def test_criterion_7_deterministic(self, suzuka, tmp_path: Path) -> None:
        again = check_session(**SUZUKA, out_root=tmp_path / "again")
        pd.testing.assert_frame_equal(again.findings, suzuka.findings)
        assert again.report.contract_version == suzuka.report.contract_version

    def test_the_gate_lets_this_session_through(self, suzuka) -> None:
        require(suzuka.report)

    def test_reported_ranges_sit_inside_the_contract_envelopes(self, suzuka) -> None:
        """Not gated: the numbers a future session's drift would move first."""
        assert suzuka.ranges["grid"]["speed"][1] <= 400.0
        assert suzuka.ranges["grid"]["source_gap_m"][1] < 500.0
        assert suzuka.ranges["corner_metrics"]["brake_gap_m"][1] < 500.0
