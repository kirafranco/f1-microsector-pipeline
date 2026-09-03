"""F009 session runner on the designed session, and the acceptance table on Suzuka 2024 Q."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import DATA_ROOT, INTERIM_ROOT
from src.segment import validation
from src.segment.events import EVENT_SCHEMA
from src.segment.phases import GRAIN_CORNER_PHASE, GRAIN_FIXED_100M, MICROSECTOR_SCHEMA
from src.segment.session import segment_session
from tests import synthetic_session as syn

SUZUKA_GRID = INTERIM_ROOT / "grid" / "2024_Japan_Q_projection"
SUZUKA_ALIGNED = INTERIM_ROOT / "aligned" / "2024_Japan_Q_projection"
SUZUKA_SNAPSHOT = DATA_ROOT / "raw" / "fastf1" / "2026-09-01" / "2024_Japan_Q"
SUZUKA_PRESENT = all(
    p.exists()
    for p in (SUZUKA_GRID / "grid.parquet", SUZUKA_ALIGNED / "alignment_meta.json", SUZUKA_SNAPSHOT / "circuit_corners.parquet")
)


@pytest.fixture()
def roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    return syn.write_session(tmp_path)


@pytest.fixture(scope="module")
def suzuka(tmp_path_factory: pytest.TempPathFactory):
    if not SUZUKA_PRESENT:
        pytest.skip("Suzuka 2024 Q grid/aligned/raw data not present under data/")
    return segment_session(SUZUKA_GRID, SUZUKA_SNAPSHOT, SUZUKA_ALIGNED, out_root=tmp_path_factory.mktemp("microsectors"))


class TestSyntheticSession:
    def test_writes_the_four_artefacts(self, roots, tmp_path: Path) -> None:
        result = segment_session(*roots, out_root=tmp_path / "out")
        for name in ("microsectors.parquet", "events.parquet", "corners_aligned.parquet", "segmentation_meta.json"):
            assert (result.root / name).exists(), name

    def test_tables_match_the_design(self, roots, tmp_path: Path) -> None:
        result = segment_session(*roots, out_root=tmp_path / "out")
        assert result.lap_length_m == syn.LAP_LENGTH_M
        assert result.events["corners"].tolist() == ["T1", "T2"]
        phases = result.microsectors[result.microsectors["grain"] == GRAIN_CORNER_PHASE]
        assert [(r.phase, float(r.start_m), float(r.end_m)) for r in phases.itertuples()] == [
            (p, s, e) for p, s, e, _ in syn.EXPECTED_PHASES
        ]
        assert (result.microsectors["grain"] == GRAIN_FIXED_100M).sum() == 30
        assert result.corners["event_id"].tolist()[:2] == [0, 1] and pd.isna(result.corners["event_id"].iloc[2])

    def test_report_passes_and_carries_the_late_braker(self, roots, tmp_path: Path) -> None:
        result = segment_session(*roots, out_root=tmp_path / "out")
        report = result.report
        assert report.ok
        assert report.partition_ok and report.fixed_bins_ok
        # BBB brakes one bin later than the session boundary on the braked event.
        assert report.brake_dev.max == pytest.approx(10.0)
        assert report.apex_dev.max <= 10.0
        assert report.jackknife_count_matches == report.jackknife_trials
        assert report.corners_without_event == 1

    def test_meta_is_json_with_the_acceptance_table(self, roots, tmp_path: Path) -> None:
        result = segment_session(*roots, out_root=tmp_path / "out")
        meta = json.loads((result.root / "segmentation_meta.json").read_text(encoding="utf-8"))
        assert meta["acceptance"]["checks"]["all"] is True
        assert meta["events"] == 2
        assert meta["microsectors"] == {GRAIN_CORNER_PHASE: 10, GRAIN_FIXED_100M: 30}
        assert meta["params"]["min_prominence_kmh"] == 8.0

    def test_schemas_on_disk(self, roots, tmp_path: Path) -> None:
        result = segment_session(*roots, out_root=tmp_path / "out")
        sectors = pd.read_parquet(result.root / "microsectors.parquet")
        events = pd.read_parquet(result.root / "events.parquet")
        for column, dtype in MICROSECTOR_SCHEMA.items():
            assert str(sectors[column].dtype) == dtype, column
        for column, dtype in EVENT_SCHEMA.items():
            assert str(events[column].dtype) == dtype, column
        assert not sectors.duplicated(["grain", "microsector_id"]).any()

    def test_rerun_is_idempotent(self, roots, tmp_path: Path) -> None:
        out = tmp_path / "out"
        first = segment_session(*roots, out_root=out)
        second = segment_session(*roots, out_root=out)
        pd.testing.assert_frame_equal(first.microsectors, second.microsectors)
        pd.testing.assert_frame_equal(first.events, second.events)
        pd.testing.assert_frame_equal(pd.read_parquet(out / "events.parquet"), second.events)

    def test_default_output_root(self, roots, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.segment.session as mod

        monkeypatch.setattr(mod, "INTERIM_ROOT", tmp_path / "interim")
        result = segment_session(*roots)
        assert result.root == tmp_path / "interim" / "microsectors" / "synthetic"


@pytest.mark.skipif(not SUZUKA_PRESENT, reason="Suzuka 2024 Q data not present under data/")
class TestSuzukaAcceptance:
    """The spec's acceptance table, on the real session, network-free."""

    def test_criterion_1_corner_phase_partition(self, suzuka) -> None:
        assert suzuka.report.partition_ok
        phases = suzuka.microsectors[suzuka.microsectors["grain"] == GRAIN_CORNER_PHASE]
        assert len(phases) == 35

    def test_criterion_2_event_table_is_the_suzuka_one(self, suzuka) -> None:
        assert suzuka.events["corners"].tolist() == [
            "T1-T2", "T3-T4", "T5", "T6", "T8-T9", "T11", "T13-T14", "T16-T17",
        ]
        unassigned = suzuka.corners.loc[suzuka.corners["event_id"].isna(), "number"].astype(int).tolist()
        assert unassigned == [7, 10, 12, 15, 18]
        assert suzuka.events["marginal"].tolist() == [False, True, True, False, False, False, False, False]
        assert suzuka.events["has_braking"].tolist() == [True, False, False, True, True, True, True, True]

    def test_criterion_3_per_lap_apex_position(self, suzuka) -> None:
        assert suzuka.report.apex_dev.p95 <= validation.APEX_DEV_P95_M

    def test_criterion_4_per_lap_braking_point(self, suzuka) -> None:
        assert suzuka.report.brake_dev.p95 <= validation.BRAKE_DEV_P95_M

    def test_criteria_5_and_6_jackknife(self, suzuka) -> None:
        r = suzuka.report
        assert r.jackknife_count_matches == r.jackknife_trials == 10
        assert r.apex_shift.p95 <= validation.JACKKNIFE_SHIFT_P95_M
        assert r.brake_on_shift.p95 <= validation.JACKKNIFE_SHIFT_P95_M
        assert r.exit_end_shift.p95 <= validation.JACKKNIFE_SHIFT_P95_M

    def test_criterion_7_corners(self, suzuka) -> None:
        assert suzuka.report.corner_offset_max_m <= validation.CORNER_OFFSET_MAX_M
        assert suzuka.report.events_without_corners == 0

    def test_criterion_8_fixed_bins(self, suzuka) -> None:
        assert suzuka.report.fixed_bins == 58 and suzuka.report.fixed_bins_ok

    def test_criterion_9_deterministic(self, suzuka, tmp_path: Path) -> None:
        again = segment_session(SUZUKA_GRID, SUZUKA_SNAPSHOT, SUZUKA_ALIGNED, out_root=tmp_path / "again")
        pd.testing.assert_frame_equal(again.microsectors, suzuka.microsectors)
        pd.testing.assert_frame_equal(again.events, suzuka.events)

    def test_everything_passes(self, suzuka) -> None:
        assert suzuka.report.ok
