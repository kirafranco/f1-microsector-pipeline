"""F003 session runner: fault tolerance, idempotency, and the real-data acceptance table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import INTERIM_ROOT
from src.grid import validation
from src.grid.resample import GRID_SCHEMA
from src.grid.session import resample_session
from tests.test_resample import synthetic_lap

SUZUKA_ALIGNED = INTERIM_ROOT / "aligned" / "2024_Japan_Q_projection"


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory):
    """One real-session run shared by the acceptance tests."""
    return resample_session(SUZUKA_ALIGNED, out_root=tmp_path_factory.mktemp("grid"))


def _aligned_snapshot(root: Path, laps: list[pd.DataFrame]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pd.concat(laps, ignore_index=True).to_parquet(root / "telemetry_aligned.parquet", index=False)
    return root


@pytest.fixture()
def snapshot(tmp_path: Path) -> Path:
    good_a = synthetic_lap()
    good_b = synthetic_lap(offset_m=2.0).assign(driver="OTH", lap_number=np.int16(3))
    # One sample only: cannot be resampled, must be recorded and skipped.
    broken = synthetic_lap().iloc[:1].assign(driver="BAD", lap_number=np.int16(1))
    return _aligned_snapshot(tmp_path / "aligned" / "synthetic", [good_a, good_b, broken])


class TestSessionRunner:
    def test_bad_lap_is_recorded_and_the_batch_continues(self, snapshot: Path, tmp_path: Path) -> None:
        result = resample_session(snapshot, out_root=tmp_path / "out")
        assert (result.laps_total, result.laps_resampled, result.laps_rejected) == (3, 2, 1)

        rejected = pd.read_parquet(result.root / "rejected_laps.parquet")
        assert rejected["driver"].tolist() == ["BAD"]
        assert "at least 2" in rejected["reason"].iloc[0]

    def test_output_contract_and_uniqueness(self, snapshot: Path, tmp_path: Path) -> None:
        result = resample_session(snapshot, out_root=tmp_path / "out")
        grid = pd.read_parquet(result.root / "grid.parquet")
        assert len(grid) == result.rows == 2 * 101
        assert list(grid.columns) == list(GRID_SCHEMA)
        for column, dtype in GRID_SCHEMA.items():
            assert str(grid[column].dtype) == dtype, column
        assert not grid.duplicated(["driver", "lap_number", "grid_index"]).any()

    def test_meta_carries_the_acceptance_table(self, snapshot: Path, tmp_path: Path) -> None:
        result = resample_session(snapshot, out_root=tmp_path / "out")
        meta = json.loads((result.root / "grid_meta.json").read_text(encoding="utf-8"))
        assert meta["laps_resampled"] == 2
        assert meta["acceptance"]["checks"]["all"] is True
        assert result.report.ok
        assert "source_gap_m" in meta["limitation"]

    def test_rerun_is_idempotent(self, snapshot: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        resample_session(snapshot, out_root=out)
        first = pd.read_parquet(out / "grid.parquet")
        resample_session(snapshot, out_root=out)
        second = pd.read_parquet(out / "grid.parquet")
        pd.testing.assert_frame_equal(first, second)

    def test_custom_spacing_reaches_the_output(self, snapshot: Path, tmp_path: Path) -> None:
        result = resample_session(snapshot, out_root=tmp_path / "out", grid_m=25.0)
        assert result.rows == 2 * 41

    def test_default_output_root_is_the_grid_interim_layer(
        self, snapshot: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.grid.session as mod

        monkeypatch.setattr(mod, "INTERIM_ROOT", tmp_path / "interim")
        result = resample_session(snapshot)
        assert result.root == tmp_path / "interim" / "grid" / "synthetic"
        assert (result.root / "grid.parquet").exists()

    def test_nothing_resampled_is_an_error(self, tmp_path: Path) -> None:
        broken = synthetic_lap().iloc[:1]
        root = _aligned_snapshot(tmp_path / "aligned" / "empty", [broken])
        with pytest.raises(RuntimeError, match="no lap resampled"):
            resample_session(root, out_root=tmp_path / "out")


@pytest.mark.skipif(
    not (SUZUKA_ALIGNED / "telemetry_aligned.parquet").exists(),
    reason="F008 output for Suzuka 2024 Q not present under data/interim/aligned",
)
class TestSuzukaAcceptance:
    """The spec's acceptance table, on the real session, network-free."""

    def test_every_aligned_lap_is_resampled(self, result) -> None:
        assert result.laps_rejected == 0
        assert result.laps_resampled == result.laps_total

    def test_criterion_1_grid_structure(self, result) -> None:
        assert result.report.structure_ok

    def test_criterion_2_speed_round_trip(self, result) -> None:
        assert result.report.speed.p95 <= validation.SPEED_P95_KMH
        assert result.report.speed.p99 <= validation.SPEED_P99_KMH

    def test_criterion_3_throttle_round_trip(self, result) -> None:
        assert result.report.throttle.p95 <= validation.THROTTLE_P95_PCT

    def test_criterion_4_rpm_round_trip(self, result) -> None:
        assert result.report.rpm.p95 <= validation.RPM_P95

    def test_criterion_5_no_invented_discrete_values(self, result) -> None:
        assert result.report.invented == {"n_gear": 0, "brake": 0, "drs": 0}

    def test_criterion_6_brake_edge_displacement(self, result) -> None:
        assert result.report.brake_edge.p95 <= validation.BRAKE_EDGE_P95_M

    def test_criterion_7_elapsed_time_increasing(self, result) -> None:
        assert result.report.elapsed_ok

    def test_grid_is_unique_per_driver_lap_point(self, result) -> None:
        grid = pd.read_parquet(result.root / "grid.parquet")
        assert not grid.duplicated(["driver", "lap_number", "grid_index"]).any()
        assert len(grid) == result.rows

    def test_the_documented_limitation_is_real_and_reported(self, result) -> None:
        """Not a gate: the empty-bin share is inherent to the source and belongs in docs."""
        assert 0.05 < result.report.empty_bin_fraction < 0.5
        assert np.isfinite(result.report.source_gap.max)
