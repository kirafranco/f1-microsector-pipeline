"""F006 stage adapters: the right entry point, the right roots, a JSON summary.

Two kinds of test. The wiring tests replace each `src` entry point and assert
what the adapter passed it -- a stage handed the wrong root would otherwise
produce a perfectly valid run over the wrong session. The rest execute for real
against the designed synthetic session, so the summaries are shaped by actual
result objects rather than by what this module hopes they look like.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.metrics.session import compute_metrics
from src.orchestration import stages
from src.orchestration.paths import SessionRun
from src.orchestration.stages import StageError
from tests import synthetic_session as syn

RUN = SessionRun(season=2024, event="Japanese Grand Prix", session="Q")


@pytest.fixture()
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A dated snapshot on a temporary raw root, with a manifest."""
    root = tmp_path / "2026-09-05" / RUN.snapshot_slug
    root.mkdir(parents=True)
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("src.orchestration.paths.FASTF1_RAW_ROOT", tmp_path)
    return root


class TestSnapshotResolution:
    def test_an_explicit_date_is_honoured(self, snapshot: Path, monkeypatch) -> None:
        monkeypatch.setattr(RUN.__class__, "snapshot_root",
                            lambda self, when: snapshot.parent.parent / str(when) / self.snapshot_slug)
        assert stages._snapshot(RUN, "2026-09-05") == snapshot

    def test_without_a_date_the_newest_snapshot_is_used(self, snapshot: Path) -> None:
        assert stages._snapshot(RUN, None) == snapshot

    def test_a_missing_snapshot_names_the_stage_that_should_have_run(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("src.orchestration.paths.FASTF1_RAW_ROOT", tmp_path)
        with pytest.raises(StageError, match="the ingest task has not run"):
            stages._snapshot(RUN, None)

    def test_a_named_date_with_no_snapshot_is_an_error(self, snapshot: Path) -> None:
        with pytest.raises(StageError, match="no snapshot at"):
            stages._snapshot(RUN, "2020-01-01")


class TestWiring:
    """Each adapter must reach its own stage with this run's own roots."""

    def test_ingest_passes_the_session_identity_through(self, monkeypatch) -> None:
        seen = {}

        def fake(season, event, session, snapshot_date=None, drivers=None):
            seen.update(season=season, event=event, session=session,
                        snapshot_date=snapshot_date, drivers=drivers)
            return _SnapshotStub()

        monkeypatch.setattr(stages, "ingest_session", fake)
        summary = stages.ingest(RUN, "2026-09-05", drivers=["VER"])
        assert seen["season"] == 2024 and seen["event"] == "Japanese Grand Prix"
        assert seen["session"] == "Q" and seen["drivers"] == ["VER"]
        assert seen["snapshot_date"].isoformat() == "2026-09-05"
        assert summary["snapshot_date"] == "2026-09-05"

    def test_align_writes_to_this_runs_aligned_root(self, snapshot: Path, monkeypatch) -> None:
        seen = {}

        def fake(snapshot_root, out_root=None, method="projection", **_):
            seen.update(snapshot_root=snapshot_root, out_root=out_root, method=method)
            return _AlignStub()

        monkeypatch.setattr(stages, "align_session", fake)
        stages.align(RUN, "2026-09-05")
        assert seen["snapshot_root"] == snapshot
        assert seen["out_root"] == RUN.aligned_root
        assert seen["method"] == "projection"

    def test_grid_reads_aligned_and_writes_grid(self, monkeypatch) -> None:
        seen = {}

        def fake(aligned_root, out_root=None, **_):
            seen.update(aligned_root=aligned_root, out_root=out_root)
            return _GridStub()

        monkeypatch.setattr(stages, "resample_session", fake)
        stages.grid(RUN)
        assert seen["aligned_root"] == RUN.aligned_root
        assert seen["out_root"] == RUN.grid_root

    def test_load_uses_every_root_and_the_run_s_season(self, snapshot: Path, monkeypatch) -> None:
        seen = {}

        class FakeConnection:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        def fake_load(snapshot_root, aligned_root, grid_root, microsector_root, processed_root,
                      connection, reference_root=None, season_for_reference=None):
            seen.update(snapshot_root=snapshot_root, aligned_root=aligned_root, grid_root=grid_root,
                        microsector_root=microsector_root, processed_root=processed_root,
                        season_for_reference=season_for_reference)
            return _LoadStub()

        monkeypatch.setattr(stages, "connect", lambda *_a, **_k: FakeConnection())
        monkeypatch.setattr(stages, "migrate", lambda *_a, **_k: [])
        monkeypatch.setattr(stages, "load_session", fake_load)
        summary = stages.load(RUN, "2026-09-05", settings=object())
        assert seen["snapshot_root"] == snapshot
        assert seen["grid_root"] == RUN.grid_root
        assert seen["processed_root"] == RUN.processed_root
        assert seen["season_for_reference"] == 2024
        assert summary["session_id"] == 7


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """The designed session, taken as far as the processed layer."""
    root = tmp_path_factory.mktemp("stages")
    roots = syn.write_full_session(root)
    processed = root / "processed" / "synthetic"
    compute_metrics(roots["grid_root"], roots["microsector_root"], roots["snapshot_root"],
                    roots["aligned_root"], out_root=processed)
    return {**roots, "processed": processed, "root": root}


@pytest.fixture()
def synthetic_run(synthetic: dict, monkeypatch: pytest.MonkeyPatch) -> SessionRun:
    """A run whose own properties resolve to the synthetic session's roots.

    The data layers are redirected rather than the properties overridden, so
    what the adapters receive is still whatever `SessionRun` computes.
    """
    monkeypatch.setattr("src.orchestration.paths.INTERIM_ROOT", synthetic["root"])
    monkeypatch.setattr("src.orchestration.paths.PROCESSED_ROOT", synthetic["root"] / "processed")
    monkeypatch.setattr(SessionRun, "slug", property(lambda _: "synthetic"))
    monkeypatch.setattr(stages, "_snapshot", lambda *_a, **_k: synthetic["snapshot_root"])
    return SessionRun(season=2024, event="Synthetic", session="Q")


class TestSummariesAreJson:
    """XComs are JSON. A DataFrame or a Path in a summary fails at runtime."""

    def test_the_run_resolves_to_the_synthetic_roots(self, synthetic_run, synthetic) -> None:
        """Guards every test below: without this they would run on empty dirs."""
        assert synthetic_run.grid_root == synthetic["grid_root"]
        assert synthetic_run.aligned_root == synthetic["aligned_root"]
        assert synthetic_run.processed_root == synthetic["processed"]

    def test_metrics_summary_serialises(self, synthetic_run) -> None:
        summary = stages.metrics(synthetic_run)
        assert json.loads(json.dumps(summary)) == summary
        assert summary["laps"] > 0 and summary["reference"]
        assert isinstance(summary["checks_ok"], bool)

    def test_validate_reports_rather_than_gates(self, synthetic_run) -> None:
        """A poorly aligned circuit is a finding for F015, not a failed run."""
        summary = stages.validate(synthetic_run)
        assert json.loads(json.dumps(summary)) == summary
        assert isinstance(summary["checks_ok"], bool)
        assert isinstance(summary["flagged"], list)

    def test_quality_raises_rather_than_returning_a_failure(self, synthetic_run, monkeypatch) -> None:
        """The gate has to stop the graph; a summary saying ok=False would let
        the load task run anyway."""
        class Failing:
            root = Path("processed")
            elapsed_s = 0.1

            class report:
                ok = False
                tables = ()
                errors = ("one",)
                warnings = ()
                contract_version = "abc123"

        monkeypatch.setattr(stages, "check_session", lambda *_a, **_k: Failing())
        with pytest.raises(StageError, match="quality gate failed with 1 error"):
            stages.quality(synthetic_run)


class TestThePipelineIsDeclaredInOrder:
    def test_every_stage_is_listed_once_and_in_order(self) -> None:
        names = [name for name, _ in stages.PIPELINE]
        assert names == ["ingest", "align", "grid", "segment", "metrics",
                         "validate", "quality", "load"]
        assert len(set(names)) == len(names)

    def test_each_entry_points_at_a_callable_in_this_module(self) -> None:
        for name, function in stages.PIPELINE:
            assert getattr(stages, name) is function

    def test_the_gate_sits_before_the_load(self) -> None:
        """F011's contract: nothing is written when quality fails."""
        names = [name for name, _ in stages.PIPELINE]
        assert names.index("quality") < names.index("load")


class _AlignStub:
    root = Path("aligned")
    laps_total, laps_aligned, laps_rejected = 74, 72, 2
    median_residual_m = 0.09
    method = "projection"
    reference_line_lap = "VER L11"


class _GridStub:
    root = Path("grid")
    grid_m = 10.0
    rows, laps_resampled, laps_total, laps_rejected = 42418, 74, 74, 0
    elapsed_s = 1.2


class _LoadStub:
    session_id, load_id = 7, 3
    rows = {"fact_telemetry_grid": 42418}
    partitions = ("fact_telemetry_grid_2024_04_q",)
    elapsed_s = 3.8


class _SnapshotStub:
    from datetime import date as _date

    root = Path("raw") / "2026-09-05" / "2024_Japanese-Grand-Prix_Q"
    snapshot_date = _date(2026, 9, 5)
    drivers_ingested = ["VER", "PER"]
    drivers_skipped = {"HUL": "no laps"}
