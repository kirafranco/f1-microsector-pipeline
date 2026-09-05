"""One function per pipeline task, each returning something Airflow can store.

These are adapters, not logic. Every one of them resolves a `SessionRun` into
the roots that stage needs, calls the entry point that already exists in
`src/`, and reduces its result object to a small JSON-serialisable dictionary
-- because an XCom is JSON, and a DataFrame is not.

Keeping them here rather than in the DAG file means the whole pipeline can be
run end to end on the designed synthetic session in a unit test, with no
scheduler involved.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

from src.align.session import align_session
from src.grid.session import resample_session
from src.ingest.fastf1_source import ingest_session
from src.metrics.session import compute_metrics
from src.orchestration.paths import SessionRun, latest_snapshot_root
from src.quality.session import check_session
from src.segment.session import segment_session
from src.validate.session import validate_session
from src.warehouse.connection import Settings, connect
from src.warehouse.load import load_session
from src.warehouse.migrations import migrate

logger = logging.getLogger(__name__)


class StageError(RuntimeError):
    """A stage cannot run because the one before it left nothing behind."""


def _snapshot(run: SessionRun, snapshot_date: str | None) -> Path:
    """The snapshot this run works from, whichever day it was taken.

    Snapshots are dated and immutable, so a pipeline re-run on a later day
    must not silently ingest a second copy for the downstream stages to
    disagree about.
    """
    if snapshot_date:
        root = run.snapshot_root(snapshot_date)
        if not (root / "manifest.json").exists():
            raise StageError(f"no snapshot at {root}")
        return root
    root = latest_snapshot_root(run)
    if root is None:
        raise StageError(f"no snapshot for {run.label}; the ingest task has not run")
    return root


def ingest(run: SessionRun, snapshot_date: str | None = None,
           drivers: list[str] | None = None) -> dict:
    """Fetch the session into an immutable raw snapshot (F002).

    Idempotent by date: a snapshot already taken today is reused, so a
    re-triggered run costs nothing and cannot produce a second copy.
    """
    parsed = date.fromisoformat(snapshot_date) if snapshot_date else None
    snapshot = ingest_session(run.season, run.event, run.session,
                              snapshot_date=parsed, drivers=drivers)
    logger.info("stage_ingest %s root=%s drivers=%d skipped=%d",
                run.label, snapshot.root, len(snapshot.drivers_ingested),
                len(snapshot.drivers_skipped))
    return {
        "root": str(snapshot.root),
        "snapshot_date": snapshot.snapshot_date.isoformat(),
        "drivers_ingested": len(snapshot.drivers_ingested),
        "drivers_skipped": sorted(snapshot.drivers_skipped),
    }


def align(run: SessionRun, snapshot_date: str | None = None) -> dict:
    """Project every flying lap onto the session's reference line (F008)."""
    result = align_session(_snapshot(run, snapshot_date), out_root=run.aligned_root,
                           method=run.method)
    logger.info("stage_align %s aligned=%d/%d residual_m=%.2f",
                run.label, result.laps_aligned, result.laps_total, result.median_residual_m)
    return {
        "root": str(result.root),
        "laps_total": int(result.laps_total),
        "laps_aligned": int(result.laps_aligned),
        "laps_rejected": int(result.laps_rejected),
        "median_residual_m": round(float(result.median_residual_m), 3),
        "method": str(result.method),
        "reference_line_lap": str(result.reference_line_lap),
    }


def grid(run: SessionRun) -> dict:
    """Resample every aligned lap onto the shared 10 m axis (F003)."""
    result = resample_session(run.aligned_root, out_root=run.grid_root)
    logger.info("stage_grid %s rows=%d laps=%d/%d",
                run.label, result.rows, result.laps_resampled, result.laps_total)
    return {
        "root": str(result.root),
        "grid_m": float(result.grid_m),
        "rows": int(result.rows),
        "laps_resampled": int(result.laps_resampled),
        "laps_rejected": int(result.laps_rejected),
        "elapsed_s": round(float(result.elapsed_s), 2),
    }


def segment(run: SessionRun, snapshot_date: str | None = None) -> dict:
    """Find the corner events and cut both micro-sector grains (F009)."""
    result = segment_session(run.grid_root, _snapshot(run, snapshot_date), run.aligned_root,
                             out_root=run.microsector_root)
    logger.info("stage_segment %s events=%d microsectors=%d",
                run.label, len(result.events), len(result.microsectors))
    return {
        "root": str(result.root),
        "lap_length_m": round(float(result.lap_length_m), 1),
        "events": int(len(result.events)),
        "microsectors": int(len(result.microsectors)),
        "partition_ok": bool(result.report.partition_ok),
        "elapsed_s": round(float(result.elapsed_s), 2),
    }


def metrics(run: SessionRun, snapshot_date: str | None = None) -> dict:
    """Delta curves, micro-sector times and corner metrics (F004)."""
    result = compute_metrics(run.grid_root, run.microsector_root, _snapshot(run, snapshot_date),
                             run.aligned_root, out_root=run.processed_root)
    logger.info("stage_metrics %s reference=%s laps=%d",
                run.label, result.report.reference_label, len(result.lap_summary))
    return {
        "root": str(result.root),
        "reference": str(result.report.reference_label),
        "laps": int(len(result.lap_summary)),
        "delta_rows": int(len(result.delta)),
        "microsector_rows": int(len(result.times)),
        "checks_ok": bool(result.report.ok),
        "elapsed_s": round(float(result.elapsed_s), 2),
    }


def validate(run: SessionRun, snapshot_date: str | None = None) -> dict:
    """Reconstruct official timing from the pipeline's own output (F010).

    Reported, not gated: a session whose residuals are poor is still loaded,
    with the numbers recorded, because "this circuit aligns badly" is a finding
    and F015 is the feature that acts on it.
    """
    result = validate_session(_snapshot(run, snapshot_date), run.aligned_root, run.grid_root,
                              run.processed_root, out_root=run.processed_root)
    report = result.report
    logger.info("stage_validate %s laps=%d checks_ok=%s closure_p95=%.3f",
                run.label, report.laps, report.ok, report.closure.p95)
    return {
        "root": str(result.root),
        "laps": int(report.laps),
        "laps_gated": int(report.laps_gated),
        "flagged": list(report.flagged),
        "lap_residual_p95_s": round(float(report.lap_residual.p95), 4),
        "closure_p95_s": round(float(report.closure.p95), 4),
        "checks_ok": bool(report.ok),
        "elapsed_s": round(float(result.elapsed_s), 2),
    }


def quality(run: SessionRun, snapshot_date: str | None = None) -> dict:
    """Run the declarative contracts over every artefact (F011).

    This task failing is the point of it: `load` never runs, and nothing
    reaches the warehouse.
    """
    result = check_session(_snapshot(run, snapshot_date), run.aligned_root, run.grid_root,
                           run.microsector_root, run.processed_root, out_root=run.processed_root)
    report = result.report
    # tables, errors and warnings are tuples of findings, not counts.
    errors, warnings = len(report.errors), len(report.warnings)
    logger.info("stage_quality %s ok=%s errors=%d warnings=%d",
                run.label, report.ok, errors, warnings)
    summary = {
        "root": str(result.root),
        "ok": bool(report.ok),
        "tables": len(report.tables),
        "errors": errors,
        "warnings": warnings,
        "contract_version": str(report.contract_version),
        "elapsed_s": round(float(result.elapsed_s), 2),
    }
    if not report.ok:
        raise StageError(f"{run.label}: quality gate failed with {errors} error(s); "
                         f"see {result.root}/quality_findings.parquet")
    return summary


def load(run: SessionRun, snapshot_date: str | None = None,
         settings: Settings | None = None, reference_root: Path | None = None) -> dict:
    """Load the session into the star schema, in one transaction (F005).

    Migrations run first and are idempotent, so a fresh database becomes a
    warehouse on the first pipeline run rather than needing a separate step.
    """
    settings = settings or Settings.from_env()
    snapshot = _snapshot(run, snapshot_date)
    with connect(settings) as connection:
        applied = migrate(connection)
        result = load_session(snapshot, run.aligned_root, run.grid_root, run.microsector_root,
                              run.processed_root, connection,
                              reference_root=reference_root, season_for_reference=run.season)
    logger.info("stage_load %s session_id=%d rows=%s", run.label, result.session_id, result.rows)
    return {
        "session_id": int(result.session_id),
        "load_id": int(result.load_id),
        "migrations_applied": [f"V{m.version:03d}__{m.name}" for m in applied],
        "rows": {name: int(count) for name, count in result.rows.items()},
        "partitions": list(result.partitions),
        "elapsed_s": round(float(result.elapsed_s), 2),
    }


#: The pipeline in order, for the DAG to wire and for a test to walk.
PIPELINE: tuple[tuple[str, Any], ...] = (
    ("ingest", ingest),
    ("align", align),
    ("grid", grid),
    ("segment", segment),
    ("metrics", metrics),
    ("validate", validate),
    ("quality", quality),
    ("load", load),
)
