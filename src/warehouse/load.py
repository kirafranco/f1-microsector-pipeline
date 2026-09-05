"""Loading one session into the warehouse: gate, write, audit, all or nothing."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

from src.quality.contracts import CONTRACTS, SESSION_TABLES
from src.quality.engine import QualityReport, require, validate_tables
from src.quality.session import load_artefacts
from src.reference.session import load_reference
from src.warehouse import dimensions as build
from src.warehouse.dimensions import read_json

logger = logging.getLogger(__name__)

SESSION_CONTRACTS = {name: CONTRACTS[name] for name in sorted(SESSION_TABLES)}

PARTITIONED_FACTS = ("fact_telemetry_grid", "fact_microsector")


def snapshot_date_of(snapshot_root: Path) -> str | None:
    """The snapshot date from ``data/raw/fastf1/<date>/<session>``.

    A snapshot kept anywhere else -- a test fixture, a one-off directory --
    simply has no date, rather than sending a folder name to a date column.
    """
    name = Path(snapshot_root).parent.name
    try:
        date.fromisoformat(name)
    except ValueError:
        return None
    return name


@dataclass(frozen=True)
class LoadResult:
    session_id: int
    load_id: int
    rows: dict[str, int]
    quality: QualityReport
    elapsed_s: float
    partitions: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.quality.ok


def _values(frame: pd.DataFrame) -> list[tuple]:
    """Rows as plain Python tuples, with every missing value as ``None``.

    pandas' nullable scalars and numpy's NaN both reach psycopg as things
    Postgres cannot read, so the conversion happens once, here.
    """
    prepared = frame.astype(object).where(pd.notna(frame), None)
    return [tuple(row) for row in prepared.itertuples(index=False, name=None)]


def copy_frame(cursor: psycopg.Cursor, table: str, frame: pd.DataFrame) -> int:
    """COPY a frame into a table. Measured at ~52,000 rows/s on the bind mount."""
    if frame.empty:
        return 0
    statement = sql.SQL("COPY {} ({}) FROM STDIN").format(
        sql.Identifier(table), sql.SQL(", ").join(sql.Identifier(c) for c in frame.columns)
    )
    with cursor.copy(statement) as copy:
        for row in _values(frame):
            copy.write_row(row)
    return len(frame)


def upsert(cursor: psycopg.Cursor, table: str, frame: pd.DataFrame, conflict: Sequence[str],
           update: Sequence[str] | None = None) -> int:
    """INSERT ... ON CONFLICT DO UPDATE, so a reload refreshes rather than fails."""
    if frame.empty:
        return 0
    columns = list(frame.columns)
    updatable = [c for c in (update if update is not None else columns) if c not in conflict]
    assignment = ", ".join(f"{c} = EXCLUDED.{c}" for c in updatable) or f"{conflict[0]} = EXCLUDED.{conflict[0]}"
    statement = sql.SQL(
        "INSERT INTO {table} ({columns}) VALUES ({placeholders}) ON CONFLICT ({conflict}) DO UPDATE SET {assignment}"
    ).format(
        table=sql.Identifier(table),
        columns=sql.SQL(", ").join(sql.Identifier(c) for c in columns),
        placeholders=sql.SQL(", ").join(sql.Placeholder() * len(columns)),
        conflict=sql.SQL(", ").join(sql.Identifier(c) for c in conflict),
        assignment=sql.SQL(assignment),
    )
    cursor.executemany(statement, _values(frame))
    return len(frame)


def ensure_partitions(cursor: psycopg.Cursor, season: int, round_number: int, session_code: str) -> list[str]:
    """Create this session's event and session partitions if they do not exist.

    The event level is what decision D10 asks for and what pruning uses; the
    session leaf is what makes a reload a TRUNCATE rather than a DELETE.
    """
    # Partition bounds must be literals -- Postgres does not accept parameters
    # in a FOR VALUES clause -- so they are composed with psycopg's SQL objects
    # rather than interpolated by hand. season and round are already ints and
    # session_code is quoted as a literal.
    season, round_number = int(season), int(round_number)
    created: list[str] = []
    code = "".join(c for c in session_code.lower() if c.isalnum())
    for parent in PARTITIONED_FACTS:
        event_partition = f"{parent}_{season}_{round_number:02d}"
        leaf = f"{event_partition}_{code}"
        cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {event} PARTITION OF {parent} "
                "FOR VALUES FROM ({season}, {low}) TO ({season}, {high}) PARTITION BY LIST (session_code)"
            ).format(
                event=sql.Identifier(event_partition), parent=sql.Identifier(parent),
                season=sql.Literal(season), low=sql.Literal(round_number), high=sql.Literal(round_number + 1),
            )
        )
        cursor.execute(
            sql.SQL("CREATE TABLE IF NOT EXISTS {leaf} PARTITION OF {event} FOR VALUES IN ({code})").format(
                leaf=sql.Identifier(leaf), event=sql.Identifier(event_partition), code=sql.Literal(session_code),
            )
        )
        created.append(leaf)
    return created


def _session_id(cursor: psycopg.Cursor, row: Mapping) -> int:
    columns = list(row)
    natural = ("season", "round", "session_code")
    assignment = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c not in natural) + ", loaded_at = now()"
    statement = sql.SQL(
        "INSERT INTO dim_session ({columns}) VALUES ({placeholders}) "
        "ON CONFLICT (season, round, session_code) DO UPDATE SET {assignment} RETURNING session_id"
    ).format(
        columns=sql.SQL(", ").join(sql.Identifier(c) for c in columns),
        placeholders=sql.SQL(", ").join(sql.Placeholder() * len(columns)),
        assignment=sql.SQL(assignment),
    )
    values = tuple(None if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in row.values())
    cursor.execute(statement, values)
    return int(cursor.fetchone()[0])


def load_session(
    snapshot_root: Path,
    aligned_root: Path,
    grid_root: Path,
    microsector_root: Path,
    processed_root: Path,
    connection: psycopg.Connection,
    reference_root: Path | None = None,
    season_for_reference: int | None = None,
) -> LoadResult:
    """Load one session. Either it is completely in the warehouse, or not at all.

    The quality gate runs before a single row is written: an error finding
    means nothing is inserted, not even a dimension.
    """
    started = time.perf_counter()
    frames = load_artefacts(snapshot_root, aligned_root, grid_root, microsector_root, processed_root)
    session_meta = read_json(Path(snapshot_root) / "session_meta.json")
    alignment_meta = read_json(Path(aligned_root) / "alignment_meta.json")
    season, round_number, session_code = build.session_identity(session_meta)

    report = validate_tables(frames, SESSION_CONTRACTS)
    require(report)  # refuses before anything is written
    logger.info("load_gate_passed season=%d round=%d session=%s warnings=%d",
                season, round_number, session_code, len(report.warnings))

    reference = load_reference(season_for_reference or season, root=reference_root)
    session_row = build.build_dim_session(
        session_meta, alignment_meta, reference,
        snapshot_date=snapshot_date_of(snapshot_root),
        contract_version=report.contract_version,
    )

    with connection.transaction():
        with connection.cursor() as cursor:
            drivers = reference.get("dim_driver")
            constructors = reference.get("dim_constructor")
            if constructors is not None and not constructors.empty:
                upsert(cursor, "dim_constructor", constructors[["season", "constructor_id", "name", "nationality"]],
                       conflict=("season", "constructor_id"))
            if drivers is not None and not drivers.empty:
                columns = ["season", "code", "driver_id", "permanent_number", "given_name",
                           "family_name", "full_name", "nationality", "date_of_birth"]
                upsert(cursor, "dim_driver", drivers[columns], conflict=("season", "code"))

            session_id = _session_id(cursor, session_row)

            laps = build.build_dim_lap(
                frames["lap_summary"], frames.get("ground_truth"), session_id, season, round_number, session_code,
                driver_entry=reference.get("driver_entry"),
            )
            upsert(cursor, "dim_lap", laps, conflict=("session_id", "code", "lap_number"))
            cursor.execute("SELECT code, lap_number, lap_id FROM dim_lap WHERE session_id = %s", (session_id,))
            lap_ids = {(str(code), int(number)): int(lap_id) for code, number, lap_id in cursor.fetchall()}

            cursor.execute("DELETE FROM dim_microsector WHERE session_id = %s", (session_id,))
            copy_frame(cursor, "dim_microsector",
                       build.build_dim_microsector(frames["microsectors"], session_id))
            cursor.execute("DELETE FROM dim_corner_event WHERE session_id = %s", (session_id,))
            copy_frame(cursor, "dim_corner_event", build.build_dim_corner_event(frames["events"], session_id))

            partitions = ensure_partitions(cursor, season, round_number, session_code)
            for leaf in partitions:
                cursor.execute(sql.SQL("TRUNCATE {}").format(sql.Identifier(leaf)))
            cursor.execute(
                "DELETE FROM fact_corner_metric WHERE lap_id IN (SELECT lap_id FROM dim_lap WHERE session_id = %s)",
                (session_id,),
            )

            grid_rows = copy_frame(cursor, "fact_telemetry_grid", build.build_fact_grid(
                frames["grid"], frames.get("delta_t"), lap_ids, season, round_number, session_code))
            microsector_rows = copy_frame(cursor, "fact_microsector", build.build_fact_microsector(
                frames["microsector_times"], lap_ids, season, round_number, session_code))
            corner_rows = copy_frame(cursor, "fact_corner_metric", build.build_fact_corner_metric(
                frames["corner_metrics"], lap_ids))

            expected = {
                "fact_telemetry_grid": len(frames["grid"]),
                "fact_microsector": len(frames["microsector_times"]),
                "fact_corner_metric": len(frames["corner_metrics"]),
            }
            for table, count in expected.items():
                cursor.execute(
                    sql.SQL("SELECT count(*) FROM {} WHERE lap_id IN "
                            "(SELECT lap_id FROM dim_lap WHERE session_id = %s)").format(sql.Identifier(table)),
                    (session_id,),
                )
                stored = int(cursor.fetchone()[0])
                if stored != count:
                    raise RuntimeError(f"{table}: loaded {stored} rows, expected {count}")

            elapsed = time.perf_counter() - started
            cursor.execute(
                "INSERT INTO load_audit (session_id, contract_version, quality_ok, quality_warnings, "
                "rows_grid, rows_microsector, rows_corner, rows_lap, snapshot_paths, elapsed_s) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING load_id",
                (session_id, report.contract_version, report.ok, len(report.warnings),
                 grid_rows, microsector_rows, corner_rows, len(laps),
                 json.dumps({"snapshot": str(snapshot_root), "aligned": str(aligned_root),
                             "grid": str(grid_root), "microsectors": str(microsector_root),
                             "processed": str(processed_root)}),
                 elapsed),
            )
            load_id = int(cursor.fetchone()[0])

    elapsed = time.perf_counter() - started
    rows = {"dim_lap": len(laps), "fact_telemetry_grid": grid_rows,
            "fact_microsector": microsector_rows, "fact_corner_metric": corner_rows}
    logger.info(
        "load_complete session_id=%d load_id=%d season=%d round=%d session=%s laps=%d grid=%d "
        "microsector=%d corner=%d warnings=%d elapsed_s=%.2f",
        session_id, load_id, season, round_number, session_code, len(laps), grid_rows,
        microsector_rows, corner_rows, len(report.warnings), elapsed,
    )
    return LoadResult(session_id=session_id, load_id=load_id, rows=rows, quality=report,
                      elapsed_s=elapsed, partitions=tuple(partitions))
