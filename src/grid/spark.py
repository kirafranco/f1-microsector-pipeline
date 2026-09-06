"""Spark as the distribution layer for F003's resampling (F013, decision D2).

The maths is not Spark's. `resample_lap` stays a pure NumPy function with no
JVM, and this module only spreads it across laps with the idiomatic grouped-map
pandas UDF -- `groupBy(driver, lap).applyInPandas(fn, schema=...)` -- declaring
the output schema explicitly, as global CLAUDE.md 3.3 requires.

The client talks to a Spark Connect server in the `spark` container, so nothing
here needs a JVM. Connect pickles a UDF **by reference** when its module is
importable on the server, which is why the project is bind-mounted there and put
on PYTHONPATH: only the import path crosses the wire, never bytecode. Client and
server run the same Python minor and the same pinned pandas, numpy and pyarrow.

D2 recorded honestly that this is not justified by data volume -- the pandas
executor does the whole 2024 season in 256 s -- so the point of proof here is
not speed but equivalence: the output of this module must be the same frame the
pandas executor produces, which `finish_session` guarantees for everything after
the per-lap call.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path, PurePosixPath

import pandas as pd

from src.grid.resample import (
    GRID_SCHEMA,
    GRID_SPACING_M,
    KEY_COLUMNS,
    ResampleError,
    resample_lap,
)
from src.grid.session import ResampleResult, finish_session

logger = logging.getLogger(__name__)

#: Where the compose mounts the project inside the Spark container. The client
#: sends a path the *server* must be able to open, so a host path is rewritten
#: onto this root before it is handed over.
CONTAINER_PROJECT_ROOT = PurePosixPath("/opt/project")

#: The Connect endpoint. `spark` is the compose service name (never localhost
#: between containers, global CLAUDE.md 2); from the conda env on the host it is
#: localhost and the port published in servicios/.env.
DEFAULT_REMOTE = "sc://localhost:15002"

#: Scratch inside the project, so the Spark container can see it. Anything the
#: server has to read must live under the bind mount; `container_path` refuses
#: the rest rather than handing over a path that resolves to nothing there.
PROJECT_TMP = "data/tmp"

#: `GRID_SCHEMA` in Spark's vocabulary. Kept beside it and pinned by a test, so
#: a column added to one and not the other fails offline rather than at runtime.
_SPARK_TYPES = {
    "string": "string",
    "Int32": "int",
    "Int16": "smallint",
    "Int8": "tinyint",
    "float32": "float",
    "boolean": "boolean",
}
GRID_SPARK_SCHEMA = ", ".join(f"{name} {_SPARK_TYPES[dtype]}" for name, dtype in GRID_SCHEMA.items())


def _resample_group(lap: pd.DataFrame) -> pd.DataFrame:
    """One lap, on a Spark worker: the same call the pandas executor makes.

    A lap the maths refuses yields no rows rather than an exception, so one bad
    lap cannot fail the session (global CLAUDE.md 3.1). Which laps those were is
    recovered client-side afterwards, with their reasons.
    """
    frame = lap.sort_values("session_time").reset_index(drop=True)
    try:
        return resample_lap(frame, grid_m=float(os.environ.get("F1_GRID_SPACING_M", GRID_SPACING_M)))
    except ResampleError:
        return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in GRID_SCHEMA.items()})


def container_path(path: Path, project_root: Path | None = None) -> str:
    """A host path rewritten onto the Spark container's view of the project."""
    from src.config import PROJECT_ROOT

    root = Path(project_root or PROJECT_ROOT).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{path} is outside the project root {root}; the Spark container cannot see it") from exc
    return str(CONTAINER_PROJECT_ROOT / PurePosixPath(*relative.parts))


def resample_session_spark(
    aligned_root: Path,
    out_root: Path | None = None,
    grid_m: float = GRID_SPACING_M,
    remote: str | None = None,
) -> ResampleResult:
    """Resample a session's laps on Spark and write the same files as the pandas runner."""
    from pyspark.sql import SparkSession

    started = time.perf_counter()
    remote = remote or os.environ.get("SPARK_CONNECT_URL", DEFAULT_REMOTE)
    source = container_path(Path(aligned_root) / "telemetry_aligned.parquet")

    spark = SparkSession.builder.remote(remote).appName("f1-grid-resample").getOrCreate()
    try:
        frame = spark.read.parquet(source)
        grouped = frame.groupBy(*KEY_COLUMNS).applyInPandas(_resample_group, schema=GRID_SPARK_SCHEMA)
        grid = grouped.toPandas()
        laps_total = frame.select(*KEY_COLUMNS).distinct().count()
    finally:
        spark.stop()

    logger.info("spark_resample remote=%s laps=%d rows=%d", remote, laps_total, len(grid))

    # Rebuild the per-lap frames the acceptance measurement needs, and recover
    # the reasons for laps Spark returned nothing for. Both are client-side and
    # cheap: the aligned parquet is the same file Spark just read, and the
    # rejected laps are a handful.
    telemetry = pd.read_parquet(Path(aligned_root) / "telemetry_aligned.parquet")
    grid = grid.astype(GRID_SCHEMA).sort_values([*KEY_COLUMNS, "grid_index"]).reset_index(drop=True)
    produced = set(map(tuple, grid[list(KEY_COLUMNS)].drop_duplicates().to_numpy()))

    grids: list[pd.DataFrame] = []
    pairs: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    rejected: list[dict] = []
    for key, lap in telemetry.groupby(list(KEY_COLUMNS), observed=True, sort=True):
        driver, lap_number = key
        lap = lap.sort_values("session_time").reset_index(drop=True)
        if key in produced:
            mask = (grid["driver"] == driver) & (grid["lap_number"] == lap_number)
            piece = grid[mask].reset_index(drop=True)
            grids.append(piece)
            pairs.append((lap, piece))
            continue
        try:
            resample_lap(lap, grid_m=grid_m)
        except ResampleError as exc:
            rejected.append({"driver": driver, "lap_number": lap_number, "reason": str(exc)})
            logger.warning("lap_rejected driver=%s lap=%s reason=%s", driver, lap_number, exc)
            continue
        raise RuntimeError(f"{aligned_root}: Spark produced no rows for {driver} L{lap_number} but the lap resamples")

    return finish_session(
        Path(aligned_root), grids, pairs, rejected, len(telemetry.groupby(list(KEY_COLUMNS), observed=True).size()),
        out_root, grid_m, started, executor="spark",
    )
