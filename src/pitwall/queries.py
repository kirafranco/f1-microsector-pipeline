"""What the pit wall reads, and how it reads it (F017).

Every statement here takes its partition keys as **scalar subqueries** against
`dim_session`, never through a joined CTE. That is F023's finding applied to a
second consumer: `fact_telemetry_grid` is partitioned by
(season, round, session_code), and when those values arrive through a join
Postgres prunes neither at plan time nor at runtime -- it scans all 96
partitions and discards almost everything. As scalar subqueries they become
InitPlans, which runtime pruning does use. A structural test enforces it, so
the pattern cannot come back in a new query.

Connections come from a pool. Measured on the F017 prototype: opening one costs
22 ms and the overlay's first execution 82 ms, while the same three queries on
a reused connection take 11 ms -- psycopg prepares a statement after its fifth
execution on a connection and the plan is cached from then on.

The role is `readonly`. This service reads; the pipeline owns the warehouse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from psycopg.rows import tuple_row
from psycopg_pool import ConnectionPool

from src.warehouse.connection import Settings

#: Kept small on purpose: one warm connection is what removes the 22 ms and the
#: 82 ms above, and a pit wall has one viewer.
POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 4

#: The channels the overlay draws, in the order they are stacked. `step` marks
#: the ones that must never be drawn as a slope: gear, brake and DRS are
#: discrete, exactly as the ingest rules say they are never interpolated.
CHANNELS: tuple[tuple[str, str, bool], ...] = (
    ("delta_t_s", "Δt (B−A)", False),
    ("speed", "Speed (km/h)", False),
    ("throttle", "Throttle (%)", False),
    ("brake", "Brake", True),
    ("n_gear", "Gear", True),
    ("drs", "DRS", True),
)

SESSIONS = """
SELECT session_id,
       season || ' R' || round || ' ' || session_code || ' — ' || event_name AS label
FROM dim_session
ORDER BY season DESC, round DESC, session_code
"""

DRIVERS = """
SELECT code, min(lap_time_s) AS best_s
FROM dim_lap
WHERE session_id = %(session_id)s
GROUP BY code
ORDER BY min(lap_time_s) NULLS LAST
"""

#: F007's `lap_a` rule: a driver's default lap is their fastest, and a lap with
#: no time sorts last rather than first.
FASTEST_LAP = """
SELECT lap_id, lap_number, lap_time_s, compound, stint
FROM dim_lap
WHERE session_id = %(session_id)s AND code = %(code)s
ORDER BY lap_time_s NULLS LAST, lap_number
LIMIT 1
"""

#: The overlay itself. Lap A anchors the partition; lap B joins on A's key
#: columns, which are constants by then, so both sides prune.
OVERLAY = """
SELECT a.distance_m    AS distance_m,
       b.t_s - a.t_s   AS delta_t_s,
       a.speed         AS speed_a,
       b.speed         AS speed_b,
       a.throttle      AS throttle_a,
       b.throttle      AS throttle_b,
       a.brake::int    AS brake_a,
       b.brake::int    AS brake_b,
       a.n_gear        AS n_gear_a,
       b.n_gear        AS n_gear_b,
       a.drs           AS drs_a,
       b.drs           AS drs_b
FROM fact_telemetry_grid a
JOIN fact_telemetry_grid b
  ON b.season = a.season
 AND b.round = a.round
 AND b.session_code = a.session_code
 AND b.lap_id = %(lap_b)s
 AND b.grid_index = a.grid_index
WHERE a.season = (SELECT season FROM dim_session WHERE session_id = %(session_id)s)
  AND a.round = (SELECT round FROM dim_session WHERE session_id = %(session_id)s)
  AND a.session_code = (SELECT session_code FROM dim_session WHERE session_id = %(session_id)s)
  AND a.lap_id = %(lap_a)s
ORDER BY a.grid_index
"""

#: Every statement above that touches a partitioned fact. The structural test
#: reads this rather than scraping the module.
PARTITIONED_STATEMENTS = {"OVERLAY": OVERLAY}


class PitWallError(RuntimeError):
    """The pit wall cannot show what was asked for."""


@dataclass(frozen=True)
class Lap:
    """One side of the comparison."""

    lap_id: int
    lap_number: int
    lap_time_s: float | None
    compound: str | None
    stint: int | None
    code: str

    @property
    def label(self) -> str:
        time = f"{self.lap_time_s:.3f} s" if self.lap_time_s is not None else "no time"
        return f"{self.code} L{self.lap_number} · {time}"


@dataclass(frozen=True)
class Overlay:
    """Two laps on one distance axis, ready to draw."""

    session_id: int
    lap_a: Lap
    lap_b: Lap
    rows: list[tuple[Any, ...]]

    @property
    def columns(self) -> list[str]:
        return ["distance_m", "delta_t_s", "speed_a", "speed_b", "throttle_a", "throttle_b",
                "brake_a", "brake_b", "n_gear_a", "n_gear_b", "drs_a", "drs_b"]

    def column(self, name: str) -> list[Any]:
        return [row[self.columns.index(name)] for row in self.rows]


def build_pool(settings: Settings | None = None, *, open_now: bool = True) -> ConnectionPool:
    """A pool on the read-only role."""
    settings = settings or Settings.from_env(role="readonly")
    pool = ConnectionPool(
        settings.dsn,
        min_size=POOL_MIN_SIZE,
        max_size=POOL_MAX_SIZE,
        open=False,
        kwargs={"row_factory": tuple_row},
        name="pitwall",
    )
    if open_now:
        pool.open(wait=True, timeout=settings.connect_timeout_s)
    return pool


def _fetch(pool: ConnectionPool, statement: str, params: dict[str, Any]) -> list[tuple[Any, ...]]:
    with pool.connection() as conn:
        return conn.execute(statement, params).fetchall()


def sessions(pool: ConnectionPool) -> list[tuple[int, str]]:
    """Every loaded session, newest first."""
    return [(int(row[0]), str(row[1])) for row in _fetch(pool, SESSIONS, {})]


def drivers(pool: ConnectionPool, session_id: int) -> list[str]:
    """Driver codes in a session, fastest first."""
    return [str(row[0]) for row in _fetch(pool, DRIVERS, {"session_id": session_id})]


def fastest_lap(pool: ConnectionPool, session_id: int, code: str) -> Lap:
    rows = _fetch(pool, FASTEST_LAP, {"session_id": session_id, "code": code})
    if not rows:
        raise PitWallError(f"{code} has no lap in session {session_id}")
    lap_id, lap_number, lap_time_s, compound, stint = rows[0]
    return Lap(int(lap_id), int(lap_number),
               None if lap_time_s is None else float(lap_time_s),
               compound, None if stint is None else int(stint), code)


def overlay(pool: ConnectionPool, session_id: int, driver_a: str, driver_b: str,
            lap_a: int | None = None, lap_b: int | None = None) -> Overlay:
    """The two laps to compare, defaulting to each driver's fastest."""
    if driver_a == driver_b:
        raise PitWallError("pick two different drivers")
    a = fastest_lap(pool, session_id, driver_a) if lap_a is None else _lap_by_id(pool, session_id, driver_a, lap_a)
    b = fastest_lap(pool, session_id, driver_b) if lap_b is None else _lap_by_id(pool, session_id, driver_b, lap_b)
    rows = _fetch(pool, OVERLAY, {"session_id": session_id, "lap_a": a.lap_id, "lap_b": b.lap_id})
    if not rows:
        raise PitWallError(f"no shared grid points between {a.label} and {b.label}")
    return Overlay(session_id, a, b, rows)


def _lap_by_id(pool: ConnectionPool, session_id: int, code: str, lap_id: int) -> Lap:
    statement = FASTEST_LAP.replace("ORDER BY lap_time_s NULLS LAST, lap_number", "").replace(
        "AND code = %(code)s", "AND code = %(code)s AND lap_id = %(lap_id)s")
    rows = _fetch(pool, statement, {"session_id": session_id, "code": code, "lap_id": lap_id})
    if not rows:
        raise PitWallError(f"lap {lap_id} is not {code}'s in session {session_id}")
    lap_id_, lap_number, lap_time_s, compound, stint = rows[0]
    return Lap(int(lap_id_), int(lap_number),
               None if lap_time_s is None else float(lap_time_s),
               compound, None if stint is None else int(stint), code)


def statements() -> Sequence[tuple[str, str]]:
    """Every SQL constant in this module, for the structural test."""
    return [("SESSIONS", SESSIONS), ("DRIVERS", DRIVERS),
            ("FASTEST_LAP", FASTEST_LAP), ("OVERLAY", OVERLAY)]
