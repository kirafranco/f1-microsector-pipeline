"""What the warehouse already holds, for the dispatcher to subtract.

Deliberately one query and nothing else. The dispatcher's only question is
which sessions it need not trigger, and `dim_session` answers it: F005 writes
that row inside the same transaction as the facts, so a session that appears
here is completely loaded, never half.
"""

from __future__ import annotations

import logging

import psycopg

logger = logging.getLogger(__name__)


def loaded_sessions(connection: psycopg.Connection) -> set[tuple[int, int, str]]:
    """Every session in the warehouse, as (season, round, session_code)."""
    with connection.cursor() as cursor:
        cursor.execute("SELECT season, round, session_code FROM dim_session")
        loaded = {(int(season), int(round_number), str(code))
                  for season, round_number, code in cursor.fetchall()}
    logger.info("warehouse_loaded_sessions count=%d", len(loaded))
    return loaded


def session_row(connection: psycopg.Connection, season: int, round_number: int,
                session_code: str) -> dict | None:
    """One session's warehouse identity, or None if it is not loaded.

    Used by the pipeline DAG's final task to report what it wrote, and by the
    tests to confirm a run landed.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT session_id, event_name, snapshot_date, loaded_at "
            "FROM dim_session WHERE season = %s AND round = %s AND session_code = %s",
            (season, round_number, session_code),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return {"session_id": int(row[0]), "event_name": row[1],
            "snapshot_date": str(row[2]), "loaded_at": row[3].isoformat()}
