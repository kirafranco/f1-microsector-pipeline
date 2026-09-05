"""Versioned SQL migrations (global CLAUDE.md 3.3: never a manual ALTER).

Numbered files applied in order, each in its own transaction, each recorded
with a checksum. A file that has already been applied and has since changed is
refused rather than silently ignored: the database and the repository must
tell the same story.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import psycopg

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = PROJECT_ROOT / "servicios" / "postgres" / "migrations"

#: V001__star_schema.sql
FILENAME = re.compile(r"^V(?P<version>\d+)__(?P<name>[a-z0-9_]+)\.sql$")

TRACKING_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     integer     PRIMARY KEY,
    name        text        NOT NULL,
    checksum    text        NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
)
"""


class MigrationError(RuntimeError):
    """The migration history on disk and in the database disagree."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]


@dataclass(frozen=True)
class AppliedMigration:
    version: int
    name: str
    checksum: str


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Every migration file, in version order."""
    if not directory.exists():
        raise MigrationError(f"no migrations directory at {directory}")
    found: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        match = FILENAME.match(path.name)
        if not match:
            raise MigrationError(f"{path.name}: expected V<version>__<name>.sql")
        found.append(Migration(int(match.group("version")), match.group("name"), path))
    versions = [m.version for m in found]
    duplicates = {v for v in versions if versions.count(v) > 1}
    if duplicates:
        raise MigrationError(f"duplicate migration versions: {sorted(duplicates)}")
    return sorted(found, key=lambda m: m.version)


def applied(connection: psycopg.Connection) -> dict[int, AppliedMigration]:
    """What the database says it already has."""
    with connection.cursor() as cursor:
        cursor.execute(TRACKING_TABLE)
        cursor.execute("SELECT version, name, checksum FROM schema_migrations ORDER BY version")
        return {row[0]: AppliedMigration(*row) for row in cursor.fetchall()}


def pending(connection: psycopg.Connection, directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Migrations not yet applied, after checking that history has not been rewritten."""
    history = applied(connection)
    outstanding: list[Migration] = []
    for migration in discover(directory):
        known = history.get(migration.version)
        if known is None:
            outstanding.append(migration)
        elif known.checksum != migration.checksum:
            raise MigrationError(
                f"V{migration.version:03d}__{migration.name}.sql changed after it was applied "
                f"(recorded {known.checksum}, on disk {migration.checksum}); "
                "write a new migration instead of editing an applied one"
            )
    return outstanding


def migrate(connection: psycopg.Connection, directory: Path = MIGRATIONS_DIR) -> list[AppliedMigration]:
    """Apply what is outstanding. Idempotent: a second run applies nothing."""
    outstanding = pending(connection, directory)
    done: list[AppliedMigration] = []

    for migration in outstanding:
        checksum = migration.checksum
        sql = migration.path.read_text(encoding="utf-8")
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute(sql)
                cursor.execute(
                    "INSERT INTO schema_migrations (version, name, checksum) VALUES (%s, %s, %s)",
                    (migration.version, migration.name, checksum),
                )
        logger.info("migration_applied version=%d name=%s checksum=%s", migration.version, migration.name, checksum)
        done.append(AppliedMigration(migration.version, migration.name, checksum))

    if not done:
        logger.info("migrations_up_to_date count=%d", len(discover(directory)))
    return done
