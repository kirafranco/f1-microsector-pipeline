"""F005 migrations: ordered, checksummed, and refusing a rewritten history."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.warehouse.migrations import MIGRATIONS_DIR, Migration, MigrationError, discover


def write(directory: Path, name: str, body: str = "SELECT 1;") -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


class TestDiscovery:
    def test_files_come_back_in_version_order(self, tmp_path: Path) -> None:
        write(tmp_path, "V010__later.sql")
        write(tmp_path, "V002__second.sql")
        write(tmp_path, "V001__first.sql")
        found = discover(tmp_path)
        assert [m.version for m in found] == [1, 2, 10], "10 sorts after 2, not between 1 and 2"
        assert [m.name for m in found] == ["first", "second", "later"]

    def test_a_badly_named_file_is_refused(self, tmp_path: Path) -> None:
        write(tmp_path, "V001__first.sql")
        write(tmp_path, "star_schema.sql")
        with pytest.raises(MigrationError, match="expected V"):
            discover(tmp_path)

    def test_duplicate_versions_are_refused(self, tmp_path: Path) -> None:
        write(tmp_path, "V001__first.sql")
        write(tmp_path, "V001__also_first.sql")
        with pytest.raises(MigrationError, match="duplicate"):
            discover(tmp_path)

    def test_a_missing_directory_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(MigrationError, match="no migrations directory"):
            discover(tmp_path / "absent")

    def test_an_empty_directory_is_allowed(self, tmp_path: Path) -> None:
        assert discover(tmp_path) == []


class TestChecksum:
    def test_it_follows_the_content(self, tmp_path: Path) -> None:
        path = write(tmp_path, "V001__first.sql", "SELECT 1;")
        migration = Migration(1, "first", path)
        original = migration.checksum
        assert original == Migration(1, "first", path).checksum

        path.write_text("SELECT 2;", encoding="utf-8")
        assert Migration(1, "first", path).checksum != original

    def test_whitespace_counts(self, tmp_path: Path) -> None:
        """A checksum that ignored formatting would let an edit slip through."""
        path = write(tmp_path, "V001__first.sql", "SELECT 1;")
        first = Migration(1, "first", path).checksum
        path.write_text("SELECT  1;", encoding="utf-8")
        assert Migration(1, "first", path).checksum != first


def ddl_only(body: str) -> str:
    """The statements, without the comments that explain them.

    The file discusses `double precision` in order to say it is not used, so a
    check over the prose would fail on its own rationale.
    """
    lines = []
    for line in body.splitlines():
        stripped = line.split("--", 1)[0]
        if stripped.strip():
            lines.append(stripped)
    return "\n".join(lines)


class TestTheProjectsMigrations:
    def test_the_repository_has_a_valid_history(self) -> None:
        found = discover(MIGRATIONS_DIR)
        assert found, "the star schema migration should exist"
        assert found[0].version == 1 and found[0].name == "star_schema"
        assert [m.version for m in found] == sorted(m.version for m in found)

    def test_the_schema_declares_what_the_spec_promised(self) -> None:
        body = (MIGRATIONS_DIR / "V001__star_schema.sql").read_text(encoding="utf-8")
        for table in ("dim_session", "dim_driver", "dim_constructor", "dim_lap", "dim_microsector",
                      "dim_corner_event", "fact_telemetry_grid", "fact_microsector", "fact_corner_metric",
                      "load_audit"):
            assert f"CREATE TABLE {table}" in body, table

    def test_facts_are_partitioned_and_typed_as_declared(self) -> None:
        body = ddl_only((MIGRATIONS_DIR / "V001__star_schema.sql").read_text(encoding="utf-8"))
        assert body.count("PARTITION BY RANGE (season, round)") == 2
        assert "double precision" not in body, "continuous telemetry is real (float4)"
        assert "PRIMARY KEY (season, round, session_code, lap_id, grid_index)" in body

    def test_no_view_or_function_is_created(self) -> None:
        """Decision D1: nothing in the schema exists for one consumer."""
        body = ddl_only((MIGRATIONS_DIR / "V001__star_schema.sql").read_text(encoding="utf-8")).upper()
        for forbidden in ("CREATE VIEW", "CREATE MATERIALIZED VIEW", "CREATE FUNCTION", "CREATE PROCEDURE"):
            assert forbidden not in body, forbidden
