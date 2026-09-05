"""F005 against the real database: migrations, loads, the gate, and the acceptance table.

Opt-in, like the other Docker tests: `pytest -m docker`. Everything happens in
a scratch database created for the run and dropped afterwards, so the session
loaded for F007 is never disturbed.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pandas as pd
import psycopg
import pytest

from src.config import DATA_ROOT, INTERIM_ROOT, PROCESSED_ROOT
from src.metrics.session import compute_metrics
from src.quality.engine import QualityGateError
from src.validate.session import validate_session
from src.warehouse.connection import Settings, SettingsError, connect
from src.warehouse.load import load_session
from src.warehouse.migrations import MIGRATIONS_DIR, migrate, pending
from tests import synthetic_session as syn

pytestmark = pytest.mark.docker

SCRATCH_DB = "f1_microsector_f005_test"
SUZUKA = (
    DATA_ROOT / "raw/fastf1/2026-09-05/2024_Japanese-Grand-Prix_Q",
    INTERIM_ROOT / "aligned/2024_Japanese-Grand-Prix_Q_projection",
    INTERIM_ROOT / "grid/2024_Japanese-Grand-Prix_Q_projection",
    INTERIM_ROOT / "microsectors/2024_Japanese-Grand-Prix_Q_projection",
    PROCESSED_ROOT / "2024_Japanese-Grand-Prix_Q_projection",
)
SUZUKA_PRESENT = all((root / name).exists() for root, name in zip(SUZUKA, (
    "laps.parquet", "telemetry_aligned.parquet", "grid.parquet", "microsectors.parquet", "corner_metrics.parquet")))
REFERENCE_ROOT = INTERIM_ROOT / "reference" / "2024"


@pytest.fixture(scope="module")
def settings() -> Settings:
    if shutil.which("docker") is None:
        pytest.skip("docker is not on PATH")
    try:
        base = Settings.from_env()
    except SettingsError as exc:
        pytest.skip(str(exc))
    try:
        with connect(base.with_database("postgres"), autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}"')
            cur.execute(f'CREATE DATABASE "{SCRATCH_DB}"')
    except SettingsError as exc:
        pytest.skip(f"the stack is not up: {exc}")
    yield base.with_database(SCRATCH_DB)
    with connect(base.with_database("postgres"), autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)')


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("f005")
    roots = syn.write_full_session(root)
    processed = root / "processed" / "synthetic"
    compute_metrics(roots["grid_root"], roots["microsector_root"], roots["snapshot_root"], roots["aligned_root"],
                    out_root=processed)
    validate_session(roots["snapshot_root"], roots["aligned_root"], roots["grid_root"], processed,
                     out_root=processed, min_laps=2)
    return {**roots, "processed_root": processed}


def load_synthetic(roots: dict[str, Path], connection, **kwargs):
    return load_session(roots["snapshot_root"], roots["aligned_root"], roots["grid_root"],
                        roots["microsector_root"], roots["processed_root"], connection, **kwargs)


def count(cursor, table: str) -> int:
    cursor.execute(f"SELECT count(*) FROM {table}")
    return int(cursor.fetchone()[0])


class TestMigrations:
    def test_criterion_1_apply_then_nothing_then_refuse_a_rewrite(self, settings: Settings, tmp_path: Path) -> None:
        with connect(settings) as conn:
            applied = migrate(conn)
            conn.commit()
            assert [m.name for m in applied] == ["star_schema"]
            assert migrate(conn) == [], "a second run applies nothing"
            assert pending(conn) == []

            rewritten = tmp_path / "migrations"
            rewritten.mkdir()
            body = (MIGRATIONS_DIR / "V001__star_schema.sql").read_text(encoding="utf-8")
            (rewritten / "V001__star_schema.sql").write_text(body + "\n-- edited after the fact\n", encoding="utf-8")
            from src.warehouse.migrations import MigrationError

            with pytest.raises(MigrationError, match="changed after it was applied"):
                pending(conn, rewritten)

    def test_the_read_only_role_can_read_and_cannot_write(self, settings: Settings) -> None:
        """Criterion 6: F001's default privileges reach tables created later."""
        env = {line.split("=", 1)[0].strip(): line.split("=", 1)[1].strip()
               for line in (Path("servicios/.env")).read_text(encoding="utf-8").splitlines()
               if "=" in line and not line.strip().startswith("#")}
        with connect(settings) as conn, conn.cursor() as cur:
            cur.execute(f'GRANT CONNECT ON DATABASE "{SCRATCH_DB}" TO {env["POSTGRES_READONLY_USER"]}')
            cur.execute(f'GRANT USAGE ON SCHEMA public TO {env["POSTGRES_READONLY_USER"]}')
            cur.execute(f'GRANT SELECT ON ALL TABLES IN SCHEMA public TO {env["POSTGRES_READONLY_USER"]}')
            conn.commit()

        reader = Settings(host=settings.host, port=settings.port, database=SCRATCH_DB,
                          user=env["POSTGRES_READONLY_USER"], password=env["POSTGRES_READONLY_PASSWORD"])
        with connect(reader) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM dim_session")
            assert cur.fetchone()[0] >= 0
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                cur.execute("INSERT INTO dim_session (season, round, session_code, session_name, event_name) "
                            "VALUES (1999, 1, 'Q', 'x', 'y')")


class TestLoad:
    def test_the_designed_session_loads_completely(self, settings: Settings, synthetic) -> None:
        with connect(settings) as conn:
            migrate(conn)
            conn.commit()
            result = load_synthetic(synthetic, conn)
            conn.commit()
            with conn.cursor() as cur:
                assert count(cur, "dim_lap") == result.rows["dim_lap"] == 4
                assert count(cur, "fact_telemetry_grid") == result.rows["fact_telemetry_grid"]
                assert count(cur, "fact_microsector") == result.rows["fact_microsector"]
                assert count(cur, "fact_corner_metric") == result.rows["fact_corner_metric"]
                assert count(cur, "load_audit") == 1

    def test_criterion_4_a_reload_changes_nothing_but_the_audit(self, settings: Settings, synthetic) -> None:
        with connect(settings) as conn:
            first = load_synthetic(synthetic, conn)
            conn.commit()
            with conn.cursor() as cur:
                before = {t: count(cur, t) for t in ("dim_lap", "fact_telemetry_grid", "fact_microsector",
                                                     "fact_corner_metric")}
                audits_before = count(cur, "load_audit")
            second = load_synthetic(synthetic, conn)
            conn.commit()
            with conn.cursor() as cur:
                after = {t: count(cur, t) for t in before}
                assert after == before
                assert count(cur, "load_audit") == audits_before + 1
            assert second.session_id == first.session_id

    def test_criterion_5_a_failing_gate_writes_nothing(self, settings: Settings, synthetic, tmp_path: Path) -> None:
        damaged_grid = tmp_path / "grid_damaged"
        damaged_grid.mkdir(parents=True, exist_ok=True)
        grid = pd.read_parquet(synthetic["grid_root"] / "grid.parquet")
        grid.loc[0, "speed"] = None  # a null in a critical column
        grid.to_parquet(damaged_grid / "grid.parquet", index=False)

        with connect(settings) as conn:
            with conn.cursor() as cur:
                before = {t: count(cur, t) for t in ("dim_session", "dim_lap", "fact_telemetry_grid", "load_audit")}
            with pytest.raises(QualityGateError, match="grid"):
                load_session(synthetic["snapshot_root"], synthetic["aligned_root"], damaged_grid,
                             synthetic["microsector_root"], synthetic["processed_root"], conn)
            conn.rollback()
            with conn.cursor() as cur:
                assert {t: count(cur, t) for t in before} == before

    def test_a_failure_inside_the_transaction_rolls_the_session_back(
        self, settings: Settings, synthetic, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session is completely in the warehouse or not at all.

        The failure is injected after the dimensions have been written, which
        is the case that would leave a half-loaded session behind.
        """
        import src.warehouse.load as module

        real_copy = module.copy_frame
        calls: list[str] = []

        def failing_copy(cursor, table, frame):
            calls.append(table)
            if table == "fact_telemetry_grid":
                raise RuntimeError("disk fell over mid-COPY")
            return real_copy(cursor, table, frame)

        monkeypatch.setattr(module, "copy_frame", failing_copy)
        with connect(settings) as conn:
            with conn.cursor() as cur:
                before = {t: count(cur, t) for t in ("dim_session", "dim_lap", "dim_microsector", "load_audit")}
            with pytest.raises(RuntimeError, match="mid-COPY"):
                load_synthetic(synthetic, conn)
            conn.rollback()
            assert "dim_microsector" in calls, "the failure came after dimensions were written"
            with conn.cursor() as cur:
                assert {t: count(cur, t) for t in before} == before


@pytest.fixture(scope="module")
def loaded(settings: Settings):
    """The real Suzuka session, loaded once into the scratch database."""
    if not SUZUKA_PRESENT:
        pytest.skip("Suzuka artefacts not present under data/")
    with connect(settings) as conn:
        migrate(conn)
        conn.commit()
        started = time.perf_counter()
        result = load_session(*SUZUKA, conn, reference_root=REFERENCE_ROOT if REFERENCE_ROOT.exists() else None)
        conn.commit()
        return result, time.perf_counter() - started


@pytest.mark.skipif(not SUZUKA_PRESENT, reason="Suzuka artefacts not present under data/")
class TestSuzukaAcceptance:
    """The spec's table, on the real session, in the scratch database."""

    def test_criterion_2_every_row_arrives(self, loaded, settings: Settings) -> None:
        result, _ = loaded
        assert result.rows == {"dim_lap": 74, "fact_telemetry_grid": 42198,
                               "fact_microsector": 6808, "fact_corner_metric": 592}

    def test_criterion_3_the_load_is_quick(self, loaded) -> None:
        _, elapsed = loaded
        assert elapsed <= 30.0

    def test_criterion_7_a_session_query_prunes_to_one_leaf(self, loaded, settings: Settings) -> None:
        with connect(settings) as conn, conn.cursor() as cur:
            cur.execute("EXPLAIN (FORMAT JSON) SELECT count(*) FROM fact_telemetry_grid "
                        "WHERE season=2024 AND round=4 AND session_code='Q'")
            plan = cur.fetchone()[0][0]["Plan"]

            def relations(node, found=None):
                found = [] if found is None else found
                if "Relation Name" in node:
                    found.append(node["Relation Name"])
                for child in node.get("Plans", []):
                    relations(child, found)
                return found

            assert relations(plan) == ["fact_telemetry_grid_2024_04_q"]

    def test_criterion_8_the_dashboard_queries_are_fast(self, loaded, settings: Settings) -> None:
        with connect(settings) as conn, conn.cursor() as cur:
            cur.execute("SELECT lap_id FROM dim_lap WHERE season = 2024 AND round = 4 AND session_code = 'Q' "
                        "ORDER BY lap_time_s LIMIT 2")
            ids = [row[0] for row in cur.fetchall()]
            assert len(ids) == 2

            started = time.perf_counter()
            cur.execute("SELECT grid_index, distance_m, speed, delta_t_s FROM fact_telemetry_grid "
                        "WHERE season=2024 AND round=4 AND session_code='Q' AND lap_id = ANY(%s) "
                        "ORDER BY lap_id, grid_index", (ids,))
            rows = cur.fetchall()
            overlay_ms = (time.perf_counter() - started) * 1000
            assert len(rows) > 1000 and overlay_ms <= 50

            started = time.perf_counter()
            cur.execute("SELECT d.corners, avg(f.time_s) FROM fact_microsector f "
                        "JOIN dim_lap l ON l.lap_id = f.lap_id "
                        "JOIN dim_microsector d ON d.session_id = l.session_id AND d.grain = f.grain "
                        "AND d.microsector_id = f.microsector_id "
                        "WHERE d.grain = 'corner_phase' AND d.phase = 'apex' "
                        "AND l.season = 2024 AND l.round = 4 GROUP BY d.corners")
            groups = cur.fetchall()
            aggregate_ms = (time.perf_counter() - started) * 1000
            assert len(groups) >= 8 and aggregate_ms <= 50

    def test_criterion_9_fact_columns_are_the_declared_types(self, loaded, settings: Settings) -> None:
        with connect(settings) as conn, conn.cursor() as cur:
            cur.execute("SELECT data_type FROM information_schema.columns WHERE table_schema='public' "
                        "AND table_name IN ('fact_telemetry_grid','fact_microsector','fact_corner_metric')")
            types = {row[0] for row in cur.fetchall()}
            assert "double precision" not in types
            assert types <= {"real", "smallint", "integer", "text", "boolean"}

    def test_criterion_10_no_view_or_function_exists(self, loaded, settings: Settings) -> None:
        with connect(settings) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM information_schema.views WHERE table_schema='public'")
            assert cur.fetchone()[0] == 0
            cur.execute("SELECT count(*) FROM information_schema.routines WHERE routine_schema='public'")
            assert cur.fetchone()[0] == 0

    def test_criterion_11_the_session_fits_its_budget(self, loaded, settings: Settings) -> None:
        with connect(settings) as conn, conn.cursor() as cur:
            cur.execute("SELECT sum(pg_total_relation_size(c.oid)) FROM pg_class c "
                        "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname='public' "
                        "AND c.relkind IN ('r','p','i')")
            total = int(cur.fetchone()[0])
            assert total <= 20 * 1024 * 1024, f"{total / 1024 / 1024:.1f} MiB for one session"

    @pytest.mark.skipif(not REFERENCE_ROOT.exists(), reason="2024 reference data not ingested")
    def test_criterion_12_constructors_come_from_the_round_entry(self, loaded, settings: Settings) -> None:
        with connect(settings) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM dim_lap WHERE season = 2024 AND round = 4 AND constructor_id IS NULL")
            assert cur.fetchone()[0] == 0
            cur.execute("SELECT DISTINCT code, team_alias, constructor_id FROM dim_lap "
                        "WHERE season = 2024 AND round = 4 AND code IN ('GAS','BOT','RIC','PER') ORDER BY code")
            resolved = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
            assert resolved["GAS"] == ("Alpine", "alpine")
            assert resolved["BOT"] == ("Kick Sauber", "sauber")
            assert resolved["RIC"] == ("RB", "rb")
            assert resolved["PER"] == ("Red Bull Racing", "red_bull")

    def test_the_audit_records_the_load(self, loaded, settings: Settings) -> None:
        result, _ = loaded
        with connect(settings) as conn, conn.cursor() as cur:
            cur.execute("SELECT quality_ok, quality_warnings, rows_grid, contract_version FROM load_audit "
                        "WHERE load_id = %s", (result.load_id,))
            ok, warnings, rows_grid, contract = cur.fetchone()
            assert ok is True and rows_grid == 42198
            assert warnings == 1, "the seven brakeless corners, as F004 measured"
            assert contract == result.quality.contract_version
