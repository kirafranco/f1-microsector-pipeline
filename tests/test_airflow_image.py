"""F006: the project's first Dockerfile, its entrypoint, and the DAG files.

Static rules only -- no Docker daemon. The build itself is checked by the
`-m docker` suite; this is the half that must hold on any machine, including
the one where someone edits the Dockerfile without a daemon running.
"""

from __future__ import annotations

import re

import pytest

from src.config import PROJECT_ROOT

AIRFLOW_DIR = PROJECT_ROOT / "servicios" / "airflow"
DOCKERFILE = AIRFLOW_DIR / "Dockerfile"
ENTRYPOINT = AIRFLOW_DIR / "entrypoint.sh"
DAGS_DIR = AIRFLOW_DIR / "dags"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def entrypoint() -> str:
    return ENTRYPOINT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dags() -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8")
            for path in sorted(DAGS_DIR.glob("*.py"))}


def _commands(script: str) -> str:
    """The script without its comments, for rules about what it does."""
    return "\n".join(line.split("#", 1)[0] for line in script.splitlines())


@pytest.fixture(scope="module")
def dockerignore() -> list[str]:
    return [line.strip() for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


class TestDockerfile:
    def test_the_base_is_an_official_image_pinned_exactly(self, dockerfile: str) -> None:
        (base,) = re.findall(r"^FROM (\S+)", dockerfile, re.MULTILINE)
        assert base.startswith("apache/airflow:"), "Apache is the publisher; no third-party rebuild"
        tag = base.split(":", 1)[1]
        assert tag != "latest"
        assert re.fullmatch(r"\d+\.\d+\.\d+-python3\.\d+", tag), f"{tag!r} is not an exact version"

    def test_dependencies_are_installed_before_the_code_is_copied(self, dockerfile: str) -> None:
        """Otherwise editing one module reinstalls pandas on every build."""
        requirements = dockerfile.index("requirements.txt")
        source = dockerfile.index("COPY --chown=airflow:root src/")
        assert requirements < source

    def test_no_secret_is_baked_into_a_layer(self, dockerfile: str) -> None:
        for line in dockerfile.splitlines():
            if line.startswith(("ENV ", "ARG ")):
                assert not re.search(r"(PASSWORD|SECRET|TOKEN|KEY)\s*=", line, re.IGNORECASE), line

    def test_it_does_not_run_as_root(self, dockerfile: str) -> None:
        users = re.findall(r"^USER (\S+)", dockerfile, re.MULTILINE)
        assert users, "no USER directive"
        assert users[-1] != "root"

    def test_the_airflow_user_exception_is_explained_in_place(self, dockerfile: str) -> None:
        """Global 2.1 asks for USER 1000:1000. This image uses the official
        airflow user instead, and a reader deserves the reason next to it."""
        assert "USER airflow" in dockerfile
        assert "50000" in dockerfile and "global 2.1" in dockerfile

    def test_logs_are_not_buffered(self, dockerfile: str) -> None:
        assert "PYTHONUNBUFFERED=1" in dockerfile

    def test_the_pipeline_is_importable_from_the_image_alone(self, dockerfile: str) -> None:
        """The bind mount shadows this at runtime, but the image must still run
        standalone (global 2.1)."""
        assert "COPY --chown=airflow:root src/ /opt/airflow/project/src/" in dockerfile
        assert "PYTHONPATH=/opt/airflow/project" in dockerfile


class TestDockerignore:
    def test_it_exists_because_a_build_context_now_does(self) -> None:
        assert DOCKERIGNORE.exists(), "a build context without a .dockerignore is a violation"

    @pytest.mark.parametrize("pattern", ["data/", "**/.env", "__pycache__/", "notebooks/",
                                         "*.parquet", "*.csv", "harness/", ".git/"])
    def test_it_excludes_what_must_never_reach_a_layer(self, dockerignore: list[str], pattern: str) -> None:
        assert pattern in dockerignore, f"{pattern} is not excluded from the build context"

    def test_the_example_env_is_not_excluded_with_the_real_one(self, dockerignore: list[str]) -> None:
        assert "!**/.env.example" in dockerignore


class TestEntrypoint:
    def test_it_fails_on_the_first_error(self, entrypoint: str) -> None:
        assert "set -euo pipefail" in entrypoint

    def test_every_credential_is_required_from_the_environment(self, entrypoint: str) -> None:
        for name in ("POSTGRES_USER", "POSTGRES_PASSWORD", "AIRFLOW_ADMIN_PASSWORD", "AIRFLOW_DB"):
            assert f'"${{{name}:?' in entrypoint, f"{name} is not required up front"

    def test_the_database_name_is_quoted_by_postgres_not_by_the_shell(self, entrypoint: str) -> None:
        """The same rule the read-only role script follows: format() with %I."""
        assert "format('CREATE DATABASE %I'" in entrypoint
        assert "\\gexec" in entrypoint

    def test_creating_the_metadata_database_is_idempotent(self, entrypoint: str) -> None:
        assert "WHERE NOT EXISTS (SELECT 1 FROM pg_database" in entrypoint

    def test_the_admin_password_is_written_before_airflow_can_generate_one(self, entrypoint: str) -> None:
        """SimpleAuthManager prints a generated password to stdout for any
        configured user missing from the file. Writing it first prevents that."""
        assert "passwords" in entrypoint
        assert entrypoint.index("AIRFLOW_ADMIN_PASSWORD") < entrypoint.index("exec airflow")

    def test_no_credential_is_written_to_a_config_file(self, entrypoint: str) -> None:
        """The metadata URL carries the password, so it never reaches a file.

        Checked over the commands only: the script explains *why* it avoids
        airflow.cfg, and a check over the prose would fail on its own rationale.
        """
        assert "airflow.cfg" not in _commands(entrypoint)
        assert "airflow config set" not in _commands(entrypoint)

    def test_the_metadata_connection_comes_from_the_container_environment(self, entrypoint: str) -> None:
        """Not exported here: a variable this process exports is invisible to
        `docker compose exec`, and the CLI would then talk to a default SQLite
        database and report the schema as unmigrated. Measured 2026-09-05."""
        assert "export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN" not in _commands(entrypoint)
        assert '"${AIRFLOW__DATABASE__SQL_ALCHEMY_CONN:?' in entrypoint

    def test_it_hands_over_to_airflow_as_pid_one(self, entrypoint: str) -> None:
        """`exec` so signals reach Airflow and the container stops cleanly."""
        assert entrypoint.rstrip().endswith('exec airflow "$@"')

    def test_it_has_unix_line_endings(self) -> None:
        """A CRLF here is the bug .gitattributes exists to prevent."""
        assert b"\r\n" not in ENTRYPOINT.read_bytes()


class TestDagFiles:
    def test_both_dags_are_present(self, dags: dict[str, str]) -> None:
        assert set(dags) == {"f1_session_pipeline.py", "f1_calendar_dispatch.py"}

    def test_they_hold_wiring_and_not_decisions(self, dags: dict[str, str]) -> None:
        """Everything with judgement in it lives in src.orchestration, which is
        testable without Airflow installed."""
        for name, body in dags.items():
            assert "from src.orchestration" in body, name

    def test_no_dag_reaches_into_a_stage_directly(self, dags: dict[str, str]) -> None:
        """A DAG calling align_session itself would bypass the path convention."""
        for name, body in dags.items():
            for forbidden in ("from src.align", "from src.grid", "from src.segment",
                              "from src.metrics", "from src.validate", "from src.quality"):
                assert forbidden not in body, f"{name} imports {forbidden}"

    def test_the_pipeline_is_triggered_not_scheduled(self, dags: dict[str, str]) -> None:
        assert "schedule=None" in dags["f1_session_pipeline.py"]

    def test_one_session_at_a_time(self, dags: dict[str, str]) -> None:
        """A 4 GB container and ~600k grid rows per session (D9)."""
        assert "max_active_runs=MAX_ACTIVE_RUNS" in dags["f1_session_pipeline.py"]
        assert "MAX_ACTIVE_RUNS = 1" in dags["f1_session_pipeline.py"]

    def test_the_sensor_frees_its_worker_between_pokes(self, dags: dict[str, str]) -> None:
        """The wait is hours and the executor is local; a held slot would block
        the whole scheduler."""
        assert 'mode="reschedule"' in dags["f1_session_pipeline.py"]

    def test_the_dispatcher_runs_hourly_and_does_not_backfill(self, dags: dict[str, str]) -> None:
        body = dags["f1_calendar_dispatch.py"]
        assert 'schedule="@hourly"' in body
        assert "catchup=False" in body

    def test_a_live_run_is_never_triggered_twice(self, dags: dict[str, str]) -> None:
        """A stable run id per session, and no reset of one already going."""
        body = dags["f1_calendar_dispatch.py"]
        assert "reset_dag_run=False" in body
        assert "skip_when_already_exists=True" in body
        assert "trigger_run_id" in body

    def test_the_quality_gate_sits_between_validate_and_load(self, dags: dict[str, str]) -> None:
        body = dags["f1_session_pipeline.py"]
        assert body.index("checked = quality(") < body.index("loaded = load(")
        assert "load(checked" in body, "load must depend on the gate, not run beside it"
