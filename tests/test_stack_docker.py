"""F001 integration: the core profile on the real Docker daemon.

Opt-in, like the network tests: `pytest -m docker`. Brings the stack up from
whatever `data/` state exists, asserts the runtime criteria, and leaves it
running -- teardown is the caller's, so a failed run can be inspected.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import pytest

from src.config import PROJECT_ROOT
from tests.stack_env import SERVICIOS, env_values, grafana_api

pytestmark = pytest.mark.docker

COMPOSE = ["docker", "compose", "-f", str(SERVICIOS / "docker-compose.yml"), "--profile", "core"]
STARTUP_TIMEOUT_S = 180


def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=600, **kwargs)


def container(service: str, env: dict[str, str]) -> str:
    return f"{env['COMPOSE_PROJECT_NAME']}-{service}"


def health(service: str, env: dict[str, str]) -> str:
    result = _run(["docker", "inspect", "-f", "{{.State.Health.Status}}", container(service, env)])
    return result.stdout.strip()


def wait_healthy(env: dict[str, str], started: float, timeout_s: int = STARTUP_TIMEOUT_S) -> dict[str, float]:
    """Seconds from ``started`` until each service reports healthy.

    ``started`` is taken *before* ``up -d``: Compose blocks that command until
    postgres is healthy so it can start grafana, so timing after it returns
    would credit postgres with a start-up of a few milliseconds.
    """
    elapsed: dict[str, float] = {}
    while time.monotonic() - started < timeout_s:
        for service in ("postgres", "grafana"):
            if service not in elapsed and health(service, env) == "healthy":
                elapsed[service] = time.monotonic() - started
        if len(elapsed) == 2:
            return elapsed
        time.sleep(2)
    raise AssertionError(f"not healthy within {timeout_s}s: {elapsed}")


def psql(env: dict[str, str], user_key: str, password_key: str, sql: str) -> subprocess.CompletedProcess:
    """Run SQL inside the container as the given role.

    The password travels through the docker client's environment (`-e VAR`
    with no value forwards it), never through the argument list, so it cannot
    surface in a process listing or in a CompletedProcess repr on failure.
    """
    return _run(
        ["docker", "exec", "-e", "PGPASSWORD", container("postgres", env),
         "psql", "-U", env[user_key], "-d", env["POSTGRES_DB"], "-tAc", sql],
        env={**os.environ, "PGPASSWORD": env[password_key]},
    )


@pytest.fixture(scope="module")
def env() -> dict[str, str]:
    if shutil.which("docker") is None:
        pytest.skip("docker is not on PATH")
    return env_values()


@pytest.fixture(scope="module")
def stack(env: dict[str, str]) -> dict[str, float]:
    started = time.monotonic()
    result = _run([*COMPOSE, "up", "-d"])
    assert result.returncode == 0, result.stderr
    return wait_healthy(env, started)


class TestStartup:
    def test_criterion_2_both_services_reach_healthy(self, stack: dict[str, float]) -> None:
        assert stack["postgres"] <= 60.0, f"postgres took {stack['postgres']:.0f}s"
        assert stack["grafana"] <= 90.0, f"grafana took {stack['grafana']:.0f}s"

    def test_criterion_7_resource_limits_are_applied(self, env: dict[str, str], stack) -> None:
        expected = {"postgres": (2 * 1024**3, 2_000_000_000), "grafana": (1024**3, 1_000_000_000)}
        for service, (memory, nano_cpus) in expected.items():
            result = _run(["docker", "inspect", "-f", "{{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}",
                           container(service, env)])
            actual_memory, actual_cpus = (int(x) for x in result.stdout.split())
            assert actual_memory == memory, service
            assert actual_cpus == nano_cpus, service

    def test_criterion_9_data_directory_permissions(self, env: dict[str, str], stack) -> None:
        """Postgres refuses a cluster directory it cannot lock down to 0700."""
        result = _run(["docker", "exec", container("postgres", env), "sh", "-c",
                       'stat -c "%a %U" "$PGDATA"'])
        mode, owner = result.stdout.split()
        assert mode == "700" and owner == "postgres"

    def test_containers_are_on_the_project_network_only(self, env: dict[str, str], stack) -> None:
        for service in ("postgres", "grafana"):
            result = _run(["docker", "inspect", "-f", "{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}",
                           container(service, env)])
            assert result.stdout.split() == [f"{env['COMPOSE_PROJECT_NAME']}-net"]


class TestReadOnlyRole:
    def test_criterion_3_select_is_allowed(self, env: dict[str, str], stack) -> None:
        result = psql(env, "POSTGRES_READONLY_USER", "POSTGRES_READONLY_PASSWORD", "select 1")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "1"

    def test_criterion_3_writes_are_refused(self, env: dict[str, str], stack) -> None:
        for statement in ("create table f001_forbidden(x int)", "create schema f001_forbidden"):
            result = psql(env, "POSTGRES_READONLY_USER", "POSTGRES_READONLY_PASSWORD", statement)
            assert result.returncode != 0, f"read-only role executed: {statement}"
            assert "permission denied" in result.stderr.lower()

    def test_new_admin_tables_are_readable_without_a_further_grant(self, env: dict[str, str], stack) -> None:
        """ALTER DEFAULT PRIVILEGES: F005's migrations must not break the dashboard."""
        created = psql(env, "POSTGRES_USER", "POSTGRES_PASSWORD",
                       "create table if not exists f001_probe(x int); insert into f001_probe values (7)")
        assert created.returncode == 0, created.stderr
        try:
            read = psql(env, "POSTGRES_READONLY_USER", "POSTGRES_READONLY_PASSWORD", "select x from f001_probe")
            assert read.returncode == 0, read.stderr
            assert read.stdout.strip() == "7"
            denied = psql(env, "POSTGRES_READONLY_USER", "POSTGRES_READONLY_PASSWORD",
                          "insert into f001_probe values (8)")
            assert denied.returncode != 0
        finally:
            psql(env, "POSTGRES_USER", "POSTGRES_PASSWORD", "drop table if exists f001_probe")


class TestGrafana:
    def test_criterion_4_datasource_is_provisioned(self, env: dict[str, str], stack) -> None:
        datasource = grafana_api(env, "/api/datasources/uid/f1-postgres")
        assert datasource["type"] == "grafana-postgresql-datasource"
        assert datasource["isDefault"] is True
        assert datasource["readOnly"] is True, "a provisioned datasource must not be UI-editable"

    def test_criterion_4_datasource_can_query_postgres(self, env: dict[str, str], stack) -> None:
        """Proves the read-only credentials interpolated and the service name resolves."""
        result = grafana_api(env, "/api/datasources/uid/f1-postgres/health")
        assert result.get("status") == "OK", result

    def test_dashboard_provider_is_loaded(self, env: dict[str, str], stack) -> None:
        health_payload = grafana_api(env, "/api/health")
        assert health_payload["database"] == "ok"
        assert health_payload["version"].startswith("12.4")


class TestPersistenceAndIdempotency:
    def test_criterion_6_second_up_recreates_nothing(self, env: dict[str, str], stack) -> None:
        before = _run([*COMPOSE, "ps", "-q"]).stdout.split()
        result = _run([*COMPOSE, "up", "-d"])
        assert result.returncode == 0, result.stderr
        assert _run([*COMPOSE, "ps", "-q"]).stdout.split() == before

    def test_criterion_5_data_survives_down_and_up(self, env: dict[str, str], stack) -> None:
        created = psql(env, "POSTGRES_USER", "POSTGRES_PASSWORD",
                       "create table if not exists f001_persist(x int); "
                       "truncate f001_persist; insert into f001_persist values (1),(2),(3)")
        assert created.returncode == 0, created.stderr

        assert _run([*COMPOSE, "down"]).returncode == 0
        restarted = time.monotonic()
        assert _run([*COMPOSE, "up", "-d"]).returncode == 0
        wait_healthy(env, restarted)
        try:
            result = psql(env, "POSTGRES_USER", "POSTGRES_PASSWORD", "select count(*) from f001_persist")
            assert result.stdout.strip() == "3", result.stderr
        finally:
            psql(env, "POSTGRES_USER", "POSTGRES_PASSWORD", "drop table if exists f001_persist")


class TestRepositoryHygiene:
    def test_criterion_8_nothing_the_stack_wrote_is_tracked(self, stack) -> None:
        result = _run(["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT))
        dirty = [line for line in result.stdout.splitlines() if "/data/" in line or line.endswith(".env")]
        assert not dirty, f"stack output is visible to git: {dirty}"

    def test_data_directories_exist_on_the_host(self, stack) -> None:
        for name in ("postgres", "grafana"):
            path = PROJECT_ROOT / "data" / name
            assert path.is_dir() and any(path.iterdir()), f"data/{name} is empty"
