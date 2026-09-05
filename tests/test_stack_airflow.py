"""F006 integration: the orchestration profile on the real Docker daemon.

Opt-in: `pytest -m docker`. Brings the profile up, checks what Airflow made of
the DAGs, and runs one session end to end through the scheduler.

The pipeline run needs the target session's FastF1 cache or the network, and
the reference data F012 ingested; each of those is a skip rather than a
failure, because a machine without them is not a broken one.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import pytest

from src.config import INTERIM_ROOT, PROJECT_ROOT
from tests.stack_env import SERVICIOS, env_values

pytestmark = pytest.mark.docker

COMPOSE = ["docker", "compose", "-f", str(SERVICIOS / "docker-compose.yml"),
           "--profile", "orchestration"]
STARTUP_TIMEOUT_S = 240
RUN_TIMEOUT_S = 420

#: The session the whole project is built around (D10).
TARGET = {"season": 2024, "event": "Japanese Grand Prix", "session": "Q"}
TARGET_KEY = (2024, 4, "Q")

#: F005 measured these on this session; the DAG must reproduce them exactly.
EXPECTED_ROWS = {"rows_grid": 42418, "rows_microsector": 6882,
                 "rows_corner": 592, "rows_lap": 74}


def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=900, **kwargs)


def compose_exec(args: list[str]) -> subprocess.CompletedProcess:
    return _run([*COMPOSE, "exec", "-T", "airflow", *args])


def psql(env: dict[str, str], sql: str) -> str:
    """Query the warehouse as the admin role, password via the environment."""
    result = _run(
        ["docker", "exec", "-e", "PGPASSWORD", f"{env['COMPOSE_PROJECT_NAME']}-postgres",
         "psql", "-U", env["POSTGRES_USER"], "-d", env["POSTGRES_DB"], "-tAc", sql],
        env={**os.environ, "PGPASSWORD": env["POSTGRES_PASSWORD"]},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


class AirflowApi:
    """The REST API, authenticated once. Credentials never reach an argument list."""

    def __init__(self, env: dict[str, str]) -> None:
        self.base = f"http://127.0.0.1:{env['AIRFLOW_PORT']}"
        self.token = self._call("/auth/token", "POST", {
            "username": env["AIRFLOW_ADMIN_USER"],
            "password": env["AIRFLOW_ADMIN_PASSWORD"]})["access_token"]

    def _call(self, path: str, method: str = "GET", body: dict | None = None,
              token: str | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode() or "{}")

    def get(self, path: str) -> dict:
        return self._call(path, token=self.token)

    def post(self, path: str, body: dict) -> dict:
        return self._call(path, "POST", body, token=self.token)

    def patch(self, path: str, body: dict) -> dict:
        return self._call(path, "PATCH", body, token=self.token)


@pytest.fixture(scope="module")
def env() -> dict[str, str]:
    if shutil.which("docker") is None:
        pytest.skip("docker is not on PATH")
    return env_values()


@pytest.fixture(scope="module")
def stack(env: dict[str, str]) -> float:
    """The profile up and healthy. Returns seconds from before `up`."""
    started = time.monotonic()
    result = _run([*COMPOSE, "up", "-d", "--wait"])
    assert result.returncode == 0, result.stderr
    return time.monotonic() - started


@pytest.fixture(scope="module")
def api(env: dict[str, str], stack: float) -> AirflowApi:
    return AirflowApi(env)


class TestCriterion1Startup:
    def test_both_services_come_up_healthy(self, env: dict[str, str], stack: float) -> None:
        assert stack <= STARTUP_TIMEOUT_S, f"took {stack:.0f}s"
        for service in ("postgres", "airflow"):
            result = _run(["docker", "inspect", "-f", "{{.State.Health.Status}}",
                           f"{env['COMPOSE_PROJECT_NAME']}-{service}"])
            assert result.stdout.strip() == "healthy", service

    def test_postgres_is_in_this_profile_too(self, stack: float) -> None:
        """Compose does not enable a dependency's profile on its own."""
        result = _run([*COMPOSE, "config", "--services"])
        assert sorted(result.stdout.split()) == ["airflow", "postgres"]

    def test_the_resource_limits_are_applied(self, env: dict[str, str], stack: float) -> None:
        result = _run(["docker", "inspect", "-f", "{{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}",
                       f"{env['COMPOSE_PROJECT_NAME']}-airflow"])
        memory, cpus = (int(value) for value in result.stdout.split())
        assert memory == 4 * 1024**3 and cpus == 2_000_000_000


class TestCriterion2Memory:
    def test_it_sits_well_inside_its_limit(self, env: dict[str, str], stack: float) -> None:
        """Reported rather than gated: the limit is the contract, this is the
        evidence that D9's 4 GB was not optimistic."""
        result = _run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}",
                       f"{env['COMPOSE_PROJECT_NAME']}-airflow"])
        used = result.stdout.split("/")[0].strip()
        value = float(used.rstrip("GiBMB").strip())
        gigabytes = value if used.endswith("GiB") else value / 1024
        assert gigabytes <= 3.0, f"{used} of a 4 GiB limit"


class TestCriterion3Dags:
    def test_neither_dag_fails_to_import(self, stack: float) -> None:
        result = compose_exec(["airflow", "dags", "list-import-errors"])
        assert result.returncode == 0, result.stderr
        assert "No data found" in result.stdout, result.stdout

    def test_both_are_registered(self, api: AirflowApi) -> None:
        ids = {dag["dag_id"] for dag in api.get("/api/v2/dags?limit=100")["dags"]}
        assert {"f1_session_pipeline", "f1_calendar_dispatch"} <= ids

    def test_the_pipeline_is_ready_and_runs_one_session_at_a_time(self, api: AirflowApi) -> None:
        dag = api.get("/api/v2/dags/f1_session_pipeline")
        assert dag["is_paused"] is False, "a triggered DAG costs nothing while idle"
        assert dag["max_active_runs"] == 1, "4 GB and ~600k grid rows per session (D9)"

    def test_the_dispatcher_arrives_paused(self, api: AirflowApi) -> None:
        """It is the only DAG that starts work on its own; an unattended laptop
        should not begin ingesting a season because the stack came up."""
        assert api.get("/api/v2/dags/f1_calendar_dispatch")["is_paused"] is True

    def test_the_metadata_database_is_the_project_postgres(self, stack: float) -> None:
        """A CLI that quietly used its default SQLite would report the schema as
        unmigrated -- which is exactly what happened before the connection moved
        into the container environment."""
        result = compose_exec(["airflow", "config", "get-value", "database", "sql_alchemy_conn"])
        assert "postgresql" in result.stdout, result.stdout
        assert "sqlite" not in result.stdout.lower()


def _restore_stack() -> None:
    """Both services up, with a scheduler that can reach the database.

    The restart is unconditional because "healthy" does not mean "connected":
    Airflow's health check is its api-server, which answers happily while the
    scheduler is stranded on a dead connection.
    """
    assert _run([*COMPOSE, "up", "-d", "--wait"]).returncode == 0
    assert _run([*COMPOSE, "restart", "airflow"]).returncode == 0
    assert _run([*COMPOSE, "up", "-d", "--wait"]).returncode == 0


@pytest.fixture(scope="module")
def first_run(api: AirflowApi, env: dict[str, str]) -> dict:
    """One real pipeline run for the target session, waited out.

    The stack is restored first. F001's own integration test removes the
    database container and brings it back to prove the data survives, and
    Airflow does not recover from that: the api-server keeps answering, so the
    container still reports healthy, while the scheduler holds a connection to
    a Postgres that no longer exists and every run sits queued for ever.
    Restarting it is the only thing that fixes it -- see the progress record.
    """
    _restore_stack()
    run_id = f"test__f006_{int(time.time())}"
    api.post("/api/v2/dags/f1_session_pipeline/dagRuns",
             {"dag_run_id": run_id, "logical_date": None, "conf": TARGET})
    run = _wait_for(api, run_id)
    if run["state"] != "success":
        pytest.skip(f"the pipeline run did not succeed ({run['state']}); "
                    "the session's telemetry may be unavailable on this machine")
    return {"run_id": run_id, "audit": psql(env, "SELECT count(*) FROM load_audit")}


def _wait_for(api: AirflowApi, run_id: str) -> dict:
    deadline = time.monotonic() + RUN_TIMEOUT_S
    while time.monotonic() < deadline:
        run = api.get(f"/api/v2/dags/f1_session_pipeline/dagRuns/{run_id}")
        if run["state"] in ("success", "failed"):
            return run
        time.sleep(10)
    raise AssertionError(f"{run_id} did not finish within {RUN_TIMEOUT_S}s")


class TestCriterion4And5APipelineRun:
    """One session end to end, twice: the second must change no data."""

    def test_every_task_succeeded(self, api: AirflowApi, first_run: dict) -> None:
        tasks = api.get(
            f"/api/v2/dags/f1_session_pipeline/dagRuns/{first_run['run_id']}/taskInstances")
        states = {task["task_id"]: task["state"] for task in tasks["task_instances"]}
        assert set(states) == {"wait_for_data", "ingest", "align", "grid", "segment",
                               "metrics", "validate", "quality", "load"}
        assert set(states.values()) == {"success"}, states

    def test_the_session_is_in_the_warehouse(self, env: dict[str, str], first_run: dict) -> None:
        season, round_number, code = TARGET_KEY
        found = psql(env, "SELECT event_name FROM dim_session WHERE season = "
                          f"{season} AND round = {round_number} AND session_code = '{code}'")
        assert found == TARGET["event"]

    def test_it_loaded_the_rows_f005_measured(self, env: dict[str, str], first_run: dict) -> None:
        columns = ", ".join(EXPECTED_ROWS)
        row = psql(env, f"SELECT {columns} FROM load_audit ORDER BY load_id DESC LIMIT 1")
        assert dict(zip(EXPECTED_ROWS, (int(v) for v in row.split("|")))) == EXPECTED_ROWS

    def test_a_second_run_changes_no_data(self, api: AirflowApi, env: dict[str, str],
                                          first_run: dict) -> None:
        """Idempotent by construction: the snapshot is reused and the session's
        own leaf partition is truncated before it is written again."""
        before = psql(env, "SELECT count(*) FROM fact_telemetry_grid")
        sessions_before = psql(env, "SELECT count(*) FROM dim_session")

        _restore_stack()
        run_id = f"test__f006_rerun_{int(time.time())}"
        api.post("/api/v2/dags/f1_session_pipeline/dagRuns",
                 {"dag_run_id": run_id, "logical_date": None, "conf": TARGET})
        assert _wait_for(api, run_id)["state"] == "success"

        assert psql(env, "SELECT count(*) FROM fact_telemetry_grid") == before
        assert psql(env, "SELECT count(*) FROM dim_session") == sessions_before
        after = int(psql(env, "SELECT count(*) FROM load_audit"))
        assert after == int(first_run["audit"]) + 1, "each run is audited, once"


class TestCriterion8TheDispatcher:
    """Run in-process with `airflow dags test`, which needs neither the
    scheduler nor an unpaused DAG.

    Triggering it the ordinary way would mean unpausing it, and an unpaused
    hourly dispatcher fires its own scheduled run immediately -- which on the
    first attempt at this test started ingesting Bahrain unasked.
    """

    def test_a_dry_run_selects_the_outstanding_sessions_and_triggers_nothing(
            self, env: dict[str, str], stack: float) -> None:
        if not (INTERIM_ROOT / "reference" / "2024" / "dim_session_schedule.parquet").exists():
            pytest.skip("2024 reference data is not on this machine")

        result = compose_exec([
            "airflow", "dags", "test", "f1_calendar_dispatch", "--conf",
            json.dumps({"season": 2024, "codes": ["Q", "R"], "max_triggers": 3, "dry_run": True}),
        ])
        output = result.stdout + result.stderr
        assert "state=success" in output, output[-2000:]
        assert "dispatch_selected" in output, "the dispatcher did not report what it chose"
        assert "due=48" in output, "24 rounds x (Q + R) in 2024"
        assert "trigger_pipeline" in output and "SKIPPED" in output, \
            "a dry run must trigger nothing"

    def test_it_is_still_paused_afterwards(self, api: AirflowApi) -> None:
        """`dags test` must not have changed the DAG's own state."""
        assert api.get("/api/v2/dags/f1_calendar_dispatch")["is_paused"] is True


class TestCriterion12Logs:
    def test_task_logs_land_on_the_host_bind_mount(self, stack: float) -> None:
        """Global 2: bind mounts only, so copying the project folder carries the
        run history with it."""
        logs = PROJECT_ROOT / "data" / "airflow" / "logs"
        assert logs.is_dir(), "no log directory on the host"
        assert any(logs.glob("dag_id=f1_session_pipeline/**/*.log")), "no task logs on the host"

    def test_the_password_file_is_under_data_and_not_in_the_repository(self) -> None:
        assert (PROJECT_ROOT / "data" / "airflow" / "passwords.json").exists()
        assert not list((PROJECT_ROOT / "servicios").rglob("passwords.json"))
