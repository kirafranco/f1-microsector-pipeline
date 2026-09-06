"""F017 integration: the `pitwall` profile on the real Docker daemon.

Opt-in, like the other stack suites: `pytest -m docker`.

The criterion that matters most is cross-consumer equivalence. D1 constrained
F005 so the schema would carry no dashboard-specific column, view or naming,
precisely so a second consumer could be added later. Two consumers reading one
schema is only an architecture if they agree, so this suite runs Grafana's own
panel SQL and the service's overlay for the same lap pair and compares the rows.
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

from src.pitwall import queries
from tests import dashboard_support as dash
from tests.stack_env import SERVICIOS, env_values

pytestmark = pytest.mark.docker

COMPOSE = ["docker", "compose", "-f", str(SERVICIOS / "docker-compose.yml"), "--profile", "pitwall"]
STARTUP_TIMEOUT_S = 180

#: Criterion 4. The prototype answered 1.4 s after `docker run`; this is measured
#: from before `up`, so it also covers postgres coming up first.
HEALTHY_BUDGET_S = 90.0
#: Criterion 5. The prototype idled at 54 MiB and reached 79 MiB after 21 renders.
IDLE_MEMORY_MAX_MIB = 256.0
#: Criterion 6. F007's dashboard budget, applied to this consumer too.
PAGE_BUDGET_MS = 250.0
WARM_HITS = 20

SESSION_ID = 39
DRIVER_A, DRIVER_B = "NOR", "PIA"


def run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=900, **kwargs)


def container(env: dict[str, str], service: str = "pitwall") -> str:
    return f"{env['COMPOSE_PROJECT_NAME']}-{service}"


def health(env: dict[str, str], service: str = "pitwall") -> str:
    return run(["docker", "inspect", "-f", "{{.State.Health.Status}}", container(env, service)]).stdout.strip()


def psql(env: dict[str, str], sql: str) -> str:
    """Query as the admin role. The password travels in the environment."""
    result = run(["docker", "exec", "-e", "PGPASSWORD", container(env, "postgres"),
                  "psql", "-U", env["POSTGRES_USER"], "-d", env["POSTGRES_DB"], "-tAc", sql],
                 env={**os.environ, "PGPASSWORD": env["POSTGRES_PASSWORD"]})
    assert result.returncode == 0, result.stderr[-400:]
    return result.stdout


def fetch(env: dict[str, str], path: str, timeout_s: int = 30) -> tuple[int, bytes, float]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{env['PITWALL_PORT']}{path}", timeout=timeout_s) as response:
            return response.status, response.read(), (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), (time.perf_counter() - started) * 1000


@pytest.fixture(scope="module")
def env() -> dict[str, str]:
    if shutil.which("docker") is None:
        pytest.skip("docker is not on PATH")
    values = env_values()
    for key in ("PITWALL_PORT", "PITWALL_BIND_ADDRESS"):
        if key not in values:
            pytest.skip(f"{key} is absent from servicios/.env")
    return values


@pytest.fixture(scope="module")
def stack(env: dict[str, str]) -> float:
    started = time.monotonic()
    result = run([*COMPOSE, "up", "-d", "--build"])
    assert result.returncode == 0, result.stderr[-800:]
    while time.monotonic() - started < STARTUP_TIMEOUT_S:
        if health(env) == "healthy":
            return time.monotonic() - started
        time.sleep(2)
    raise AssertionError(f"pitwall not healthy within {STARTUP_TIMEOUT_S}s: {health(env)!r}")


@pytest.fixture(scope="module")
def overlay(env: dict[str, str], stack: float) -> dict:
    status, body, _ = fetch(
        env, f"/api/overlay.json?session_id={SESSION_ID}&driver_a={DRIVER_A}&driver_b={DRIVER_B}")
    assert status == 200, body[:300]
    return json.loads(body)


class TestCriterion4Startup:
    def test_it_reaches_healthy_within_budget(self, stack: float) -> None:
        assert stack <= HEALTHY_BUDGET_S, f"healthy after {stack:.0f}s"

    def test_postgres_came_up_with_it(self, env: dict[str, str], stack: float) -> None:
        """D9: the profile must yield a working pair, not a service with no data."""
        assert health(env, "postgres") == "healthy"


class TestCriteria3And5TheContainer:
    def test_it_runs_as_uid_1000_with_a_name(self, env: dict[str, str], stack: float) -> None:
        assert "uid=1000(" in run(["docker", "exec", container(env), "id"]).stdout

    def test_the_resource_limits_are_the_visualisation_row(self, env: dict[str, str], stack: float) -> None:
        """Global 2.2 gives visualisation 1 GB / 1 core. No exception needed,
        which is one of the reasons this was chosen over Superset."""
        result = run(["docker", "inspect", "-f", "{{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}", container(env)])
        memory, nano_cpus = (int(v) for v in result.stdout.split())
        assert memory == 1024**3
        assert nano_cpus == 1_000_000_000

    def test_it_idles_well_inside_its_limit(self, env: dict[str, str], stack: float) -> None:
        used = run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container(env)]).stdout.split("/")[0].strip()
        value = float(used.rstrip("GMKiB"))
        mib = value * 1024 if used.endswith("GiB") else value
        assert mib <= IDLE_MEMORY_MAX_MIB, f"idling at {used}"

    def test_the_port_is_published_on_loopback_only(self, env: dict[str, str], stack: float) -> None:
        """The page has no authentication, by decision 6. It stays on loopback."""
        bindings = json.loads(run(["docker", "inspect", "-f", "{{json .NetworkSettings.Ports}}",
                                   container(env)]).stdout)["8000/tcp"]
        assert [b["HostIp"] for b in bindings] == ["127.0.0.1"], bindings

    def test_it_mounts_nothing_writable(self, env: dict[str, str], stack: float) -> None:
        """It reads the warehouse over the network and holds nothing."""
        mounts = json.loads(run(["docker", "inspect", "-f", "{{json .Mounts}}", container(env)]).stdout)
        writable = [m["Destination"] for m in mounts if m.get("RW")]
        assert writable == [], writable


class TestCriteria6And7TheService:
    def test_health_answers(self, env: dict[str, str], stack: float) -> None:
        status, body, _ = fetch(env, "/health")
        assert status == 200 and json.loads(body) == {"ok": True}

    def test_the_index_lists_the_loaded_sessions(self, env: dict[str, str], stack: float) -> None:
        status, body, _ = fetch(env, "/")
        assert status == 200 and b"Abu Dhabi" in body

    def test_the_page_renders_within_the_dashboard_budget(self, env: dict[str, str], stack: float) -> None:
        """F007's 250 ms budget, applied to the second consumer as well."""
        path = f"/pitwall?session_id={SESSION_ID}&driver_a={DRIVER_A}&driver_b={DRIVER_B}"
        fetch(env, path)  # warm the pool and let psycopg prepare the statement
        times = [fetch(env, path)[2] for _ in range(WARM_HITS)]
        p95 = sorted(times)[int(round(0.95 * (len(times) - 1)))]
        assert p95 <= PAGE_BUDGET_MS, f"p95 {p95:.0f} ms over {PAGE_BUDGET_MS:.0f} ms; times {sorted(times)}"

    def test_plotly_is_served_by_the_service_itself(self, env: dict[str, str], stack: float) -> None:
        """Global 4.0: no unvetted executable fetched at runtime, and a pit wall
        has to work with no internet."""
        status, body, _ = fetch(env, "/static/plotly.min.js")
        assert status == 200 and len(body) > 1_000_000

    def test_the_page_fetches_nothing_external(self, env: dict[str, str], stack: float) -> None:
        _, body, _ = fetch(env, f"/pitwall?session_id={SESSION_ID}&driver_a={DRIVER_A}&driver_b={DRIVER_B}")
        page = body.decode()
        assert 'src="/static/plotly.min.js"' in page
        assert "cdn.plot.ly" not in page and 'src="http' not in page

    def test_asking_for_one_driver_twice_is_a_404(self, env: dict[str, str], stack: float) -> None:
        status, _, _ = fetch(env, f"/pitwall?session_id={SESSION_ID}&driver_a=NOR&driver_b=NOR")
        assert status == 404


class TestCriterion8TheInteraction:
    """The reason this service exists, checked on the figure the browser gets."""

    def test_every_panel_shares_one_x_axis(self, overlay: dict) -> None:
        axes = {k: v for k, v in overlay["figure"]["layout"].items() if k.startswith("xaxis")}
        anchored = [v.get("matches") for v in axes.values() if v.get("matches")]
        assert len(axes) == 6
        assert len(anchored) == 5 and len(set(anchored)) == 1

    def test_the_tooltip_is_unified_and_the_crosshair_crosses(self, overlay: dict) -> None:
        layout = overlay["figure"]["layout"]
        assert layout["hovermode"] == "x unified"
        axes = [v for k, v in layout.items() if k.startswith("xaxis")]
        assert all(a.get("showspikes") and a.get("spikemode") == "across" for a in axes)


class TestCriterion9CrossConsumerEquivalence:
    """The service and Grafana must return the same numbers for the same laps.

    This is D1's constraint on F005 made testable: one schema, no view, no
    dashboard-specific column, two different consumers that agree.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def grafana_rows(env: dict[str, str], overlay: dict) -> list[list]:
        """Grafana's own overlay panel SQL, run with the same two laps."""
        dashboard = dash.load()
        sql = next(sql for title, sql in dash.panel_sql(dashboard) if "vs $driver_b" in title)
        resolved = {"session": str(SESSION_ID),
                    "lap_a": str(overlay["lap_a"]["lap_id"]),
                    "lap_b": str(overlay["lap_b"]["lap_id"]),
                    "driver_a": f"'{DRIVER_A}'", "driver_b": f"'{DRIVER_B}'"}
        statement = dash.substitute(sql, resolved)
        assert dash.unresolved(statement) == [], dash.unresolved(statement)
        output = psql(env, statement.replace("\n", " "))
        return [line.split("|") for line in output.strip().splitlines()]

    def test_both_consumers_return_the_same_number_of_points(self, overlay: dict, grafana_rows: list) -> None:
        assert len(overlay["rows"]) == len(grafana_rows)

    def test_the_shared_channels_agree_value_for_value(self, overlay: dict, grafana_rows: list) -> None:
        """Grafana's panel carries distance, Δt, both speeds, throttles and
        brakes; the service adds gear and DRS. Every shared column must match."""
        columns = overlay["columns"]
        service = [[row[columns.index(name)] for name in
                    ("distance_m", "delta_t_s", "speed_a", "speed_b",
                     "throttle_a", "throttle_b", "brake_a", "brake_b")]
                   for row in overlay["rows"]]
        mismatches = []
        for index, (mine, theirs) in enumerate(zip(service, grafana_rows)):
            for column, (a, b) in enumerate(zip(mine, theirs)):
                if a is None and b == "":
                    continue
                if abs(float(a) - float(b)) > 1e-6:
                    mismatches.append((index, column, a, b))
        assert mismatches == [], mismatches[:5]

    def test_the_laps_compared_are_each_drivers_fastest(self, env: dict[str, str], overlay: dict) -> None:
        """Both consumers default a driver's lap the same way (F007's rule)."""
        for side, code in (("lap_a", DRIVER_A), ("lap_b", DRIVER_B)):
            expected = psql(env, "SELECT lap_id FROM dim_lap WHERE session_id = "
                                 f"{SESSION_ID} AND code = '{code}' ORDER BY lap_time_s NULLS LAST LIMIT 1").strip()
            assert str(overlay[side]["lap_id"]) == expected


class TestCriteria10And11TheSchemaIsUntouched:
    def test_the_service_cannot_write(self, env: dict[str, str], stack: float) -> None:
        """It connects as the read-only role; a write must be refused."""
        script = (
            "import os, psycopg;"
            "dsn=f\"postgresql://{os.environ['POSTGRES_READONLY_USER']}:{os.environ['POSTGRES_READONLY_PASSWORD']}\""
            "f\"@{os.environ['POSTGRES_HOST']}:5432/{os.environ['POSTGRES_DB']}\";"
            "c=psycopg.connect(dsn);"
            "\ntry:\n c.execute('CREATE TABLE pitwall_should_not_exist (x int)');print('ALLOWED')"
            "\nexcept psycopg.errors.InsufficientPrivilege:\n print('REFUSED')"
        )
        result = run(["docker", "exec", container(env), "python", "-c", script])
        assert "REFUSED" in result.stdout, result.stdout + result.stderr[-300:]

    def test_it_connects_as_the_read_only_role(self, env: dict[str, str], stack: float) -> None:
        result = run(["docker", "exec", container(env), "python", "-c",
                      "from src.warehouse.connection import Settings;"
                      "print(Settings.from_env(role='readonly').user)"])
        assert result.stdout.strip() == env["POSTGRES_READONLY_USER"]

    def test_the_database_still_holds_no_view_or_function(self, env: dict[str, str], stack: float) -> None:
        """D1's constraint on F005: the schema serves any consumer because it
        was built for none of them. Adding a second one must not have changed it."""
        views = psql(env, "SELECT count(*) FROM pg_views WHERE schemaname = 'public'").strip()
        functions = psql(env, "SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
                              "WHERE n.nspname = 'public'").strip()
        assert (views, functions) == ("0", "0")

    def test_grafana_is_untouched_by_this_feature(self, env: dict[str, str], stack: float) -> None:
        """Additive, not a migration: the dashboard file is the one F023 left."""
        dashboard = dash.load()
        assert dashboard["uid"] == "f1-pit-wall"
        assert len(dash.panel_sql(dashboard)) == 18


class TestIdempotency:
    def test_a_second_up_recreates_nothing(self, env: dict[str, str], stack: float) -> None:
        before = run(["docker", "inspect", "-f", "{{.Id}}", container(env)]).stdout.strip()
        assert run([*COMPOSE, "up", "-d"]).returncode == 0
        after = run(["docker", "inspect", "-f", "{{.Id}}", container(env)]).stdout.strip()
        assert before == after
