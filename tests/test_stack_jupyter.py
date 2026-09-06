"""F014 integration: the `dev` profile on the real Docker daemon.

Opt-in, like the other stack suites: `pytest -m docker`. Brings up postgres and
jupyter, asserts the runtime criteria, and executes the versioned smoke
notebook headlessly -- which is the acceptance vehicle, so a broken kernel, a
broken mount or a mis-granted role all fail here rather than being discovered
by hand three weeks later.

Credentials come from `StackEnv`, whose repr hides its values, and travel to
docker through the environment rather than an argument list.
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

from src.config import PROJECT_ROOT
from tests.stack_env import SERVICIOS, env_values

pytestmark = pytest.mark.docker

COMPOSE = ["docker", "compose", "-f", str(SERVICIOS / "docker-compose.yml"), "--profile", "dev"]
STARTUP_TIMEOUT_S = 180

#: Criterion 5. The prototype reached healthy 3.5 s after `docker run`; this is
#: measured from before `up`, so it also covers postgres coming up first.
HEALTHY_BUDGET_S = 90.0
#: Criterion 6. The prototype idled at 69 MiB and sat at 72 MiB after a run.
IDLE_MEMORY_MAX_MIB = 512.0
#: Criterion 8. The prototype ran the three cells in 6.2 s, ~5 s of it kernel start.
NOTEBOOK_BUDGET_S = 30.0
SMOKE_NOTEBOOK = "/opt/project/notebooks/00_stack_smoke.ipynb"


def run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=900, **kwargs)


def container(env: dict[str, str], service: str = "jupyter") -> str:
    return f"{env['COMPOSE_PROJECT_NAME']}-{service}"


def health(env: dict[str, str], service: str = "jupyter") -> str:
    return run(["docker", "inspect", "-f", "{{.State.Health.Status}}", container(env, service)]).stdout.strip()


def api(env: dict[str, str], path: str, token: str | None = None, timeout_s: int = 10) -> int:
    """HTTP status from the notebook server. Any status is a result, not an error."""
    request = urllib.request.Request(f"http://127.0.0.1:{env['JUPYTER_PORT']}{path}")
    if token:
        request.add_header("Authorization", f"token {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


@pytest.fixture(scope="module")
def env() -> dict[str, str]:
    if shutil.which("docker") is None:
        pytest.skip("docker is not on PATH")
    values = env_values()
    for key in ("JUPYTER_TOKEN", "JUPYTER_PORT", "JUPYTER_BIND_ADDRESS"):
        if key not in values:
            pytest.skip(f"{key} is absent from servicios/.env")
    return values


@pytest.fixture(scope="module")
def stack(env: dict[str, str]) -> float:
    """Seconds from `up` until jupyter reports healthy.

    The image is built first and not timed. F017 found the same fixture shape
    failing once when a rebase invalidated the build cache: a cold image build
    inside the timed window measures how warm Docker's cache is, not how
    quickly the service starts, which is what the criterion is about.
    """
    built = run([*COMPOSE, "build", "jupyter"])
    assert built.returncode == 0, built.stderr[-800:]
    started = time.monotonic()
    result = run([*COMPOSE, "up", "-d"])
    assert result.returncode == 0, result.stderr[-800:]
    while time.monotonic() - started < STARTUP_TIMEOUT_S:
        if health(env) == "healthy":
            return time.monotonic() - started
        time.sleep(2)
    raise AssertionError(f"jupyter not healthy within {STARTUP_TIMEOUT_S}s: {health(env)!r}")


@pytest.fixture(scope="module")
def executed_notebook(env: dict[str, str], stack: float) -> tuple[dict, float]:
    """The smoke notebook run headlessly inside the container, and its seconds.

    `--stdout` on purpose: the executed copy never touches the versioned file,
    so running the suite cannot put outputs into git.
    """
    started = time.monotonic()
    result = run(["docker", "exec", container(env), "jupyter", "nbconvert",
                  "--to", "notebook", "--execute", "--stdout", SMOKE_NOTEBOOK])
    elapsed = time.monotonic() - started
    assert result.returncode == 0, result.stderr[-800:]
    return json.loads(result.stdout), elapsed


class TestCriterion5Startup:
    def test_it_reaches_healthy_within_budget(self, stack: float) -> None:
        assert stack <= HEALTHY_BUDGET_S, f"healthy after {stack:.0f}s"

    def test_postgres_came_up_with_it(self, env: dict[str, str], stack: float) -> None:
        """D9: the dev profile has to yield a working pair, not a lone kernel."""
        assert health(env, "postgres") == "healthy"


class TestCriteria3And4TheImage:
    def test_it_runs_as_uid_1000_with_a_name(self, env: dict[str, str], stack: float) -> None:
        """F013's lesson: a numeric USER with no passwd entry breaks tooling."""
        result = run(["docker", "exec", container(env), "id"])
        assert "uid=1000(" in result.stdout, result.stdout

    def test_its_versions_are_the_conda_environments(self, env: dict[str, str], stack: float) -> None:
        """One requirements.txt across the env, the images and every kernel."""
        import numpy
        import pandas
        import pyarrow

        result = run(["docker", "exec", container(env), "python", "-c",
                      "import sys,pandas,numpy,pyarrow;"
                      "print(sys.version.split()[0],pandas.__version__,numpy.__version__,pyarrow.__version__)"])
        inside = result.stdout.split()
        assert inside[1:] == [pandas.__version__, numpy.__version__, pyarrow.__version__], result.stdout
        assert inside[0].startswith("3.12."), inside[0]

    def test_its_dependencies_do_not_conflict(self, env: dict[str, str], stack: float) -> None:
        result = run(["docker", "exec", container(env), "pip", "check"])
        assert result.returncode == 0, result.stdout

    def test_the_resource_limits_are_applied(self, env: dict[str, str], stack: float) -> None:
        result = run(["docker", "inspect", "-f", "{{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}", container(env)])
        memory, nano_cpus = (int(v) for v in result.stdout.split())
        assert memory == 4 * 1024**3
        assert nano_cpus == 2_000_000_000


class TestCriterion6Memory:
    def test_it_idles_well_inside_its_limit(self, env: dict[str, str], stack: float) -> None:
        result = run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container(env)])
        used = result.stdout.split("/")[0].strip()
        value = float(used.rstrip("GMKiB"))
        mib = value * 1024 if used.endswith("GiB") else value
        assert mib <= IDLE_MEMORY_MAX_MIB, f"idling at {used}"


class TestCriterion7Authentication:
    def test_the_api_is_reachable_without_a_token(self, env: dict[str, str], stack: float) -> None:
        """What the healthcheck probes; it says nothing about the files."""
        assert api(env, "/api") == 200

    def test_contents_are_refused_without_a_token(self, env: dict[str, str], stack: float) -> None:
        assert api(env, "/api/contents") == 403

    def test_contents_are_served_with_the_token(self, env: dict[str, str], stack: float) -> None:
        assert api(env, "/api/contents", token=env["JUPYTER_TOKEN"]) == 200

    def test_the_port_is_published_on_loopback_only(self, env: dict[str, str], stack: float) -> None:
        """A notebook server executes arbitrary code; it does not face a LAN."""
        result = run(["docker", "inspect", "-f", "{{json .NetworkSettings.Ports}}", container(env)])
        bindings = json.loads(result.stdout)["8888/tcp"]
        assert [b["HostIp"] for b in bindings] == ["127.0.0.1"], bindings


class TestCriteria8To10TheSmokeNotebook:
    def test_no_cell_raised(self, executed_notebook: tuple[dict, float]) -> None:
        notebook, _ = executed_notebook
        errors = [f"{o.get('ename')}: {o.get('evalue')}"
                  for cell in notebook["cells"] for o in cell.get("outputs", [])
                  if o.get("output_type") == "error"]
        assert errors == [], errors

    def test_it_ran_within_budget(self, executed_notebook: tuple[dict, float]) -> None:
        _, elapsed = executed_notebook
        assert elapsed <= NOTEBOOK_BUDGET_S, f"{elapsed:.1f}s"

    def test_the_kernel_imported_the_project(self, executed_notebook: tuple[dict, float]) -> None:
        assert "src imports from /opt/project" in self.text(executed_notebook)

    def test_it_read_interim_parquet_through_the_bind_mount(self, executed_notebook: tuple[dict, float]) -> None:
        assert "grid rows in" in self.text(executed_notebook)

    def test_the_read_only_role_refused_a_write(self, executed_notebook: tuple[dict, float]) -> None:
        """Criterion 10, and the reason a kernel gets that role at all."""
        assert "write refused as it should be" in self.text(executed_notebook)

    def test_the_versioned_file_still_has_no_outputs(self, executed_notebook) -> None:
        """`--stdout` keeps outputs out of the file even when the suite runs it.

        Read from disk rather than asked of git, so this holds on the commit
        that introduces the notebook as well as on every one after.
        """
        on_disk = json.loads((PROJECT_ROOT / "notebooks" / "00_stack_smoke.ipynb").read_text(encoding="utf-8"))
        stored = [c.get("id") for c in on_disk["cells"]
                  if c.get("outputs") or c.get("execution_count") is not None]
        assert stored == [], f"running the suite wrote outputs into the versioned notebook: {stored}"

    @staticmethod
    def text(executed: tuple[dict, float]) -> str:
        notebook, _ = executed
        return "\n".join("".join(o.get("text", []))
                         for cell in notebook["cells"] for o in cell.get("outputs", []))


class TestIdempotency:
    def test_a_second_up_recreates_nothing(self, env: dict[str, str], stack: float) -> None:
        before = run(["docker", "inspect", "-f", "{{.Id}}", container(env)]).stdout.strip()
        result = run([*COMPOSE, "up", "-d"])
        assert result.returncode == 0, result.stderr[-400:]
        after = run(["docker", "inspect", "-f", "{{.Id}}", container(env)]).stdout.strip()
        assert before == after, "the container was recreated"


class TestRepositoryHygiene:
    def test_the_service_wrote_nothing_tracked(self, env: dict[str, str], stack: float) -> None:
        """What the container writes must be invisible to git.

        Only paths a running service touches are considered -- the source and
        compose changes of whatever feature is in progress are not the stack's
        doing. Same shape as F001's criterion 8.
        """
        result = run(["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT))
        dirty = [line for line in result.stdout.splitlines()
                 if "/data/" in line or line.endswith(".env") or ".ipynb_checkpoints" in line]
        assert dirty == [], f"stack output is visible to git: {dirty}"

    def test_jupyter_state_persists_under_the_gitignored_data_tree(self) -> None:
        assert (PROJECT_ROOT / "data" / "jupyter").is_dir()
