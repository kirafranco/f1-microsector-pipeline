"""F001: the global container standards, asserted as rules against the compose file.

These tests never start Docker. They read `servicios/` and check that what is
written there obeys the standards in the global CLAUDE.md -- bind mounts only,
pinned images, profiles, limits, loopback ports, no credential anywhere.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from src.config import PROJECT_ROOT

SERVICIOS = PROJECT_ROOT / "servicios"
COMPOSE_PATH = SERVICIOS / "docker-compose.yml"
ENV_EXAMPLE = SERVICIOS / ".env.example"
ENV_FILE = SERVICIOS / ".env"
PROVISIONING = SERVICIOS / "grafana" / "provisioning"

#: Global CLAUDE.md 2, resource limits table.
LIMITS = {
    "postgres": {"memory": "2G", "cpus": "2"},
    "grafana": {"memory": "1G", "cpus": "1"},
}
RESTART_POLICIES = {"unless-stopped", "always"}
VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def services(compose: dict) -> dict:
    return compose["services"]


def env_keys(path: Path) -> set[str]:
    return {
        line.split("=", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.strip().startswith("#")
    }


class TestVolumes:
    def test_no_named_volumes_anywhere(self, compose: dict) -> None:
        """Global 2: NEVER named volumes, so the project folder is portable."""
        assert "volumes" not in compose, "top-level volumes: declares named volumes"

    def test_every_mount_is_a_relative_bind(self, services: dict) -> None:
        for name, service in services.items():
            for volume in service.get("volumes", []):
                assert isinstance(volume, str), f"{name}: use the short bind syntax"
                source = volume.split(":", 1)[0]
                assert source.startswith(("./", "../")), f"{name}: {source!r} is not a relative bind mount"

    def test_service_data_lives_under_the_gitignored_data_directory(self, services: dict) -> None:
        for name, service in services.items():
            sources = [v.split(":", 1)[0] for v in service.get("volumes", [])]
            data_mounts = [s for s in sources if s.startswith("../data/")]
            assert data_mounts, f"{name}: no persistent mount under ../data/"
            assert all(s == f"../data/{name}" for s in data_mounts), f"{name}: data mount is not ../data/{name}"

    def test_configuration_mounts_are_read_only(self, services: dict) -> None:
        for name, service in services.items():
            for volume in service.get("volumes", []):
                if not volume.split(":", 1)[0].startswith("../data/"):
                    assert volume.endswith(":ro"), f"{name}: {volume!r} should be mounted read-only"


class TestImages:
    def test_pinned_to_an_exact_version(self, services: dict) -> None:
        for name, service in services.items():
            image = service["image"]
            assert ":" in image, f"{name}: image is not tagged"
            tag = image.rsplit(":", 1)[1]
            assert tag != "latest", f"{name}: latest is not a pin"
            assert re.fullmatch(r"\d+(\.\d+)+", tag), f"{name}: {tag!r} is not an exact version"

    def test_official_images_only(self, services: dict) -> None:
        assert services["postgres"]["image"].startswith("postgres:")
        assert services["grafana"]["image"].startswith("grafana/grafana-oss:")

    def test_no_build_context_yet(self, services: dict) -> None:
        """Unmodified official images need no Dockerfile; the first one brings
        the .dockerignore obligation with it (F013/F014)."""
        for name, service in services.items():
            assert "build" not in service, f"{name}: a build context needs a .dockerignore"
        assert not (SERVICIOS / ".dockerignore").exists()


class TestProfilesAndNetwork:
    def test_every_service_is_in_the_core_profile(self, services: dict) -> None:
        """D9: nothing starts that the task does not need."""
        for name, service in services.items():
            assert service.get("profiles") == ["core"], f"{name}: not in the core profile"

    def test_one_custom_bridge_network(self, compose: dict) -> None:
        networks = compose["networks"]
        assert len(networks) == 1
        (definition,) = networks.values()
        assert definition["driver"] == "bridge"
        assert "${COMPOSE_PROJECT_NAME}" in definition["name"]

    def test_every_service_joins_it(self, compose: dict, services: dict) -> None:
        name = next(iter(compose["networks"]))
        for service_name, service in services.items():
            assert service.get("networks") == [name], f"{service_name}: not on {name}"

    def test_services_address_each_other_by_name(self, services: dict) -> None:
        """Global 2: never localhost or a static IP between containers."""
        environment = services["grafana"]["environment"]
        assert environment["POSTGRES_HOST"] == "postgres"
        rendered = yaml.safe_dump(services)
        assert "localhost:5432" not in rendered
        assert not re.search(r"\b\d{1,3}(\.\d{1,3}){3}:5432", rendered)


class TestRuntimePolicy:
    def test_restart_policies(self, services: dict) -> None:
        for name, service in services.items():
            assert service.get("restart") in RESTART_POLICIES, f"{name}: restart policy missing"

    @pytest.mark.parametrize("name", sorted(LIMITS))
    def test_resource_limits_match_the_global_table(self, services: dict, name: str) -> None:
        limits = services[name]["deploy"]["resources"]["limits"]
        assert limits["memory"] == LIMITS[name]["memory"]
        assert str(limits["cpus"]) == LIMITS[name]["cpus"]

    def test_postgres_healthcheck_queries_the_database(self, services: dict) -> None:
        """pg_isready answers yes on the temporary server the entrypoint runs
        during first initialisation, so it cannot gate a dependent service."""
        test = " ".join(services["postgres"]["healthcheck"]["test"])
        assert "psql" in test and "select 1" in test
        assert "pg_isready" not in test

    def test_grafana_waits_for_a_healthy_database(self, services: dict) -> None:
        depends = services["grafana"]["depends_on"]
        assert depends["postgres"]["condition"] == "service_healthy"
        assert "healthcheck" in services["grafana"]

    def test_host_ports_are_loopback_only(self, services: dict) -> None:
        for name, service in services.items():
            for mapping in service.get("ports", []):
                assert mapping.startswith("127.0.0.1:"), f"{name}: {mapping!r} is exposed beyond loopback"


class TestCredentials:
    def test_every_compose_variable_is_documented(self) -> None:
        """Compose resolves ${VAR} from .env, so every one must be a key there."""
        used = set(VAR_PATTERN.findall(COMPOSE_PATH.read_text(encoding="utf-8")))
        missing = used - env_keys(ENV_EXAMPLE)
        assert not missing, f"used in compose but not in .env.example: {sorted(missing)}"

    def test_every_provisioning_variable_reaches_grafana(self, services: dict) -> None:
        """Grafana resolves ${VAR} from its own container environment, which
        compose sets -- literally for the service name, from .env for the rest."""
        used: set[str] = set()
        for path in sorted(PROVISIONING.rglob("*.yml")):
            used |= set(VAR_PATTERN.findall(path.read_text(encoding="utf-8")))
        provided = set(services["grafana"]["environment"])
        missing = used - provided
        assert not missing, f"provisioning needs {sorted(missing)}, not in the grafana environment"

    def test_example_carries_placeholders_only(self) -> None:
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
            if "PASSWORD" in line and "=" in line and not line.strip().startswith("#"):
                assert line.split("=", 1)[1].strip() == "CHANGE_ME", f"{line.split('=')[0]}: real value in .env.example"

    def test_compose_project_name_is_defined(self) -> None:
        assert "COMPOSE_PROJECT_NAME" in env_keys(ENV_EXAMPLE)

    def test_no_credential_is_written_into_a_versioned_file(self, services: dict) -> None:
        """Every secret is a ${VAR} reference, resolved at runtime."""
        for name, service in services.items():
            for key, value in service.get("environment", {}).items():
                if "PASSWORD" in key or "SECRET" in key:
                    assert VAR_PATTERN.fullmatch(str(value)), f"{name}.{key}: not injected from .env"

    @pytest.mark.skipif(not ENV_FILE.exists(), reason="no local .env to compare against")
    def test_local_secrets_do_not_appear_in_any_versioned_file(self) -> None:
        """Reads .env to check for leaks and never prints what it read."""
        secrets = {
            line.split("=", 1)[1].strip()
            for line in ENV_FILE.read_text(encoding="utf-8").splitlines()
            if "PASSWORD" in line and "=" in line and not line.strip().startswith("#")
        }
        secrets = {s for s in secrets if len(s) > 8}
        assert secrets, ".env has no passwords to check"
        for path in sorted(SERVICIOS.rglob("*")):
            if not path.is_file() or path.name == ".env":
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            leaked = [p.name for p in [path] if any(s in content for s in secrets)]
            assert not leaked, f"{path.name} contains a value from .env"

    def test_gitignore_covers_env_and_data(self) -> None:
        rules = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert "**/.env" in rules
        assert "data/" in rules


class TestProvisioning:
    def test_datasource_is_provisioned_as_code(self) -> None:
        source = yaml.safe_load((PROVISIONING / "datasources" / "postgres.yml").read_text(encoding="utf-8"))
        (datasource,) = source["datasources"]
        assert datasource["uid"] == "f1-postgres"
        assert datasource["type"] == "grafana-postgresql-datasource"
        assert datasource["url"] == "${POSTGRES_HOST}:5432"
        assert datasource["editable"] is False

    def test_datasource_connects_as_the_read_only_role(self) -> None:
        source = yaml.safe_load((PROVISIONING / "datasources" / "postgres.yml").read_text(encoding="utf-8"))
        (datasource,) = source["datasources"]
        assert datasource["user"] == "${POSTGRES_READONLY_USER}"
        assert datasource["secureJsonData"]["password"] == "${POSTGRES_READONLY_PASSWORD}"

    def test_dashboard_provider_reads_versioned_json(self) -> None:
        providers = yaml.safe_load((PROVISIONING / "dashboards" / "dashboards.yml").read_text(encoding="utf-8"))
        (provider,) = providers["providers"]
        assert provider["type"] == "file"
        assert provider["allowUiUpdates"] is False, "a dashboard edited only in the UI does not exist"
        assert (PROVISIONING / "dashboards" / "json").is_dir()

    def test_init_script_creates_a_read_only_role(self) -> None:
        script = (SERVICIOS / "postgres" / "initdb" / "01-readonly-role.sh").read_text(encoding="utf-8")
        assert "CREATE ROLE" in script and "NOSUPERUSER" in script
        assert "ALTER DEFAULT PRIVILEGES" in script, "later tables would not be readable"
        assert "set -euo pipefail" in script
        assert "%I" in script and "%L" in script, "role and password must be quoted by format()"
