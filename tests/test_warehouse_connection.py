"""F005 connection settings: configurable, and never leaking the password."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.warehouse.connection import DEFAULT_HOST, Settings, SettingsError, read_env_file

ENV_BODY = """
# a comment
COMPOSE_PROJECT_NAME=f1-microsector
POSTGRES_USER=f1_admin
POSTGRES_PASSWORD=s3cr3t-from-the-file
POSTGRES_DB=f1_microsector
POSTGRES_PORT=55432
"""


@pytest.fixture()
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text(ENV_BODY, encoding="utf-8")
    return path


class TestReadEnvFile:
    def test_reads_pairs_and_skips_comments(self, env_file: Path) -> None:
        values = read_env_file(env_file)
        assert values["POSTGRES_USER"] == "f1_admin"
        assert "# a comment" not in values
        assert len(values) == 5

    def test_a_missing_file_is_not_an_error(self, tmp_path: Path) -> None:
        assert read_env_file(tmp_path / "absent") == {}


class TestSettings:
    def test_from_the_env_file(self, env_file: Path) -> None:
        settings = Settings.from_env(env_file, environ={})
        assert settings.user == "f1_admin" and settings.port == 55432
        assert settings.database == "f1_microsector" and settings.host == DEFAULT_HOST

    def test_the_environment_overrides_the_file(self, env_file: Path) -> None:
        """A container points at the service name without editing the host's file."""
        settings = Settings.from_env(env_file, environ={"POSTGRES_HOST": "postgres", "POSTGRES_PORT": "5432"})
        assert settings.host == "postgres" and settings.port == 5432

    def test_a_missing_password_is_an_error_naming_the_file(self, tmp_path: Path) -> None:
        with pytest.raises(SettingsError, match="POSTGRES_PASSWORD"):
            Settings.from_env(tmp_path / "absent", environ={})

    def test_with_database_keeps_the_server(self, env_file: Path) -> None:
        settings = Settings.from_env(env_file, environ={}).with_database("scratch")
        assert settings.database == "scratch" and settings.user == "f1_admin"
        assert settings.password == "s3cr3t-from-the-file"


class TestHostDefault:
    """Measured on 2026-09-04: 127.0.0.1 connects in 0.03 s, localhost in 10 s.

    This machine resolves localhost to ::1 first and Docker publishes the port
    on IPv4 only, so the IPv6 attempt stalls until the timeout -- and without a
    timeout it never returns at all. The default is the literal address, and a
    timeout is always set.
    """

    def test_the_default_host_is_the_literal_ipv4_address(self) -> None:
        assert DEFAULT_HOST == "127.0.0.1"
        assert Settings().host == "127.0.0.1"

    def test_every_dsn_carries_a_connect_timeout(self, env_file: Path) -> None:
        settings = Settings.from_env(env_file, environ={})
        assert "connect_timeout=" in settings.dsn
        assert settings.connect_timeout_s > 0

    def test_the_timeout_is_configurable(self, env_file: Path) -> None:
        settings = Settings.from_env(env_file, environ={"POSTGRES_CONNECT_TIMEOUT": "3"})
        assert settings.connect_timeout_s == 3 and "connect_timeout=3" in settings.dsn


class TestSecrecy:
    """A password reached a terminal once during F001. Not again."""

    def test_repr_hides_the_password(self, env_file: Path) -> None:
        settings = Settings.from_env(env_file, environ={})
        assert "s3cr3t" not in repr(settings)
        assert "f1_admin" in repr(settings), "everything else stays legible"

    def test_safe_dsn_omits_it_and_dsn_carries_it(self, env_file: Path) -> None:
        settings = Settings.from_env(env_file, environ={})
        assert "s3cr3t" not in settings.safe_dsn
        assert "password" not in settings.safe_dsn
        assert "s3cr3t-from-the-file" in settings.dsn

    def test_a_connection_failure_does_not_echo_the_password(self) -> None:
        settings = Settings(host="127.0.0.1", port=1, database="none", user="u", password="s3cr3t-from-the-file")
        from src.warehouse.connection import connect

        with pytest.raises(SettingsError) as excinfo:
            connect(settings)
        assert "s3cr3t" not in str(excinfo.value)
        assert "dbname=none" in str(excinfo.value)
