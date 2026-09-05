"""Connection settings for the project's Postgres.

Credentials come from ``servicios/.env`` when it is there and from the
environment otherwise, so the same code works from the conda env on the host
(D3) and from inside a container, where the host is the service name.

The password is kept out of ``repr``, out of logs and out of exception
messages: an assertion failure printing a fixture is exactly how one escaped
during F001, and that is not repeated here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

from src.config import PROJECT_ROOT

ENV_FILE = PROJECT_ROOT / "servicios" / ".env"

#: The literal IPv4 address, not "localhost". Measured on 2026-09-04: this
#: machine resolves localhost to ::1 before 127.0.0.1, Docker publishes the
#: port on IPv4 only, and every connection therefore stalls on the IPv6
#: attempt -- 10 s with a timeout set, indefinitely without one. Inside the
#: compose network the host is the service name and this default does not apply.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 55432

#: So a wrong host or a stopped stack fails in seconds instead of hanging.
DEFAULT_CONNECT_TIMEOUT_S = 10


class SettingsError(RuntimeError):
    """The database connection is not configured."""


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """Key-value pairs from a .env file; an absent file is not an error."""
    if not path.exists():
        return {}
    return {
        line.split("=", 1)[0].strip(): line.split("=", 1)[1].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.strip().startswith("#")
    }


@dataclass(frozen=True)
class Settings:
    """Where the warehouse is and who connects to it."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    database: str = "f1_microsector"
    user: str = "f1_admin"
    password: str = field(default="", repr=False)
    connect_timeout_s: int = DEFAULT_CONNECT_TIMEOUT_S

    @classmethod
    def from_env(cls, env_file: Path = ENV_FILE, environ: dict[str, str] | None = None,
                 database: str | None = None) -> "Settings":
        """Settings from ``servicios/.env``, overridden by the environment.

        The environment wins so a container can point at the service name
        without editing the file the host uses.
        """
        environ = dict(os.environ if environ is None else environ)
        values = {**read_env_file(env_file), **environ}
        password = values.get("POSTGRES_PASSWORD", "")
        if not password:
            raise SettingsError(
                f"no POSTGRES_PASSWORD in {env_file} or the environment; copy servicios/.env.example to .env"
            )
        return cls(
            host=values.get("POSTGRES_HOST", DEFAULT_HOST),
            port=int(values.get("POSTGRES_PORT", DEFAULT_PORT)),
            database=database or values.get("POSTGRES_DB", "f1_microsector"),
            user=values.get("POSTGRES_USER", "f1_admin"),
            password=password,
            connect_timeout_s=int(values.get("POSTGRES_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT_S)),
        )

    def with_database(self, database: str) -> "Settings":
        """The same server, a different database -- what the tests use."""
        return Settings(host=self.host, port=self.port, database=database, user=self.user,
                        password=self.password, connect_timeout_s=self.connect_timeout_s)

    @property
    def dsn(self) -> str:
        """Connection string. Never log this: it carries the password."""
        return (
            f"host={self.host} port={self.port} dbname={self.database} "
            f"user={self.user} password={self.password} connect_timeout={self.connect_timeout_s}"
        )

    @property
    def safe_dsn(self) -> str:
        """The same connection, safe to log."""
        return f"host={self.host} port={self.port} dbname={self.database} user={self.user}"


def connect(settings: Settings | None = None, *, autocommit: bool = False) -> psycopg.Connection:
    """Open a connection. Failures name the server, never the credentials."""
    settings = settings or Settings.from_env()
    try:
        return psycopg.connect(settings.dsn, autocommit=autocommit)
    except psycopg.OperationalError as exc:
        message = str(exc).replace(settings.password, "***") if settings.password else str(exc)
        raise SettingsError(f"cannot connect to {settings.safe_dsn}: {message}") from None
