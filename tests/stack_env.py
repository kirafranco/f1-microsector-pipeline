"""Reaching the running stack from a test, without printing its credentials.

Extracted from the F001 integration tests so that F007's dashboard tests reuse
the same secret-hiding container rather than growing a second copy of it.
"""

from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT

SERVICIOS = PROJECT_ROOT / "servicios"


class StackEnv(dict):
    """The local .env with a repr that hides its values.

    pytest prints fixture values when an assertion fails, so a plain dict would
    put passwords into the terminal and into logs. Values stay usable; only the
    representation is blind.
    """

    def __repr__(self) -> str:
        return f"<StackEnv: {len(self)} keys, values hidden>"

    __str__ = __repr__


def env_values() -> StackEnv:
    """The local .env. Values are secrets: never printed, never asserted on."""
    path = SERVICIOS / ".env"
    if not path.exists():
        pytest.skip("servicios/.env is absent; copy .env.example and fill it in")
    return StackEnv(
        (line.split("=", 1)[0].strip(), line.split("=", 1)[1].strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.strip().startswith("#")
    )


def grafana_api(env: dict[str, str], path: str, body: dict | None = None, timeout_s: int = 60) -> dict:
    """One Grafana API call, authenticated as the admin from .env.

    The credentials go into a header built here and never into a URL or an
    argument list.
    """
    url = f"http://127.0.0.1:{env['GRAFANA_PORT']}{path}"
    token = base64.b64encode(
        f"{env['GRAFANA_ADMIN_USER']}:{env['GRAFANA_ADMIN_PASSWORD']}".encode()).decode()
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method="POST" if body is not None else "GET",
        headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode())
