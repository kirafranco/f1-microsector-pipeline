"""Shared reading of the F007 dashboard file.

The dashboard is a JSON document rather than Python, so the tests have to take
it apart to assert anything about it: which panels exist, which SQL they carry,
and which tables that SQL touches.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import PROJECT_ROOT
from src.warehouse.migrations import MIGRATIONS_DIR

DASHBOARD_DIR = PROJECT_ROOT / "servicios" / "grafana" / "provisioning" / "dashboards" / "json"
DASHBOARD_PATH = DASHBOARD_DIR / "pit-wall" / "pit-wall.json"

DATASOURCE_UID = "f1-postgres"
DATASOURCE_TYPE = "grafana-postgresql-datasource"

#: A CTE or derived-table name is not a table: `WITH s AS (` and `, sectors AS (`.
_CTE = re.compile(r"(?:WITH|,)\s+([a-z_][a-z0-9_]*)\s+AS\s*\(", re.IGNORECASE)
#: `FROM dim_lap`, `JOIN fact_microsector`, `CROSS JOIN s`. A derived table
#: (`FROM (`) does not match, which is what we want.
_RELATION = re.compile(r"\b(?:FROM|JOIN)\s+([a-z_][a-z0-9_]*)", re.IGNORECASE)
#: `$__timeFilter(...)`, `$__timeFrom()`, and the rest of Grafana's time macros.
TIME_MACRO = re.compile(r"\$__time\w*")


def load() -> dict:
    return json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))


def panels(dashboard: dict) -> list[dict]:
    """The real panels. Row headers are layout, not visualisations."""
    return [panel for panel in dashboard["panels"] if panel["type"] != "row"]


def panel_named(dashboard: dict, fragment: str) -> dict:
    """One panel, by exact title or by an unambiguous fragment of it.

    Exact matches win: 'Brake' is the title of the brake channel panel and also
    a word inside the pit-wall panel's title.
    """
    found = panels(dashboard)
    exact = [p for p in found if p["title"].lower() == fragment.lower()]
    if len(exact) == 1:
        return exact[0]
    matches = [p for p in found if fragment.lower() in p["title"].lower()]
    if len(matches) != 1:
        raise LookupError(f"{fragment!r} matched {len(matches)} panels")
    return matches[0]


def panel_sql(dashboard: dict) -> list[tuple[str, str]]:
    return [(panel["title"], target["rawSql"])
            for panel in panels(dashboard)
            for target in panel["targets"]]


def variables(dashboard: dict) -> dict[str, dict]:
    return {variable["name"]: variable for variable in dashboard["templating"]["list"]}


def variable_sql(dashboard: dict) -> list[tuple[str, str]]:
    return [(name, variable["query"])
            for name, variable in variables(dashboard).items()
            if variable["type"] == "query"]


def all_sql(dashboard: dict) -> list[tuple[str, str]]:
    return variable_sql(dashboard) + panel_sql(dashboard)


def relations(sql: str) -> set[str]:
    """Every relation the statement reads, minus the ones it defines itself."""
    defined = {name.lower() for name in _CTE.findall(sql)}
    return {name.lower() for name in _RELATION.findall(sql)} - defined


def schema_tables() -> set[str]:
    """The tables the migrations create.

    Read from the migration file rather than listed here, so that a dashboard
    querying something F005 does not define fails whatever the schema becomes.
    """
    found: set[str] = set()
    for migration in sorted(MIGRATIONS_DIR.glob("*.sql")):
        body = migration.read_text(encoding="utf-8")
        found |= {name.lower() for name in re.findall(r"CREATE TABLE\s+(?:IF NOT EXISTS\s+)?(\w+)", body, re.IGNORECASE)}
    return found


def substitute(sql: str, values: dict[str, object]) -> str:
    """Interpolate template variables the way Grafana does before sending SQL.

    Grafana resolves `${name:sqlstring}` to a quoted literal and a bare `$name`
    to the raw value; the tests have to do the same to run a panel's query.
    """
    out = sql
    for name, value in values.items():
        literal = "'" + str(value).replace("'", "''") + "'"
        out = out.replace(f"${{{name}:sqlstring}}", literal)
        out = out.replace(f"${{{name}}}", str(value))
        out = re.sub(rf"\${name}\b", str(value), out)
    return out


def unresolved(sql: str) -> list[str]:
    """Variable references left in a statement after substitution."""
    return [match for match in re.findall(r"\$\{?[a-z_][a-z0-9_]*", sql, re.IGNORECASE)]
