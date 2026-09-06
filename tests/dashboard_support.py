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


def resolve_like_grafana(dashboard: dict, run_query) -> dict[str, str]:
    """Resolve every variable the way Grafana's frontend does, in dependency order.

    F024 exists because nothing did this. The live suite built its own
    self-consistent combination and substituted it by hand, which proves each
    panel *can* answer -- not that the dashboard's own default state lands
    somewhere that returns rows. A stale `lap_a` from a warehouse reload
    survived exactly that gap and showed a person eighteen empty panels.

    The rules modelled here are Grafana's:

    * run the variable's query with whatever is already resolved;
    * keep a saved `current` only if it is among the returned options;
    * otherwise fall back to the first option -- unless `allowCustomValue`
      permits a value that is not in the list, which is what lets a stale id
      survive.

    ``run_query(sql)`` takes interpolated SQL and returns a list of
    ``(value, text)`` pairs, so this helper stays free of any HTTP client.
    """
    order = _dependency_order(dashboard)
    resolved: dict[str, str] = {}
    for name in order:
        variable = variables(dashboard)[name]
        if variable["type"] != "query":
            current = variable.get("current") or {}
            if current.get("value") is not None:
                resolved[name] = str(current["value"])
            continue
        statement = substitute(variable["query"], resolved)
        if unresolved(statement):
            raise AssertionError(f"{name}: cannot resolve, still needs {unresolved(statement)}")
        options = [str(value) for value, _text in run_query(statement)]
        saved = (variable.get("current") or {}).get("value")
        saved = None if saved in (None, "", []) else str(saved)
        if saved is not None and saved in options:
            resolved[name] = saved
        elif saved is not None and variable.get("allowCustomValue", True):
            # Grafana keeps a value outside the options list unless told not to.
            resolved[name] = saved
        elif options:
            resolved[name] = options[0]
        else:
            raise AssertionError(f"{name}: the variable query returned no options")
    return resolved


def _dependency_order(dashboard: dict) -> list[str]:
    """Variable names, parents before the variables that interpolate them."""
    defined = variables(dashboard)
    needs = {
        name: {other for other in defined if other != name and _mentions(v.get("query", ""), other)}
        for name, v in defined.items()
    }
    order: list[str] = []
    while len(order) < len(defined):
        ready = [n for n in defined if n not in order and needs[n] <= set(order)]
        if not ready:
            raise AssertionError(f"variables form a cycle: {needs}")
        order.extend(sorted(ready))
    return order


def _mentions(sql: str, name: str) -> bool:
    if not isinstance(sql, str):
        return False
    return bool(re.search(r"\$\{?" + re.escape(name) + r"[:}\s]|\$" + re.escape(name) + r"\b", sql))
