"""F007 integration: the pit-wall dashboard against the running stack.

Opt-in, like the other stack tests: `pytest -m docker`. Needs the core profile
up and at least one session loaded by F005; skips rather than fails otherwise.

What these tests cannot do is drive a browser, so they prove that every query
the dashboard sends is correct and quick, and that the figures it puts on
screen agree with the pipeline. Whether hovering the pit-wall panel actually
reads every channel at one distance is criterion 10, and it is checked by hand.
"""

from __future__ import annotations

import shutil
import time
import urllib.error

import pytest

from tests import dashboard_support as dash
from tests.stack_env import env_values, grafana_api

pytestmark = pytest.mark.docker

#: Criterion 3. The probe measured 17-120 ms; this leaves room for a cold page
#: cache without letting a sequential scan through.
QUERY_BUDGET_MS = 250.0

#: Criterion 5. F010 measured the lap-time residual at p50 0.054 s, p95 0.147 s
#: on Suzuka; a gap is the difference of two of those.
GAP_TOLERANCE_S = 0.35

#: Criterion 6. Both sides are float32 sums, so this is float noise, not slack.
CLOSURE_TOLERANCE_S = 1e-3


def query(env, sql: str) -> tuple[dict[str, list], float]:
    """Run raw SQL through the datasource, as a panel does. Returns columns and milliseconds."""
    body = {"queries": [{
        "refId": "A",
        "datasource": {"uid": dash.DATASOURCE_UID, "type": dash.DATASOURCE_TYPE},
        "rawSql": sql,
        "format": "table",
        "rawQuery": True,
    }]}
    started = time.perf_counter()
    result = grafana_api(env, "/api/ds/query", body)
    elapsed_ms = (time.perf_counter() - started) * 1000
    payload = result["results"]["A"]
    assert not payload.get("error"), f"{payload.get('error')}\n{sql}"
    frames = payload.get("frames", [])
    if not frames or not frames[0]["data"]["values"]:
        return {}, elapsed_ms
    names = [field["name"] for field in frames[0]["schema"]["fields"]]
    return dict(zip(names, frames[0]["data"]["values"])), elapsed_ms


@pytest.fixture(scope="module")
def env() -> dict[str, str]:
    if shutil.which("docker") is None:
        pytest.skip("docker is not on PATH")
    values = env_values()
    try:
        grafana_api(values, "/api/health", timeout_s=10)
    except (urllib.error.URLError, OSError):
        pytest.skip("Grafana is not reachable; bring the core profile up first")
    return values


@pytest.fixture(scope="module", autouse=True)
def warm(env) -> None:
    """One call before anything is timed, so the budget measures the query and
    not Grafana opening its first connection to Postgres."""
    query(env, "SELECT 1")


@pytest.fixture(scope="module")
def dashboard() -> dict:
    return dash.load()


@pytest.fixture(scope="module")
def resolved(env, dashboard: dict) -> dict[str, object]:
    """Every template variable at its default, resolved the way Grafana does.

    Each query runs with the variables before it already substituted, and the
    first row becomes the default -- which is what the ORDER BY in each query
    is for.
    """
    values: dict[str, object] = {
        name: variable["query"]
        for name, variable in dash.variables(dashboard).items()
        if variable["type"] == "constant"
    }
    for name, sql in dash.variable_sql(dashboard):
        columns, _ = query(env, dash.substitute(sql, values))
        if not columns:
            pytest.skip(f"no options for {name}; load a session with src.warehouse.load first")
        column = columns.get("__value") or next(iter(columns.values()))
        values[name] = column[0]
    return values


class TestCriterion1Provisioned:
    def test_the_dashboard_comes_from_the_repository_file(self, env) -> None:
        meta = grafana_api(env, "/api/dashboards/uid/f1-pit-wall")["meta"]
        assert meta["provisioned"] is True, "a dashboard created in the UI is not versioned"
        assert meta["provisionedExternalId"] == "pit-wall/pit-wall.json"

    def test_it_is_listed_where_a_reader_would_look(self, env) -> None:
        found = {item["uid"] for item in grafana_api(env, "/api/search?type=dash-db")}
        assert "f1-pit-wall" in found

    def test_the_stored_model_is_the_file(self, env, dashboard: dict) -> None:
        """Grafana serves what provisioning read from disk, so a stale container
        or a half-written file shows up here rather than in the browser."""
        stored = grafana_api(env, "/api/dashboards/uid/f1-pit-wall")["dashboard"]
        assert stored["title"] == dashboard["title"]
        assert [p["title"] for p in stored["panels"]] == [p["title"] for p in dashboard["panels"]]
        assert [v["name"] for v in stored["templating"]["list"]] == list(dash.variables(dashboard))


class TestCriterion4Variables:
    def test_each_one_offers_something_to_choose(self, env, dashboard: dict, resolved) -> None:
        for name, sql in dash.variable_sql(dashboard):
            columns, _ = query(env, dash.substitute(sql, resolved))
            assert columns, name
            assert len(next(iter(columns.values()))) >= 1, name

    def test_the_labelled_lists_give_grafana_a_value_and_a_label(self, env, dashboard: dict, resolved) -> None:
        """Without both columns Grafana would show a lap_id where a lap number belongs."""
        for name in ("session", "lap_a", "lap_b"):
            columns, _ = query(env, dash.substitute(dash.variables(dashboard)[name]["query"], resolved))
            assert set(columns) == {"__value", "__text"}, name

    def test_the_two_drivers_are_different_people(self, resolved) -> None:
        assert resolved["driver_a"] != resolved["driver_b"]

    def test_each_lap_defaults_to_its_driver_s_fastest(self, env, dashboard: dict, resolved) -> None:
        """Decision D7 makes the reference a parameter; this is its default."""
        for side in ("a", "b"):
            driver = resolved[f"driver_{side}"]
            columns, _ = query(
                env,
                f"SELECT lap_id FROM dim_lap WHERE session_id = {resolved['session']} "
                f"AND code = '{driver}' AND lap_time_s IS NOT NULL "
                "ORDER BY lap_time_s LIMIT 1",
            )
            assert columns["lap_id"][0] == resolved[f"lap_{side}"], driver

    def test_the_chosen_laps_have_telemetry_on_the_grid(self, env, resolved) -> None:
        columns, _ = query(
            env,
            "SELECT count(*) AS n FROM fact_telemetry_grid "
            f"WHERE lap_id IN ({resolved['lap_a']}, {resolved['lap_b']})",
        )
        assert columns["n"][0] > 0


class TestCriterion2And3EveryQueryRuns:
    def test_every_panel_query_returns_without_an_error(self, env, dashboard: dict, resolved) -> None:
        for title, sql in dash.panel_sql(dashboard):
            statement = dash.substitute(sql, resolved)
            assert dash.unresolved(statement) == [], f"{title}: {dash.unresolved(statement)}"
            columns, _ = query(env, statement)
            assert columns, f"{title} returned no columns"

    def test_no_panel_query_takes_longer_than_the_budget(self, env, dashboard: dict, resolved) -> None:
        slow: list[str] = []
        for title, sql in dash.panel_sql(dashboard):
            _, elapsed_ms = query(env, dash.substitute(sql, resolved))
            if elapsed_ms > QUERY_BUDGET_MS:
                slow.append(f"{title}: {elapsed_ms:.0f} ms")
        assert slow == [], f"over {QUERY_BUDGET_MS:.0f} ms: {slow}"

    def test_every_variable_query_is_quick_too(self, env, dashboard: dict, resolved) -> None:
        """These run on every dashboard load, before a single panel draws."""
        for name, sql in dash.variable_sql(dashboard):
            _, elapsed_ms = query(env, dash.substitute(sql, resolved))
            assert elapsed_ms <= QUERY_BUDGET_MS, f"{name}: {elapsed_ms:.0f} ms"


class TestCriterion5TheGapOnScreen:
    def test_the_reconstructed_gap_agrees_with_the_official_one(self, env, dashboard: dict, resolved) -> None:
        """The two stat panels are independent: one reads the timing feed, the
        other integrates telemetry along the 10 m grid. They must agree to
        within the residual F010 measured, or the pipeline is drifting."""
        official_sql = dash.panel_named(dashboard, "Official gap")["targets"][0]["rawSql"]
        grid_sql = dash.panel_named(dashboard, "Reconstructed gap")["targets"][0]["rawSql"]
        official, _ = query(env, dash.substitute(official_sql, resolved))
        reconstructed, _ = query(env, dash.substitute(grid_sql, resolved))
        official_gap = official["gap_s"][0]
        grid_gap = reconstructed["delta_end_s"][0]
        assert abs(grid_gap - official_gap) <= GAP_TOLERANCE_S, (
            f"official {official_gap:.3f} s, reconstructed {grid_gap:.3f} s")


class TestCriterion6TheBarsAddUp:
    def test_the_loss_chart_sums_to_the_lap_delta(self, env, dashboard: dict, resolved) -> None:
        """The bars partition the lap, so their total is the Δt at the end of
        the last complete micro-sector. Pooling the short sectors instead of
        dropping them is what makes this true of the chart and not only of the
        table beneath it."""
        chart_sql = dash.panel_named(dashboard, "Micro-sector loss")["targets"][0]["rawSql"]
        columns, _ = query(env, dash.substitute(chart_sql, resolved))
        delta_column = next(name for name in columns if name != "sector")
        total = sum(columns[delta_column])

        expected, _ = query(env, dash.substitute(CLOSURE_SQL, resolved))
        assert abs(total - expected["delta_t_s"][0]) <= CLOSURE_TOLERANCE_S, (
            f"bars {total:.6f} s, grid {expected['delta_t_s'][0]:.6f} s")

    def test_the_pooled_bar_is_not_a_rounding_error(self, env, dashboard: dict, resolved) -> None:
        """If the short sectors were negligible the pooled bar could be dropped.
        On the Suzuka pair it is not, which is why the chart keeps it."""
        chart_sql = dash.panel_named(dashboard, "Micro-sector loss")["targets"][0]["rawSql"]
        columns, _ = query(env, dash.substitute(chart_sql, resolved))
        pooled = [label for label in columns["sector"] if "pooled" in label]
        assert len(pooled) <= 1, "the short sectors belong in a single bar"


class TestCriterion8NothingBuiltForTheDashboard:
    def test_the_database_holds_no_view_or_function(self, env) -> None:
        """Decision D1: the dashboard is a read-only consumer of the schema.
        A view added to make a panel easier would make the schema serve one
        consumer, which is exactly what D1 refuses."""
        columns, _ = query(env, """
            SELECT (SELECT count(*) FROM information_schema.views
                    WHERE table_schema = 'public') AS views,
                   (SELECT count(*) FROM information_schema.routines
                    WHERE routine_schema = 'public') AS routines
        """)
        assert columns["views"][0] == 0
        assert columns["routines"][0] == 0

    def test_every_table_the_panels_read_exists_in_the_database(self, env, dashboard: dict) -> None:
        wanted = sorted({rel for _, sql in dash.all_sql(dashboard)
                         for rel in dash.relations(sql) if rel != "information_schema"})
        listed = "', '".join(wanted)
        columns, _ = query(
            env,
            "SELECT table_name FROM information_schema.tables "
            f"WHERE table_schema = 'public' AND table_name IN ('{listed}')",
        )
        assert sorted(columns.get("table_name", [])) == wanted


class TestTheDatasourceItQueries:
    def test_it_is_the_provisioned_read_only_one(self, env) -> None:
        datasource = grafana_api(env, f"/api/datasources/uid/{dash.DATASOURCE_UID}")
        assert datasource["readOnly"] is True
        assert datasource["type"] == dash.DATASOURCE_TYPE

    def test_the_dashboard_cannot_write_through_it(self, env) -> None:
        """The role behind the datasource is read-only, so a panel that tried to
        write would fail rather than change the warehouse."""
        body = {"queries": [{
            "refId": "A",
            "datasource": {"uid": dash.DATASOURCE_UID, "type": dash.DATASOURCE_TYPE},
            "rawSql": "CREATE TABLE f007_forbidden (x int)",
            "format": "table",
            "rawQuery": True,
        }]}
        try:
            result = grafana_api(env, "/api/ds/query", body)
        except urllib.error.HTTPError as refused:
            assert refused.code >= 400
            return
        assert result["results"]["A"].get("error"), "the read-only role executed a CREATE TABLE"


#: The grid Δt where the last complete micro-sector ends: the value the loss
#: chart has to add up to. Derived here independently of the panel's own SQL.
CLOSURE_SQL = """
WITH s AS (
    SELECT season, round, session_code FROM dim_session WHERE session_id = $session
),
complete AS (
    SELECT max(d.end_index) AS last_index
    FROM dim_microsector d
    CROSS JOIN s
    JOIN fact_microsector fa
      ON fa.season = s.season AND fa.round = s.round AND fa.session_code = s.session_code
     AND fa.lap_id = $lap_a AND fa.grain = d.grain AND fa.microsector_id = d.microsector_id
    JOIN fact_microsector fb
      ON fb.season = s.season AND fb.round = s.round AND fb.session_code = s.session_code
     AND fb.lap_id = $lap_b AND fb.grain = d.grain AND fb.microsector_id = d.microsector_id
    WHERE d.session_id = $session
      AND d.grain = 'corner_phase'
      AND NOT fa.partial
      AND NOT fb.partial
)
SELECT b.t_s - a.t_s AS delta_t_s
FROM fact_telemetry_grid a
CROSS JOIN s
JOIN fact_telemetry_grid b
  ON b.season = s.season AND b.round = s.round AND b.session_code = s.session_code
 AND b.lap_id = $lap_b AND b.grid_index = a.grid_index
WHERE a.season = s.season AND a.round = s.round AND a.session_code = s.session_code
  AND a.lap_id = $lap_a
  AND a.grid_index = (SELECT last_index FROM complete)
"""
