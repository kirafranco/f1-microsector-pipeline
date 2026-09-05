"""F007: what the pit-wall dashboard file must say, checked without a stack.

These are the offline half of the acceptance criteria. The half that needs a
running Grafana and a loaded session lives in test_dashboard_live.py.
"""

from __future__ import annotations

import json

import pytest

from tests import dashboard_support as dash


@pytest.fixture(scope="module")
def dashboard() -> dict:
    return dash.load()


class TestTheFile:
    def test_it_sits_where_the_provider_looks(self) -> None:
        """F001 points the file provider at this directory; nothing else loads it."""
        assert dash.DASHBOARD_PATH.exists()
        assert dash.DASHBOARD_PATH.parent.parent == dash.DASHBOARD_DIR

    def test_it_is_valid_json_with_a_fixed_identity(self, dashboard: dict) -> None:
        """The uid is fixed so F016 can link to it and a redeploy updates rather than duplicates."""
        assert dashboard["uid"] == "f1-pit-wall"
        assert dashboard["title"]
        assert "f1" in dashboard["tags"]

    def test_it_is_read_only_in_the_browser(self, dashboard: dict) -> None:
        """allowUiUpdates is false, so a UI edit would be discarded on the next reload anyway."""
        assert dashboard["editable"] is False

    def test_every_panel_explains_itself(self, dashboard: dict) -> None:
        """A pit-wall figure without its caveat is worse than no figure."""
        undocumented = [p["title"] for p in dash.panels(dashboard) if not p.get("description")]
        assert undocumented == []

    def test_panel_ids_are_unique(self, dashboard: dict) -> None:
        ids = [panel["id"] for panel in dashboard["panels"]]
        assert len(ids) == len(set(ids))


class TestCriterion7NoTimeAxis:
    def test_the_time_picker_is_hidden(self, dashboard: dict) -> None:
        assert dashboard["timepicker"]["hidden"] is True

    def test_no_query_uses_a_time_macro(self, dashboard: dict) -> None:
        """The x-axis is distance. A time macro would silently filter the lap."""
        offenders = [name for name, sql in dash.all_sql(dashboard) if dash.TIME_MACRO.search(sql)]
        assert offenders == []

    def test_the_dashboard_does_not_promise_a_shared_crosshair(self, dashboard: dict) -> None:
        """Grafana's Trend visualisation ignores graphTooltip: its documentation
        lists 'no shared cursor/crosshair' among the ways it differs from a time
        series panel. Setting 1 here would look like synchronisation the panels
        cannot deliver, so the pit-wall panel carries every channel instead.
        """
        assert dashboard["graphTooltip"] == 0


class TestCriterion7Datasource:
    def test_every_panel_names_the_provisioned_datasource(self, dashboard: dict) -> None:
        for panel in dash.panels(dashboard):
            assert panel["datasource"]["uid"] == dash.DATASOURCE_UID, panel["title"]
            assert panel["datasource"]["type"] == dash.DATASOURCE_TYPE, panel["title"]
            for target in panel["targets"]:
                assert target["datasource"]["uid"] == dash.DATASOURCE_UID, panel["title"]

    def test_every_variable_names_it_too(self, dashboard: dict) -> None:
        for name, variable in dash.variables(dashboard).items():
            if variable["type"] == "query":
                assert variable["datasource"]["uid"] == dash.DATASOURCE_UID, name


class TestCriterion8OnlyTheSchema:
    def test_every_relation_read_is_a_table_the_migrations_create(self, dashboard: dict) -> None:
        """Decision D1: the dashboard adapts to the schema, never the reverse.
        A view created for this dashboard would show up here as an unknown name.
        """
        known = dash.schema_tables() | {"information_schema"}
        for name, sql in dash.all_sql(dashboard):
            unknown = {rel for rel in dash.relations(sql) if rel not in known}
            assert unknown == set(), f"{name} reads {sorted(unknown)}"

    def test_the_schema_it_checks_against_is_not_empty(self) -> None:
        """Guards the test above: an empty table set would let anything pass."""
        assert "fact_telemetry_grid" in dash.schema_tables()


class TestThePitWallPanel:
    """Option A: one Trend panel carrying every channel, because stacked panels
    cannot share a crosshair. This is the panel the whole decision turned on."""

    def test_it_is_a_trend_panel_on_a_distance_axis(self, dashboard: dict) -> None:
        panel = dash.panel_named(dashboard, "Δt, speed, throttle and brake")
        assert panel["type"] == "trend"
        assert panel["options"]["xField"] == "distance_m"

    def test_its_tooltip_reads_every_series_at_once(self, dashboard: dict) -> None:
        """'All' in the UI is 'multi' in the file: without it, hovering reads one line."""
        panel = dash.panel_named(dashboard, "Δt, speed, throttle and brake")
        assert panel["options"]["tooltip"]["mode"] == "multi"

    def test_it_carries_delta_speed_throttle_and_brake_for_both_laps(self, dashboard: dict) -> None:
        panel = dash.panel_named(dashboard, "Δt, speed, throttle and brake")
        sql = panel["targets"][0]["rawSql"]
        for series in ('"Δt (B−A)"', '"speed A"', '"speed B"', '"throttle A"',
                       '"throttle B"', '"brake A"', '"brake B"'):
            assert series in sql, series

    def test_each_series_is_given_an_axis_on_purpose(self, dashboard: dict) -> None:
        """Δt left, speed right, the flags hidden: eight series on one pair of
        auto-scaled axes would be unreadable."""
        panel = dash.panel_named(dashboard, "Δt, speed, throttle and brake")
        placement = {
            override["matcher"]["options"]: prop["value"]
            for override in panel["fieldConfig"]["overrides"]
            for prop in override["properties"]
            if prop["id"] == "custom.axisPlacement"
        }
        assert placement["Δt (B−A)"] == "left"
        assert placement["speed A"] == placement["speed B"] == "right"
        for hidden in ("throttle A", "throttle B", "brake A", "brake B"):
            assert placement[hidden] == "hidden", hidden


class TestDiscreteChannels:
    """Project rule: nGear, Brake and DRS are never linearly interpolated.
    Drawing a straight line between two samples of a flag invents a state the
    car was never in, so the panels step instead."""

    STEPPED = ("Brake", "Gear", "DRS")

    @pytest.mark.parametrize("title", STEPPED)
    def test_the_channel_panel_steps(self, dashboard: dict, title: str) -> None:
        panel = dash.panel_named(dashboard, title)
        assert panel["fieldConfig"]["defaults"]["custom"]["lineInterpolation"] == "stepAfter"

    def test_the_brake_traces_on_the_pit_wall_panel_step_too(self, dashboard: dict) -> None:
        panel = dash.panel_named(dashboard, "Δt, speed, throttle and brake")
        stepped = {
            override["matcher"]["options"]
            for override in panel["fieldConfig"]["overrides"]
            for prop in override["properties"]
            if prop["id"] == "custom.lineInterpolation" and prop["value"] == "stepAfter"
        }
        assert stepped == {"brake A", "brake B"}


class TestCriterion9Uncertainty:
    def test_each_braking_point_carries_its_own_spacing(self, dashboard: dict) -> None:
        """Decision D6: a braking point interpolated across a 40 m gap is not
        the same measurement as one taken between samples 8 m apart."""
        sql = dash.panel_named(dashboard, "Corner by corner")["targets"][0]["rawSql"]
        assert sql.count("brake_gap_m") == 2, "one ± per lap, not a session-wide figure"
        assert "' m ± '" in sql

    def test_the_delta_column_names_the_window_below_which_it_means_nothing(self, dashboard: dict) -> None:
        sql = dash.panel_named(dashboard, "Corner by corner")["targets"][0]["rawSql"]
        assert "± 20 m" in sql and "D6" in sql

    def test_thin_telemetry_at_the_line_is_flagged_next_to_the_lap_times(self, dashboard: dict) -> None:
        """F010: a lap whose telemetry was sparse at the timing line has a less
        trustworthy Δt at both ends."""
        sql = dash.panel_named(dashboard, "Tyres")["targets"][0]["rawSql"]
        assert "start_coverage_poor" in sql and "end_coverage_poor" in sql


class TestTheLossChart:
    def test_short_sectors_are_pooled_rather_than_dropped(self, dashboard: dict) -> None:
        """A sector under the threshold is noise on its own but not collectively:
        on the Suzuka pair the three short ones carry -0.104 s against the ranked
        +0.184 s. Dropping them would leave bars that do not add up to the lap.
        """
        sql = dash.panel_named(dashboard, "Micro-sector loss")["targets"][0]["rawSql"]
        assert "UNION ALL" in sql
        assert "sectors pooled" in sql
        assert ">= $min_sector_m" in sql and "< $min_sector_m" in sql

    def test_it_ranks_only_complete_sectors(self, dashboard: dict) -> None:
        """A sector the lap ended inside has no comparable time."""
        sql = dash.panel_named(dashboard, "Micro-sector loss")["targets"][0]["rawSql"]
        assert sql.count("NOT fa.partial") == 2 and sql.count("NOT fb.partial") == 2

    def test_the_table_below_it_totals_the_delta_column(self, dashboard: dict) -> None:
        """The closure is visible on screen, not only in a test."""
        panel = dash.panel_named(dashboard, "Every micro-sector")
        footer = panel["options"]["footer"]
        assert footer["show"] is True
        assert footer["reducer"] == ["sum"]
        assert footer["fields"] == ["Δ B−A s"]


class TestVariables:
    def test_the_designed_set_is_present(self, dashboard: dict) -> None:
        assert set(dash.variables(dashboard)) == {
            "session", "driver_a", "driver_b", "lap_a", "lap_b", "min_sector_m"}

    def test_a_lap_is_never_compared_with_itself(self, dashboard: dict) -> None:
        """Driver B's options exclude whoever is selected as A."""
        assert "code <> ${driver_a:sqlstring}" in dash.variables(dashboard)["driver_b"]["query"]

    def test_each_lap_list_belongs_to_its_own_driver(self, dashboard: dict) -> None:
        for side in ("a", "b"):
            query = dash.variables(dashboard)[f"lap_{side}"]["query"]
            assert f"code = ${{driver_{side}:sqlstring}}" in query

    def test_the_lap_lists_default_to_the_fastest(self, dashboard: dict) -> None:
        """Decision D7 makes the reference a parameter; the default is the one
        anybody would pick first."""
        for side in ("a", "b"):
            assert "ORDER BY lap_time_s NULLS LAST" in dash.variables(dashboard)[f"lap_{side}"]["query"]

    def test_the_drivers_are_ordered_by_their_best_lap(self, dashboard: dict) -> None:
        for side in ("a", "b"):
            assert "ORDER BY min(lap_time_s)" in dash.variables(dashboard)[f"driver_{side}"]["query"]

    def test_the_noise_threshold_is_the_one_f004_measured(self, dashboard: dict) -> None:
        threshold = dash.variables(dashboard)["min_sector_m"]
        assert threshold["type"] == "constant"
        assert threshold["query"] == "30"

    def test_the_option_lists_are_built_by_sql_not_typed_in(self, dashboard: dict) -> None:
        """A hardcoded driver list would go stale the moment another session loads."""
        for name, variable in dash.variables(dashboard).items():
            if variable["type"] == "query":
                assert variable["options"] == [], name
                assert variable["refresh"] == 1, name


class TestSubstitution:
    """The helper the live tests rely on, so a green live run means something."""

    def test_it_resolves_every_form_grafana_writes(self) -> None:
        sql = "WHERE session_id = $session AND code = ${driver_a:sqlstring} AND n >= ${min_sector_m}"
        out = dash.substitute(sql, {"session": 2, "driver_a": "VER", "min_sector_m": 30})
        assert out == "WHERE session_id = 2 AND code = 'VER' AND n >= 30"
        assert dash.unresolved(out) == []

    def test_it_does_not_confuse_a_prefix_for_a_variable(self) -> None:
        out = dash.substitute("$lap_a $lap_b", {"lap_a": 1, "lap_b": 2})
        assert out == "1 2"

    def test_it_escapes_a_quote_rather_than_breaking_the_statement(self) -> None:
        assert dash.substitute("= ${d:sqlstring}", {"d": "O'C"}) == "= 'O''C'"


class TestRelationParsing:
    """The criterion 8 test is only as good as this."""

    def test_a_cte_is_not_mistaken_for_a_table(self) -> None:
        sql = "WITH s AS (SELECT 1), t AS (SELECT 2) SELECT * FROM s CROSS JOIN t JOIN dim_lap l ON true"
        assert dash.relations(sql) == {"dim_lap"}

    def test_a_derived_table_is_not_either(self) -> None:
        assert dash.relations("SELECT * FROM (SELECT * FROM dim_lap) x") == {"dim_lap"}

    def test_it_finds_what_it_should(self) -> None:
        assert dash.relations("SELECT * FROM fact_microsector f JOIN dim_lap l ON true") == {
            "fact_microsector", "dim_lap"}


class TestItStaysLoadable:
    def test_the_file_round_trips(self, dashboard: dict) -> None:
        """Grafana rejects the whole dashboard on a malformed document."""
        assert json.loads(json.dumps(dashboard))["uid"] == "f1-pit-wall"

    def test_no_panel_query_is_left_holding_an_unknown_variable(self, dashboard: dict) -> None:
        """Every $name in the SQL must be one the dashboard defines, or Grafana
        sends the literal text to Postgres."""
        declared = set(dash.variables(dashboard))
        for name, sql in dash.all_sql(dashboard):
            used = {ref.lstrip("${") for ref in dash.unresolved(sql)}
            assert used <= declared, f"{name} uses {sorted(used - declared)}"
