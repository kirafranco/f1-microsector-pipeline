"""F017: the pit wall, checked without a database or a container.

The service is deliberately shaped so almost all of it is testable here: the
figure is a pure function of rows, the SQL is inspectable text, and the routes
are thin enough to drive with a stub pool. Only the live suite needs Docker.
"""

from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from src.pitwall import app as app_module
from src.pitwall import figure as figure_module
from src.pitwall import queries
from src.warehouse.connection import ROLE_KEYS, Settings, SettingsError

DISTANCE = [0.0, 10.0, 20.0, 30.0, 40.0]


def designed_overlay() -> queries.Overlay:
    """Five grid points where every channel does something checkable."""
    rows = [
        # distance, Δt,  spd_a, spd_b, thr_a, thr_b, brk_a, brk_b, gear_a, gear_b, drs_a, drs_b
        (0.0, 0.000, 300.0, 300.0, 100.0, 100.0, 0, 0, 8, 8, 0, 0),
        (10.0, 0.050, 290.0, 295.0, 80.0, 100.0, 0, 0, 8, 8, 0, 12),
        (20.0, 0.120, 210.0, 250.0, 0.0, 40.0, 1, 0, 5, 7, 0, 12),
        (30.0, 0.090, 150.0, 160.0, 0.0, 0.0, 1, 1, 3, 3, 0, 0),
        (40.0, -0.030, 180.0, 175.0, 60.0, 55.0, 0, 0, 4, 4, 0, 0),
    ]
    return queries.Overlay(
        session_id=39,
        lap_a=queries.Lap(101, 15, 82.595, "SOFT", 1, "NOR"),
        lap_b=queries.Lap(202, 15, 82.804, "SOFT", 1, "PIA"),
        rows=rows,
    )


class TestTheOverlayRows:
    def test_columns_match_the_select_list(self) -> None:
        """The dataclass names the columns and `column()` indexes by that name,
        so a column added to the SQL and not to the list reads the wrong one.

        Every selected column is explicitly aliased, which is what makes this
        a one-line check rather than a SQL parser.
        """
        select = queries.OVERLAY[queries.OVERLAY.index("SELECT"):queries.OVERLAY.index("FROM")]
        assert re.findall(r"AS (\w+)", select) == designed_overlay().columns

    def test_a_column_can_be_read_by_name(self) -> None:
        assert designed_overlay().column("distance_m") == DISTANCE

    def test_a_lap_labels_itself_with_its_time(self) -> None:
        assert designed_overlay().lap_a.label == "NOR L15 · 82.595 s"

    def test_a_lap_without_a_time_says_so_rather_than_crashing(self) -> None:
        lap = queries.Lap(1, 3, None, None, None, "HUL")
        assert lap.label == "HUL L3 · no time"


class TestTheSqlLetsPostgresPrune:
    """F023's finding, applied to the second consumer that reads the same facts.

    A partition key arriving through a joined CTE makes Postgres scan all 96
    partitions -- 259 MB per dashboard refresh, before F023. These statements
    must never reintroduce it, and this is checked structurally so a new query
    written the old way fails here rather than in production.
    """

    def test_every_statement_that_reads_a_partitioned_fact_is_declared(self) -> None:
        partitioned = {name for name, sql in queries.statements()
                       if "fact_telemetry_grid" in sql or "fact_microsector" in sql}
        assert partitioned == set(queries.PARTITIONED_STATEMENTS)

    @pytest.mark.parametrize("name,sql", list(queries.PARTITIONED_STATEMENTS.items()))
    def test_it_takes_no_partition_key_from_a_joined_cte(self, name: str, sql: str) -> None:
        assert "CROSS JOIN s" not in sql
        assert not re.search(r"\bJOIN\s+s\s+ON\b", sql), name

    @pytest.mark.parametrize("name,sql", list(queries.PARTITIONED_STATEMENTS.items()))
    def test_each_partition_key_is_a_constant_or_a_sibling(self, name: str, sql: str) -> None:
        for key in ("season", "round", "session_code"):
            for match in re.finditer(rf"\.{key}\s*=\s*(\S+)", sql):
                right = match.group(1)
                assert right.startswith("(SELECT") or re.fullmatch(r"\w+\.\w+", right), f"{name}: {key} = {right}"

    @pytest.mark.parametrize("name,sql", list(queries.PARTITIONED_STATEMENTS.items()))
    def test_it_anchors_on_dim_session(self, name: str, sql: str) -> None:
        assert "FROM dim_session WHERE session_id" in sql, name

    def test_no_statement_interpolates_a_value(self) -> None:
        """Parameters, always: this service takes input from a query string."""
        for name, sql in queries.statements():
            assert "%s" not in sql, f"{name}: positional parameters are ambiguous here"
            assert not re.search(r"=\s*['\"]?\{", sql), f"{name}: looks interpolated"

    def test_a_drivers_default_lap_is_their_fastest_and_a_no_time_lap_sorts_last(self) -> None:
        assert "ORDER BY lap_time_s NULLS LAST" in queries.FASTEST_LAP


class TestTheFigureIsTheInteraction:
    """D1 chose Grafana for a dashboard-wide crosshair and its Trend panel has
    none. These four properties are the thing this service exists to add."""

    @staticmethod
    @pytest.fixture(scope="class")
    def built():
        return json.loads(figure_module.build(designed_overlay()).to_json())

    def test_every_x_axis_is_linked_to_one(self, built: dict) -> None:
        axes = {k: v for k, v in built["layout"].items() if k.startswith("xaxis")}
        assert len(axes) == len(figure_module.PANELS)
        anchored = [v.get("matches") for v in axes.values() if v.get("matches")]
        assert len(anchored) == len(axes) - 1, "the panels do not pan and zoom together"
        assert len(set(anchored)) == 1, "the panels are linked to different axes"

    def test_the_tooltip_reads_every_channel_at_one_distance(self, built: dict) -> None:
        assert built["layout"]["hovermode"] == "x unified"

    def test_the_crosshair_is_drawn_across_all_the_panels(self, built: dict) -> None:
        axes = [v for k, v in built["layout"].items() if k.startswith("xaxis")]
        assert all(a.get("showspikes") and a.get("spikemode") == "across" for a in axes)

    def test_only_the_bottom_panel_labels_the_axis_and_it_says_metres(self, built: dict) -> None:
        titles = [v.get("title", {}).get("text") for k, v in built["layout"].items() if k.startswith("xaxis")]
        assert [t for t in titles if t] == ["distance (m)"]

    def test_discrete_channels_are_drawn_as_steps(self, built: dict) -> None:
        """Gear, brake and DRS are never interpolated upstream; a sloped line
        here would draw a gear change that did not happen."""
        stepped = {name for name, _, _, step in
                   ((t, a, b, s) for t, a, b, s in figure_module.PANELS) if step}
        assert stepped == {"Brake", "Gear", "DRS"}
        for trace in built["data"]:
            channel = trace["name"].split(" ", 1)[-1] if " " in trace["name"] else trace["name"]
            if channel in {"brake", "gear", "drs"}:
                assert trace["line"]["shape"] == "hv", trace["name"]

    def test_continuous_channels_are_not_stepped(self, built: dict) -> None:
        for trace in built["data"]:
            if trace["name"].endswith(("speed", "throttle")):
                assert trace["line"]["shape"] == "linear", trace["name"]

    def test_each_driver_keeps_one_colour_across_every_panel(self, built: dict) -> None:
        colours: dict[str, set[str]] = {}
        for trace in built["data"]:
            if trace["name"] == "Δt":
                continue
            colours.setdefault(trace["name"].split(" ")[0], set()).add(trace["line"]["color"])
        assert colours == {"NOR": {figure_module.COLOUR_A}, "PIA": {figure_module.COLOUR_B}}

    def test_the_legend_lists_each_driver_once(self, built: dict) -> None:
        shown = [t["name"] for t in built["data"] if t.get("showlegend")]
        assert len(shown) == 2, shown

    def test_the_delta_belongs_to_neither_car_and_is_filled_to_zero(self, built: dict) -> None:
        delta = next(t for t in built["data"] if t["name"] == "Δt")
        assert delta["fill"] == "tozeroy"
        assert delta["line"]["color"] == figure_module.COLOUR_DELTA
        assert delta["y"] == [0.0, 0.05, 0.12, 0.09, -0.03]

    def test_the_title_names_both_laps_and_their_times(self, built: dict) -> None:
        assert built["layout"]["title"]["text"] == "NOR L15 · 82.595 s    vs    PIA L15 · 82.804 s"

    def test_it_draws_the_rows_it_was_given(self, built: dict) -> None:
        speed_a = next(t for t in built["data"] if t["name"] == "NOR speed")
        assert speed_a["x"] == DISTANCE
        assert speed_a["y"] == [300.0, 290.0, 210.0, 150.0, 180.0]


class TestTheHtmlPage:
    def test_it_loads_plotly_from_this_service_and_not_a_cdn(self) -> None:
        """Global 4.0: no unvetted executable fetched at runtime, and a pit
        wall must work with no internet."""
        html = figure_module.to_html(designed_overlay(), app_module.SCRIPT_PATH)
        assert app_module.SCRIPT_PATH in html
        assert "cdn.plot.ly" not in html and "unpkg.com" not in html
        assert not re.search(r'src="https?://', html), "the page fetches something external"


class StubPool:
    """Stands in for the connection pool, so the routes are testable offline."""

    def __init__(self, overlay: queries.Overlay | None = None) -> None:
        self.overlay = overlay or designed_overlay()
        self.calls: list[str] = []


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    pool = StubPool()
    monkeypatch.setattr(app_module, "get_pool", lambda: pool)
    app_module.app.dependency_overrides[app_module.get_pool] = lambda: pool
    monkeypatch.setattr(queries, "sessions", lambda p: [(39, "2024 R24 Q — Abu Dhabi Grand Prix")])
    monkeypatch.setattr(queries, "drivers", lambda p, s: ["NOR", "PIA", "VER"])

    def fake_overlay(p, session_id, driver_a, driver_b, lap_a=None, lap_b=None):
        if driver_a == driver_b:
            raise queries.PitWallError("pick two different drivers")
        return pool.overlay

    monkeypatch.setattr(queries, "overlay", fake_overlay)
    with TestClient(app_module.app) as test_client:
        yield test_client
    app_module.app.dependency_overrides.clear()


class TestTheRoutes:
    def test_health_answers_without_touching_a_table(self, client: TestClient) -> None:
        assert client.get("/health").json() == {"ok": True}

    def test_the_script_is_served_by_this_service(self, client: TestClient) -> None:
        response = client.get(app_module.SCRIPT_PATH)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/javascript")
        assert "immutable" in response.headers["cache-control"]
        assert len(response.content) > 1_000_000, "that is not the plotly bundle"

    def test_the_index_offers_the_loaded_sessions(self, client: TestClient) -> None:
        body = client.get("/").text
        assert 'value="39"' in body and "Abu Dhabi Grand Prix" in body

    def test_the_pit_wall_page_renders(self, client: TestClient) -> None:
        response = client.get("/pitwall", params={"session_id": 39, "driver_a": "NOR", "driver_b": "PIA"})
        assert response.status_code == 200
        assert response.headers["x-rows"] == "5"
        assert "plotly" in response.text.lower()

    def test_the_json_carries_the_rows_as_well_as_the_figure(self, client: TestClient) -> None:
        """The rows are what makes cross-consumer equivalence checkable."""
        body = client.get("/api/overlay.json",
                          params={"session_id": 39, "driver_a": "NOR", "driver_b": "PIA"}).json()
        assert body["columns"] == designed_overlay().columns
        assert body["rows"][0] == [0.0, 0.0, 300.0, 300.0, 100.0, 100.0, 0, 0, 8, 8, 0, 0]
        assert body["figure"]["layout"]["hovermode"] == "x unified"

    def test_asking_for_one_driver_twice_is_a_404_not_a_500(self, client: TestClient) -> None:
        response = client.get("/pitwall", params={"session_id": 39, "driver_a": "NOR", "driver_b": "NOR"})
        assert response.status_code == 404
        assert "two different drivers" in response.json()["detail"]

    def test_a_missing_parameter_is_refused(self, client: TestClient) -> None:
        assert client.get("/pitwall", params={"session_id": 39}).status_code == 422

    def test_a_nonsense_session_is_refused_before_it_reaches_sql(self, client: TestClient) -> None:
        response = client.get("/api/drivers", params={"session_id": -1})
        assert response.status_code == 422


class TestItConnectsAsTheReadOnlyRole:
    def test_the_role_names_the_read_only_credentials(self) -> None:
        assert ROLE_KEYS["readonly"][:2] == ("POSTGRES_READONLY_USER", "POSTGRES_READONLY_PASSWORD")

    def test_settings_read_the_read_only_user(self, tmp_path) -> None:
        env = {"POSTGRES_READONLY_USER": "f1_readonly", "POSTGRES_READONLY_PASSWORD": "x",
               "POSTGRES_DB": "f1_microsector", "POSTGRES_HOST": "postgres"}
        settings = Settings.from_env(tmp_path / "absent.env", env, role="readonly")
        assert settings.user == "f1_readonly"
        assert "password" not in settings.safe_dsn

    def test_a_missing_read_only_password_does_not_fall_back_to_admin(self, tmp_path) -> None:
        """A consumer that asked to be unable to write must not silently gain it."""
        env = {"POSTGRES_USER": "f1_admin", "POSTGRES_PASSWORD": "admin-secret",
               "POSTGRES_DB": "f1_microsector"}
        with pytest.raises(SettingsError, match="POSTGRES_READONLY_PASSWORD"):
            Settings.from_env(tmp_path / "absent.env", env, role="readonly")

    def test_an_unknown_role_is_refused(self, tmp_path) -> None:
        with pytest.raises(SettingsError, match="unknown role"):
            Settings.from_env(tmp_path / "absent.env", {"POSTGRES_PASSWORD": "x"}, role="root")

    def test_admin_is_still_the_default(self, tmp_path) -> None:
        """Every existing caller keeps the behaviour it had."""
        env = {"POSTGRES_USER": "f1_admin", "POSTGRES_PASSWORD": "x", "POSTGRES_DB": "d"}
        assert Settings.from_env(tmp_path / "absent.env", env).user == "f1_admin"

    def test_the_pool_is_small_because_a_pit_wall_has_one_viewer(self) -> None:
        assert queries.POOL_MIN_SIZE == 1 and queries.POOL_MAX_SIZE <= 4
