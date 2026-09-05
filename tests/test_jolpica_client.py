"""F012 client: cache first, page correctly, pace itself, and never call absence success."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.reference.jolpica import (
    PAGE_SIZE,
    JolpicaClient,
    JolpicaEmptyError,
    JolpicaError,
    cache_key,
)

FIXTURES = Path(__file__).parent / "fixtures" / "jolpica"


def payload(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


class FakeResponse:
    def __init__(self, body: dict, status: int = 200) -> None:
        self._body = body
        self.status_code = status

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests_error(self.status_code)
            raise error


def requests_error(status: int) -> Exception:
    """An exception shaped like requests', so retry.is_transient can read it."""
    error = RuntimeError(f"HTTP {status}")
    error.response = type("R", (), {"status_code": status})()
    return error


class FakeSession:
    """Records calls and returns queued payloads."""

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": headers})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item  # a connection error: nothing came back at all
        if isinstance(item, FakeResponse):
            return item  # an HTTP error status: a response did come back
        return FakeResponse(item)


class ExplodingSession:
    """Fails the test if the network is touched at all."""

    def get(self, *args, **kwargs):
        raise AssertionError("network was used when the cache should have answered")


def client_for(tmp_path: Path, session, **kwargs) -> JolpicaClient:
    return JolpicaClient(cache_dir=tmp_path / "cache", session=session, sleep=lambda _: None, **kwargs)


class TestCache:
    def test_first_call_fetches_and_stores(self, tmp_path: Path) -> None:
        session = FakeSession([payload("drivers_2024")])
        client = client_for(tmp_path, session)
        client.get("2024/drivers", limit=100)
        assert client.requests_made == 1 and client.cache_hits == 0
        assert len(list((tmp_path / "cache").glob("*.json"))) == 1

    def test_second_call_never_touches_the_network(self, tmp_path: Path) -> None:
        client_for(tmp_path, FakeSession([payload("drivers_2024")])).get("2024/drivers", limit=100)
        offline = client_for(tmp_path, ExplodingSession())
        result = offline.get("2024/drivers", limit=100)
        assert offline.cache_hits == 1 and offline.requests_made == 0
        assert result["MRData"]["DriverTable"]["Drivers"][0]["code"] == "ALB"

    def test_refresh_bypasses_the_cache(self, tmp_path: Path) -> None:
        session = FakeSession([payload("drivers_2024"), payload("drivers_2024")])
        client = client_for(tmp_path, session)
        client.get("2024/drivers", limit=100)
        client.get("2024/drivers", limit=100, refresh=True)
        assert client.requests_made == 2

    def test_cached_file_records_the_url_and_time(self, tmp_path: Path) -> None:
        client = client_for(tmp_path, FakeSession([payload("drivers_2024")]))
        client.get("2024/drivers", limit=100)
        document = json.loads(next((tmp_path / "cache").glob("*.json")).read_text(encoding="utf-8"))
        assert document["url"].endswith("2024/drivers.json")
        assert document["fetched_at"].endswith("Z")
        assert "MRData" in document["payload"]

    def test_an_unreadable_cache_entry_is_refetched(self, tmp_path: Path) -> None:
        client = client_for(tmp_path, FakeSession([payload("drivers_2024")]))
        client.get("2024/drivers", limit=100)
        next((tmp_path / "cache").glob("*.json")).write_text("not json", encoding="utf-8")
        again = client_for(tmp_path, FakeSession([payload("drivers_2024")]))
        again.get("2024/drivers", limit=100)
        assert again.requests_made == 1

    def test_cache_key_ignores_parameter_order(self) -> None:
        assert cache_key("2024/drivers", {"limit": 100, "offset": 0}) == cache_key("2024/drivers", {"offset": 0, "limit": 100})
        assert cache_key("2024/drivers", {"limit": 100}) != cache_key("2024/drivers", {"limit": 30})
        assert cache_key("2024/drivers", {}) != cache_key("2025/drivers", {})


class TestPaging:
    def _page(self, total: int, rows: list[dict]) -> dict:
        return {"MRData": {"total": str(total), "DriverTable": {"Drivers": rows}}}

    def test_a_single_page_is_returned_whole(self, tmp_path: Path) -> None:
        client = client_for(tmp_path, FakeSession([payload("drivers_2024")]))
        rows = client.get_all("2024/drivers", "DriverTable", "Drivers")
        assert len(rows) == 25

    def test_several_pages_are_concatenated(self, tmp_path: Path) -> None:
        first = self._page(PAGE_SIZE + 3, [{"driverId": f"d{i}"} for i in range(PAGE_SIZE)])
        second = self._page(PAGE_SIZE + 3, [{"driverId": f"d{i}"} for i in range(PAGE_SIZE, PAGE_SIZE + 3)])
        client = client_for(tmp_path, FakeSession([first, second]))
        rows = client.get_all("2024/drivers", "DriverTable", "Drivers")
        assert len(rows) == PAGE_SIZE + 3
        assert client.requests_made == 2

    def test_nested_results_are_merged_by_round_not_repeated(self, tmp_path: Path) -> None:
        """Qualifying totals count entries, not races: paging naively would
        return the same race twice with different slices of its results."""
        def race(entries: list[dict]) -> dict:
            return {"season": "2024", "round": "4", "QualifyingResults": entries}

        first = {"MRData": {"total": "150", "RaceTable": {"Races": [race([{"position": str(i)} for i in range(100)])]}}}
        second = {"MRData": {"total": "150", "RaceTable": {"Races": [race([{"position": str(i)} for i in range(100, 150)])]}}}
        client = client_for(tmp_path, FakeSession([first, second]))
        rows = client.get_all("2024/4/qualifying", "RaceTable", "Races", nested_key="QualifyingResults")
        assert len(rows) == 1, "one race, not one per page"
        assert len(rows[0]["QualifyingResults"]) == 150

    def test_a_real_qualifying_payload_keeps_its_twenty_entries(self, tmp_path: Path) -> None:
        client = client_for(tmp_path, FakeSession([payload("qualifying_2024_4")]))
        rows = client.qualifying(2024, 4)
        assert len(rows) == 1
        assert len(rows[0]["QualifyingResults"]) == 20
        assert client.requests_made == 1, "20 entries fit one page"

    def test_an_empty_total_raises_rather_than_returning_nothing(self, tmp_path: Path) -> None:
        client = client_for(tmp_path, FakeSession([payload("empty")]))
        with pytest.raises(JolpicaEmptyError, match="no rows"):
            client.get_all("1800/races", "RaceTable", "Races")

    def test_a_missing_table_is_an_error(self, tmp_path: Path) -> None:
        broken = {"MRData": {"total": "1"}}
        client = client_for(tmp_path, FakeSession([broken]))
        with pytest.raises(JolpicaError, match="no RaceTable"):
            client.get_all("2024/races", "RaceTable", "Races")

    def test_a_response_without_the_envelope_is_an_error(self, tmp_path: Path) -> None:
        client = client_for(tmp_path, FakeSession([{"unexpected": True}]))
        with pytest.raises(JolpicaError, match="MRData"):
            client.get("2024/races")


class TestPacingAndRetries:
    def test_live_requests_are_paced(self, tmp_path: Path) -> None:
        slept: list[float] = []
        # Consumed in order: after request 1, the pace check before request 2,
        # then after request 2. The gap the client sees is 0.05 s.
        ticks = iter([0.0, 0.05, 0.05, 10.0, 10.0])
        client = JolpicaClient(
            cache_dir=tmp_path / "cache",
            session=FakeSession([payload("drivers_2024"), payload("constructors_2024")]),
            sleep=slept.append,
            clock=lambda: next(ticks),
            min_interval_s=0.25,
        )
        client.get("2024/drivers", limit=1)
        client.get("2024/constructors", limit=1)
        assert slept and slept[0] == pytest.approx(0.20, abs=1e-6)

    def test_a_rate_limit_is_retried_then_succeeds(self, tmp_path: Path) -> None:
        """A 429 is a request the API already charged us for, so it counts."""
        session = FakeSession([FakeResponse({}, status=429), payload("drivers_2024")])
        client = client_for(tmp_path, session)
        result = client.get("2024/drivers", limit=100)
        assert client.requests_made == 2
        assert result["MRData"]["total"] == "25"

    def test_a_connection_error_is_retried_and_not_counted(self, tmp_path: Path) -> None:
        """Nothing reached the API, so nothing is charged against the budget."""
        session = FakeSession([ConnectionError("dropped"), payload("drivers_2024")])
        client = client_for(tmp_path, session)
        client.get("2024/drivers", limit=100)
        assert client.requests_made == 1
        assert len(session.calls) == 2

    def test_a_permanent_status_is_not_retried(self, tmp_path: Path) -> None:
        session = FakeSession([FakeResponse({}, status=404)])
        client = client_for(tmp_path, session)
        with pytest.raises(Exception, match="404"):
            client.get("2024/nope", limit=100)
        assert len(session.calls) == 1

    def test_an_empty_result_is_not_retried(self, tmp_path: Path) -> None:
        """JolpicaEmptyError is permanent: five attempts would be pointless."""
        session = FakeSession([payload("empty")])
        client = client_for(tmp_path, session)
        with pytest.raises(JolpicaEmptyError):
            client.get_all("1800/races", "RaceTable", "Races")
        assert client.requests_made == 1

    def test_the_user_agent_identifies_the_project(self, tmp_path: Path) -> None:
        session = FakeSession([payload("drivers_2024")])
        client_for(tmp_path, session).get("2024/drivers", limit=100)
        assert "f1-microsector-pipeline" in session.calls[0]["headers"]["User-Agent"]
