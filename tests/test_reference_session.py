"""F012 season ingest: artefacts, quality, idempotency, and the live acceptance run."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import DATA_ROOT
from src.quality.contracts import CONTRACTS, REFERENCE_CONTRACTS, SESSION_TABLES
from src.quality.engine import require, validate_tables
from src.reference import session as mod
from src.reference.jolpica import JolpicaClient, JolpicaEmptyError
from src.reference.session import TABLE_FILES, ingest_reference, load_reference

FIXTURES = Path(__file__).parent / "fixtures" / "jolpica"
SUZUKA_LAPS = DATA_ROOT / "raw" / "fastf1" / "2026-09-01" / "2024_Japan_Q" / "laps.parquet"


def payload(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


class RecordedSession:
    """Answers from the recorded fixtures, so the ingest runs with no network."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(url)
        if "/races" in url:
            body = payload("races_2024")
        elif "/drivers" in url:
            body = payload("drivers_2024")
        elif "/constructors" in url:
            body = payload("constructors_2024")
        elif "/qualifying" in url:
            body = payload("qualifying_2024_4") if "/4/" in url else payload("empty")
        elif "/results" in url:
            body = payload("empty")
        else:
            raise AssertionError(f"unexpected request {url}")

        class Response:
            status_code = 200

            def json(self_inner):
                return body

            def raise_for_status(self_inner):
                return None

        return Response()


@pytest.fixture()
def client(tmp_path: Path) -> JolpicaClient:
    return JolpicaClient(cache_dir=tmp_path / "cache", session=RecordedSession(), sleep=lambda _: None)


class TestIngest:
    def test_builds_every_table(self, client: JolpicaClient, tmp_path: Path) -> None:
        result = ingest_reference(2024, rounds=[4], out_root=tmp_path / "reference",
                                  snapshot_date="2026-09-04", client=client)
        assert set(result.tables) == set(TABLE_FILES) == set(REFERENCE_CONTRACTS)
        assert result.counts == {"dim_event": 2, "dim_session_schedule": 10, "dim_driver": 25,
                                 "dim_constructor": 10, "driver_entry": 20}

    def test_writes_parquet_and_an_immutable_snapshot(self, client: JolpicaClient, tmp_path: Path) -> None:
        result = ingest_reference(2024, rounds=[4], out_root=tmp_path / "reference",
                                  snapshot_date="2026-09-04", client=client)
        for filename in TABLE_FILES.values():
            assert (result.root / filename).exists(), filename
        assert (result.root / "reference_meta.json").exists()
        for name in ("races", "drivers", "constructors", "driver_entries"):
            assert (result.snapshot_root / f"{name}.json").exists(), name
        assert result.snapshot_root.parts[-2:] == ("2026-09-04", "2024")

    def test_meta_records_the_source_and_the_join_rule(self, client: JolpicaClient, tmp_path: Path) -> None:
        result = ingest_reference(2024, rounds=[4], out_root=tmp_path / "reference", client=client)
        meta = json.loads((result.root / "reference_meta.json").read_text(encoding="utf-8"))
        assert meta["source"] == "https://api.jolpi.ca/ergast/f1"
        assert meta["season"] == 2024 and meta["rounds_ingested"] == [4]
        assert "never from matching" in meta["note"]

    def test_a_round_without_results_is_skipped_not_fatal(self, client: JolpicaClient, tmp_path: Path) -> None:
        """Round 5 is in the schedule but has no recorded result: a future race."""
        result = ingest_reference(2024, rounds=[4, 5], out_root=tmp_path / "reference", client=client)
        assert result.rounds_ingested == (4,)
        assert [r for r, _ in result.rounds_skipped] == [5]
        assert len(result.tables["driver_entry"]) == 20

    def test_a_season_with_no_entries_at_all_raises(self, client: JolpicaClient, tmp_path: Path) -> None:
        with pytest.raises(JolpicaEmptyError, match="no round produced"):
            ingest_reference(2024, rounds=[5], out_root=tmp_path / "reference", client=client)

    def test_second_run_uses_only_the_cache(self, client: JolpicaClient, tmp_path: Path) -> None:
        first = ingest_reference(2024, rounds=[4], out_root=tmp_path / "reference", client=client)
        assert first.requests_made > 0

        offline = JolpicaClient(cache_dir=client.cache_dir, session=None, sleep=lambda _: None)
        offline.session = _Exploding()
        second = ingest_reference(2024, rounds=[4], out_root=tmp_path / "reference2", client=offline)
        assert second.requests_made == 0 and second.cache_hits == first.requests_made
        for name in TABLE_FILES:
            pd.testing.assert_frame_equal(first.tables[name], second.tables[name])

    def test_load_reference_reads_what_was_written(self, client: JolpicaClient, tmp_path: Path) -> None:
        result = ingest_reference(2024, rounds=[4], out_root=tmp_path / "reference", client=client)
        loaded = load_reference(2024, root=result.root)
        assert set(loaded) == set(TABLE_FILES)
        pd.testing.assert_frame_equal(loaded["dim_event"], result.tables["dim_event"])

    def test_default_output_root(self, client: JolpicaClient, tmp_path: Path,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "REFERENCE_ROOT", tmp_path / "reference")
        monkeypatch.setattr(mod, "JOLPICA_RAW_ROOT", tmp_path / "raw")
        result = ingest_reference(2024, rounds=[4], client=client)
        assert result.root == tmp_path / "reference" / "2024"


class _Exploding:
    def get(self, *args, **kwargs):
        raise AssertionError("network used on a cached re-run")


class TestQuality:
    def test_every_reference_table_passes_its_contract(self, client: JolpicaClient, tmp_path: Path) -> None:
        result = ingest_reference(2024, rounds=[4], out_root=tmp_path / "reference", client=client)
        report = validate_tables(result.tables, REFERENCE_CONTRACTS)
        assert report.ok, [str(f) for f in report.errors]
        require(report)

    def test_an_orphan_entry_is_caught(self, client: JolpicaClient, tmp_path: Path) -> None:
        result = ingest_reference(2024, rounds=[4], out_root=tmp_path / "reference", client=client)
        tables = {name: frame.copy() for name, frame in result.tables.items()}
        tables["driver_entry"].loc[0, "constructor_id"] = "ferrari_gmbh"
        report = validate_tables(tables, REFERENCE_CONTRACTS)
        assert not report.ok
        assert any("dim_constructor" in f.rule for f in report.errors)

    def test_reference_contracts_joined_the_main_registry(self) -> None:
        assert set(REFERENCE_CONTRACTS) <= set(CONTRACTS)
        assert not (set(REFERENCE_CONTRACTS) & SESSION_TABLES)
        assert len(CONTRACTS) == len(SESSION_TABLES) + len(REFERENCE_CONTRACTS)


@pytest.mark.skipif(not SUZUKA_LAPS.exists(), reason="Suzuka snapshot not present under data/")
class TestJoinToFastF1:
    def test_every_fastf1_driver_resolves_to_a_driver_and_a_constructor(
        self, client: JolpicaClient, tmp_path: Path
    ) -> None:
        """Criterion 2: the join that F005 depends on, on the real snapshot."""
        result = ingest_reference(2024, rounds=[4], out_root=tmp_path / "reference", client=client)
        laps = pd.read_parquet(SUZUKA_LAPS)
        codes = set(laps["driver"].astype(str))
        entries = result.tables["driver_entry"]
        entries = entries[entries["round"] == 4]

        assert len(codes) == 20
        assert codes <= set(entries["code"].astype(str)), codes - set(entries["code"].astype(str))
        assert codes <= set(result.tables["dim_driver"]["code"].astype(str))
        resolved = entries[entries["code"].isin(codes)]
        assert resolved["constructor_id"].notna().all()
        assert set(resolved["constructor_id"]) <= set(result.tables["dim_constructor"]["constructor_id"])

    def test_car_numbers_agree_with_the_snapshot(self, client: JolpicaClient, tmp_path: Path) -> None:
        result = ingest_reference(2024, rounds=[4], out_root=tmp_path / "reference", client=client)
        laps = pd.read_parquet(SUZUKA_LAPS)[["driver", "driver_number"]].drop_duplicates()
        entries = result.tables["driver_entry"].query("round == 4")
        merged = laps.merge(entries, left_on="driver", right_on="code", how="inner")
        assert len(merged) == 20
        assert (merged["driver_number"].astype(int) == merged["car_number"].astype(int)).all()


@pytest.mark.network
class TestLiveIngest:
    """Criteria 1, 4, 5 and 10 against the real API. Run with `pytest -m network`."""

    def test_a_whole_season(self, tmp_path: Path) -> None:
        client = JolpicaClient(cache_dir=tmp_path / "cache")
        result = ingest_reference(2024, out_root=tmp_path / "reference", client=client)

        assert result.counts["dim_event"] == 24
        assert result.counts["dim_driver"] == 25
        assert result.counts["dim_constructor"] == 10
        assert result.rounds_skipped == ()
        assert client.requests_made <= 30
        assert result.elapsed_s <= 60.0

        schedule = result.tables["dim_session_schedule"]
        qualifying = schedule[(schedule["round"] == 4) & (schedule["session"] == "Q")]
        assert qualifying["session_start_utc"].iloc[0] == pd.Timestamp("2024-04-06 06:00:00", tz="UTC")
        sprint = schedule[schedule["round"] == 5]["session"].tolist()
        assert "SQ" in sprint and "S" in sprint

        report = validate_tables(result.tables, REFERENCE_CONTRACTS)
        assert report.ok, [str(f) for f in report.errors]
