"""F005 dimension and fact builders, on the designed session and the F012 fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.metrics.session import compute_metrics
from src.quality.session import load_artefacts
from src.reference.tables import build_constructor_table, build_driver_entries, build_driver_table, build_event_table, build_session_schedule
from src.validate.session import validate_session
from src.warehouse import dimensions as build
from src.warehouse.dimensions import DimensionError
from tests import synthetic_session as syn

FIXTURES = Path(__file__).parent / "fixtures" / "jolpica"


def jolpica(name: str, table_key: str, list_key: str) -> list[dict]:
    payload = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return payload["MRData"][table_key][list_key]


@pytest.fixture(scope="module")
def reference() -> dict[str, pd.DataFrame]:
    races = jolpica("races_2024", "RaceTable", "Races")
    return {
        "dim_event": build_event_table(races, 2024),
        "dim_session_schedule": build_session_schedule(races, 2024),
        "dim_driver": build_driver_table(jolpica("drivers_2024", "DriverTable", "Drivers"), 2024),
        "dim_constructor": build_constructor_table(jolpica("constructors_2024", "ConstructorTable", "Constructors"), 2024),
        "driver_entry": build_driver_entries(jolpica("qualifying_2024_4", "RaceTable", "Races"), 2024),
    }


@pytest.fixture(scope="module")
def frames(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    root = tmp_path_factory.mktemp("warehouse")
    roots = syn.write_full_session(root)
    processed = root / "processed" / "synthetic"
    compute_metrics(roots["grid_root"], roots["microsector_root"], roots["snapshot_root"], roots["aligned_root"],
                    out_root=processed)
    validate_session(roots["snapshot_root"], roots["aligned_root"], roots["grid_root"], processed,
                     out_root=processed, min_laps=2)
    return load_artefacts(roots["snapshot_root"], roots["aligned_root"], roots["grid_root"],
                          roots["microsector_root"], processed)


SUZUKA_META = {"season": 2024, "round_number": 4, "session_requested": "Q", "session_name": "Qualifying",
               "event_name": "Japanese Grand Prix", "country": "Japan", "location": "Suzuka",
               "session_date": "2024-04-06 06:00:00"}
ALIGNMENT_META = {"method": "projection", "official_lap_length_m": 5807.0, "reference_line_length_m": 5722.15}


class TestSessionIdentity:
    def test_reads_the_snapshot_metadata(self) -> None:
        assert build.session_identity(SUZUKA_META) == (2024, 4, "Q")

    def test_incomplete_metadata_is_an_error(self) -> None:
        with pytest.raises(DimensionError, match="no usable identity"):
            build.session_identity({"season": 2024})


class TestDimSession:
    def test_reference_data_supplies_the_circuit_and_the_start(self, reference) -> None:
        row = build.build_dim_session(SUZUKA_META, ALIGNMENT_META, reference,
                                      snapshot_date="2026-09-01", contract_version="abc123")
        assert row["circuit_id"] == "suzuka" and row["circuit_name"] == "Suzuka Circuit"
        assert row["locality"] == "Suzuka" and row["country"] == "Japan"
        assert row["session_start_utc"] == pd.Timestamp("2024-04-06 06:00:00", tz="UTC")
        assert row["official_lap_length_m"] == 5807.0 and row["alignment_method"] == "projection"
        assert row["snapshot_date"] == "2026-09-01" and row["contract_version"] == "abc123"

    def test_it_falls_back_to_the_snapshot_when_reference_data_is_absent(self) -> None:
        """A session can be loaded before its season's reference data exists."""
        row = build.build_dim_session(SUZUKA_META, ALIGNMENT_META, None)
        assert row["event_name"] == "Japanese Grand Prix" and row["circuit_id"] is None
        assert row["session_start_utc"] == pd.Timestamp("2024-04-06 06:00:00", tz="UTC")

    def test_a_session_not_in_the_schedule_keeps_its_own_metadata(self, reference) -> None:
        meta = {**SUZUKA_META, "round_number": 99}
        row = build.build_dim_session(meta, ALIGNMENT_META, reference)
        assert row["round"] == 99 and row["circuit_id"] is None


class TestConstructorResolution:
    def test_it_uses_the_round_entry_not_the_team_string(self, reference) -> None:
        codes = pd.Series(["GAS", "BOT", "RIC", "PER"])
        resolved = build.resolve_constructors(codes, reference["driver_entry"], 2024, 4)
        assert resolved.tolist() == ["alpine", "sauber", "rb", "red_bull"]

    def test_a_round_with_no_entries_resolves_to_nothing(self, reference) -> None:
        resolved = build.resolve_constructors(pd.Series(["VER"]), reference["driver_entry"], 2024, 99)
        assert resolved.isna().all()

    def test_no_reference_data_resolves_to_nothing_rather_than_guessing(self) -> None:
        resolved = build.resolve_constructors(pd.Series(["VER"]), None, 2024, 4)
        assert resolved.isna().all()


class TestDimLap:
    def test_it_carries_identity_timing_and_coverage(self, frames) -> None:
        laps = build.build_dim_lap(frames["lap_summary"], frames.get("ground_truth"), 7, 2024, 4, "Q")
        assert len(laps) == 4
        assert (laps["session_id"] == 7).all()
        assert (laps["season"] == 2024).all() and (laps["session_code"] == "Q").all()
        assert laps["lap_time_s"].notna().all()
        assert laps["lap_residual_s"].notna().all(), "ground truth merged in"
        assert laps["start_coverage_poor"].notna().all()
        assert laps["is_reference"].sum() == 1

    def test_the_team_string_is_an_alias_not_a_key(self, frames) -> None:
        laps = build.build_dim_lap(frames["lap_summary"], frames.get("ground_truth"), 7, 2024, 4, "Q")
        assert (laps["team_alias"] == "Synthetic").all()
        assert laps["constructor_id"].isna().all(), "no reference data was supplied, so no constructor is invented"

    def test_column_order_matches_the_table(self, frames) -> None:
        laps = build.build_dim_lap(frames["lap_summary"], frames.get("ground_truth"), 7, 2024, 4, "Q")
        assert list(laps.columns) == build.DIM_LAP_COLUMNS

    def test_an_empty_lap_summary_is_an_error(self) -> None:
        with pytest.raises(DimensionError, match="no lap to load"):
            build.build_dim_lap(pd.DataFrame(columns=["driver", "lap_number"]), None, 1, 2024, 4, "Q")


class TestDimensionsFromSegmentation:
    def test_microsectors_get_a_length(self, frames) -> None:
        sectors = build.build_dim_microsector(frames["microsectors"], 7)
        assert list(sectors.columns) == build.DIM_MICROSECTOR_COLUMNS
        assert (sectors["length_m"] == sectors["end_m"] - sectors["start_m"]).all()
        assert (sectors["session_id"] == 7).all()

    def test_corner_events_carry_their_boundaries(self, frames) -> None:
        events = build.build_dim_corner_event(frames["events"], 7)
        assert list(events.columns) == build.DIM_CORNER_EVENT_COLUMNS
        assert events["apex_m"].notna().all()


class TestFacts:
    def lap_ids(self, frames) -> dict[tuple, int]:
        keys = sorted(set(zip(frames["grid"]["driver"].astype(str), frames["grid"]["lap_number"].astype(int))))
        return {key: index + 1 for index, key in enumerate(keys)}

    def test_grid_rows_carry_time_and_delta(self, frames) -> None:
        ids = self.lap_ids(frames)
        fact = build.build_fact_grid(frames["grid"], frames.get("delta_t"), ids, 2024, 4, "Q")
        assert list(fact.columns) == build.FACT_GRID_COLUMNS
        assert len(fact) == len(frames["grid"])
        assert fact["t_s"].notna().all()
        assert set(fact["lap_id"]) == set(ids.values())

    def test_without_delta_the_grid_time_is_used(self, frames) -> None:
        ids = self.lap_ids(frames)
        fact = build.build_fact_grid(frames["grid"], None, ids, 2024, 4, "Q")
        assert fact["t_s"].notna().all() and fact["delta_t_s"].isna().all()

    def test_microsector_and_corner_facts(self, frames) -> None:
        ids = self.lap_ids(frames)
        sectors = build.build_fact_microsector(frames["microsector_times"], ids, 2024, 4, "Q")
        corners = build.build_fact_corner_metric(frames["corner_metrics"], ids)
        assert list(sectors.columns) == build.FACT_MICROSECTOR_COLUMNS
        assert list(corners.columns) == build.FACT_CORNER_COLUMNS
        assert len(sectors) == len(frames["microsector_times"])
        assert len(corners) == len(frames["corner_metrics"])

    def test_an_unmapped_lap_is_refused_rather_than_silently_dropped(self, frames) -> None:
        ids = self.lap_ids(frames)
        ids.pop(next(iter(ids)))
        with pytest.raises(DimensionError, match="no lap_id"):
            build.build_fact_grid(frames["grid"], frames.get("delta_t"), ids, 2024, 4, "Q")
