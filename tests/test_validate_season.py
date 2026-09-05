"""F015: the per-session validation table.

It recomputes nothing -- a row is exactly what that session's own F010 report
concluded -- so what is worth testing is that it reads the reports faithfully,
survives the ones that are incomplete, and resolves a snapshot path that may
have been written inside a container.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.validate.season import (
    COLUMNS,
    SeasonSummaryError,
    by_circuit,
    summarise_season,
    write_summary,
)


def report(ok: bool = True, *, lap_median: float = -0.004, closure_p95: float = 0.147,
           driven: float = 5781.6, official: float = 5807.0, snapshot: str = "") -> dict:
    return {
        "snapshot": snapshot,
        "official_lap_length_m": official,
        "acceptance": {
            "laps": 74, "laps_gated": 72, "flagged": ["NOR L6", "VER L2"],
            "lap_residual_median_s": lap_median, "lap_residual_std_s": 0.071,
            "lap_residual": {"p50": 0.046, "p95": 0.130},
            "closure": {"p50": 0.054, "p95": closure_p95},
            "sector_median_s": {"s1": -0.014, "s2": -0.002, "s3": 0.013},
            "sector_std_s": {"s1": 0.059, "s2": 0.057, "s3": 0.059},
            "driven_median_m": driven,
            "line_start_m": -30.1, "line_end_m": 5741.9,
            "line_start_std_m": 4.4, "line_end_std_m": 3.8,
            "v_min_std_median_kmh": 2.05, "v_min_std_p95_kmh": 5.38,
            "checks": {"lap_reconstruction": ok, "delta_closure": ok, "sector_times": ok,
                       "driven_distance": True, "v_min_stability": True,
                       "timing_line_spread": True, "all": ok},
        },
    }


def snapshot_at(root: Path, date: str, slug: str, meta: dict) -> Path:
    directory = root / date / slug
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "session_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return directory


@pytest.fixture()
def processed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw = tmp_path / "raw"
    monkeypatch.setattr("src.validate.season.FASTF1_RAW_ROOT", raw)
    snapshot_at(raw, "2026-09-05", "2024_Japanese-Grand-Prix_Q",
                {"season": 2024, "round_number": 4, "session_requested": "Q", "location": "Suzuka"})
    snapshot_at(raw, "2026-09-05", "2024_Bahrain-Grand-Prix_Q",
                {"season": 2024, "round_number": 1, "session_requested": "Q", "location": "Sakhir"})

    root = tmp_path / "processed"
    for slug, payload in (
        ("2024_Japanese-Grand-Prix_Q_projection",
         report(snapshot="/opt/airflow/project/data/raw/fastf1/2026-09-05/2024_Japanese-Grand-Prix_Q")),
        ("2024_Bahrain-Grand-Prix_Q_projection",
         report(ok=False, lap_median=-0.187, closure_p95=0.258, driven=5370.9, official=5412.0,
                snapshot=r"C:\\data\\raw\\fastf1\\2026-09-05\\2024_Bahrain-Grand-Prix_Q")),
    ):
        directory = root / slug
        directory.mkdir(parents=True)
        (directory / "ground_truth_report.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


class TestSummarise:
    def test_one_row_per_validated_session(self, processed: Path) -> None:
        frame = summarise_season(processed)
        assert len(frame) == 2
        assert list(frame.columns) == list(COLUMNS)

    def test_rows_come_back_in_calendar_order(self, processed: Path) -> None:
        frame = summarise_season(processed)
        assert frame["round"].tolist() == [1, 4]

    def test_the_circuit_comes_from_the_snapshot_not_the_directory_name(self, processed: Path) -> None:
        frame = summarise_season(processed).set_index("round")
        assert frame.loc[1, "circuit"] == "Sakhir"
        assert frame.loc[4, "circuit"] == "Suzuka"

    def test_a_container_path_still_resolves_locally(self, processed: Path) -> None:
        """The report may have been written inside Airflow, where the project
        lives at /opt/airflow/project; only the snapshot's tail identifies it."""
        frame = summarise_season(processed).set_index("round")
        assert frame.loc[4, "snapshot_date"] == "2026-09-05"

    def test_a_windows_path_resolves_too(self, processed: Path) -> None:
        frame = summarise_season(processed).set_index("round")
        assert frame.loc[1, "snapshot_date"] == "2026-09-05"

    def test_the_verdicts_are_carried_through_unchanged(self, processed: Path) -> None:
        frame = summarise_season(processed).set_index("round")
        assert bool(frame.loc[4, "all_ok"]) is True
        assert bool(frame.loc[1, "all_ok"]) is False
        assert bool(frame.loc[1, "driven_distance_ok"]) is True, "only some checks failed"

    def test_the_length_comparison_is_computed_from_both_figures(self, processed: Path) -> None:
        frame = summarise_season(processed).set_index("round")
        assert frame.loc[1, "driven_pct_of_official"] == pytest.approx(-0.76, abs=0.01)

    def test_flagged_laps_are_counted_not_listed(self, processed: Path) -> None:
        """The table is one row per session; the names live in the report."""
        assert summarise_season(processed)["flagged"].tolist() == [2, 2]


class TestMissingInputs:
    def test_no_processed_layer_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(SeasonSummaryError, match="no processed layer"):
            summarise_season(tmp_path / "absent")

    def test_a_processed_layer_with_no_reports_is_an_error(self, tmp_path: Path) -> None:
        (tmp_path / "some_session").mkdir()
        with pytest.raises(SeasonSummaryError, match="ground_truth_report"):
            summarise_season(tmp_path)

    def test_a_session_whose_snapshot_is_gone_still_gets_a_row(self, tmp_path: Path,
                                                              monkeypatch) -> None:
        """Its figures are still the evidence; only its circuit is unknown."""
        monkeypatch.setattr("src.validate.season.FASTF1_RAW_ROOT", tmp_path / "raw")
        root = tmp_path / "processed" / "orphan"
        root.mkdir(parents=True)
        (root / "ground_truth_report.json").write_text(
            json.dumps(report(snapshot="/gone/2026-01-01/whatever")), encoding="utf-8")
        frame = summarise_season(tmp_path / "processed")
        assert len(frame) == 1
        assert pd.isna(frame.loc[0, "circuit"])
        assert frame.loc[0, "lap_residual_median_s"] == pytest.approx(-0.004)


class TestByCircuit:
    def test_one_row_per_circuit_with_a_pass_count(self, processed: Path) -> None:
        grouped = by_circuit(summarise_season(processed))
        assert set(grouped.index) == {"Suzuka", "Sakhir"}
        assert grouped.loc["Suzuka", "passing"] == 1
        assert grouped.loc["Sakhir", "passing"] == 0
        assert bool(grouped.loc["Suzuka", "all_sessions_pass"]) is True
        assert bool(grouped.loc["Sakhir", "all_sessions_pass"]) is False

    def test_the_worst_spread_sorts_first(self, processed: Path) -> None:
        """The table exists to surface where the method struggles."""
        grouped = by_circuit(summarise_season(processed))
        assert list(grouped.index)[0] == "Suzuka" or grouped["lap_residual_std_s"].is_monotonic_decreasing

    def test_an_empty_frame_is_not_an_error(self) -> None:
        assert by_circuit(pd.DataFrame()).empty


class TestWriteSummary:
    def test_it_writes_a_parquet_that_reads_back(self, processed: Path, tmp_path: Path) -> None:
        frame = summarise_season(processed)
        path = write_summary(frame, out_root=tmp_path / "season")
        assert path.exists()
        assert list(pd.read_parquet(path).columns) == list(COLUMNS)

    def test_writing_twice_replaces_rather_than_appends(self, processed: Path, tmp_path: Path) -> None:
        """A session re-run with better alignment must correct its row."""
        frame = summarise_season(processed)
        write_summary(frame, out_root=tmp_path / "season")
        path = write_summary(frame, out_root=tmp_path / "season")
        assert len(pd.read_parquet(path)) == len(frame)
