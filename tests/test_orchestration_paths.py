"""F006 run paths: the naming convention, finally written down as code."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.config import FASTF1_RAW_ROOT, INTERIM_ROOT, PROCESSED_ROOT
from src.orchestration.paths import RunPathError, SessionRun, latest_snapshot_root

SUZUKA = SessionRun(season=2024, event="Japan", session="Q")


class TestSlug:
    def test_the_snapshot_slug_is_the_one_f002_writes(self) -> None:
        """The existing Suzuka snapshot is `2024_Japan_Q`; this must agree."""
        assert SUZUKA.snapshot_slug == "2024_Japan_Q"

    def test_derived_artefacts_carry_the_alignment_method(self) -> None:
        """Two methods over one snapshot must not overwrite each other -- and
        `data/interim/aligned/` still holds the `_anchors` root from the
        comparison that settled D5."""
        assert SUZUKA.slug == "2024_Japan_Q_projection"
        assert SessionRun(2024, "Japan", "Q", method="anchors").slug == "2024_Japan_Q_anchors"

    def test_an_event_name_with_spaces_is_made_filesystem_safe(self) -> None:
        run = SessionRun(season=2024, event="Japanese Grand Prix", session="Q")
        assert run.snapshot_slug == "2024_Japanese-Grand-Prix_Q"
        assert " " not in run.slug

    def test_the_label_reads_like_a_log_line(self) -> None:
        assert SUZUKA.label == "2024 Japan Q"


class TestRoots:
    def test_each_stage_gets_its_own_root_under_the_right_layer(self) -> None:
        assert SUZUKA.aligned_root == INTERIM_ROOT / "aligned" / "2024_Japan_Q_projection"
        assert SUZUKA.grid_root == INTERIM_ROOT / "grid" / "2024_Japan_Q_projection"
        assert SUZUKA.microsector_root == INTERIM_ROOT / "microsectors" / "2024_Japan_Q_projection"
        assert SUZUKA.processed_root == PROCESSED_ROOT / "2024_Japan_Q_projection"

    def test_they_match_what_is_already_on_disk(self) -> None:
        """The roots F008 and F003 wrote by hand are the ones this produces."""
        assert SUZUKA.grid_root.name == "2024_Japan_Q_projection"
        assert SUZUKA.processed_root.parent == PROCESSED_ROOT

    def test_the_snapshot_root_is_dated_and_undecorated(self) -> None:
        """Snapshots are immutable and per-day; the method is not theirs."""
        root = SUZUKA.snapshot_root(date(2026, 9, 1))
        assert root == FASTF1_RAW_ROOT / "2026-09-01" / "2024_Japan_Q"
        assert SUZUKA.snapshot_root("2026-09-01") == root


class TestValidation:
    @pytest.mark.parametrize("kwargs, match", [
        ({"event": ""}, "event is empty"),
        ({"session": "  "}, "session is empty"),
        ({"season": 1930}, "before Formula 1 existed"),
    ])
    def test_a_run_that_cannot_be_named_is_refused(self, kwargs: dict, match: str) -> None:
        base = {"season": 2024, "event": "Japan", "session": "Q"}
        with pytest.raises(RunPathError, match=match):
            SessionRun(**{**base, **kwargs})


class TestFromParams:
    def test_it_builds_from_a_dag_run(self) -> None:
        run = SessionRun.from_params({"season": "2024", "event": "Japan", "session": "Q"})
        assert run == SUZUKA

    def test_the_method_defaults_to_the_one_d5_chose(self) -> None:
        assert SessionRun.from_params({"season": 2024, "event": "Japan", "session": "Q"}).method == "projection"

    def test_a_missing_param_is_an_error_not_a_path(self, ) -> None:
        """Otherwise a typo in the trigger becomes a directory called None."""
        with pytest.raises(RunPathError, match=r"missing \['session'\]"):
            SessionRun.from_params({"season": 2024, "event": "Japan"})

    def test_round_trips_through_a_dictionary(self) -> None:
        assert SessionRun.from_params(SUZUKA.to_dict()) == SUZUKA


class TestLatestSnapshot:
    def build(self, root: Path, days: list[str], slug: str = "2024_Japan_Q",
              manifest: bool = True) -> None:
        for day in days:
            directory = root / day / slug
            directory.mkdir(parents=True)
            if manifest:
                (directory / "manifest.json").write_text("{}", encoding="utf-8")

    def test_the_newest_dated_snapshot_wins(self, tmp_path: Path) -> None:
        self.build(tmp_path, ["2026-08-30", "2026-09-01", "2026-08-15"])
        found = latest_snapshot_root(SUZUKA, root=tmp_path)
        assert found is not None and found.parent.name == "2026-09-01"

    def test_a_directory_without_a_manifest_is_not_a_snapshot(self, tmp_path: Path) -> None:
        """A half-written or interrupted ingest must not be picked up."""
        self.build(tmp_path, ["2026-09-01"], manifest=False)
        self.build(tmp_path, ["2026-08-30"])
        found = latest_snapshot_root(SUZUKA, root=tmp_path)
        assert found is not None and found.parent.name == "2026-08-30"

    def test_nothing_ingested_is_none_not_an_error(self, tmp_path: Path) -> None:
        assert latest_snapshot_root(SUZUKA, root=tmp_path) is None

    def test_a_missing_raw_root_is_none(self, tmp_path: Path) -> None:
        assert latest_snapshot_root(SUZUKA, root=tmp_path / "absent") is None

    def test_another_session_s_snapshot_is_not_mistaken_for_this_one(self, tmp_path: Path) -> None:
        self.build(tmp_path, ["2026-09-01"], slug="2024_Japan_R")
        assert latest_snapshot_root(SUZUKA, root=tmp_path) is None
