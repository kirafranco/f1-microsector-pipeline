"""F016 end to end.

Two suites. The first builds the findings for the designed synthetic session,
which runs anywhere and proves the wiring. The second is the acceptance table
from the spec, measured against the real Suzuka artefacts -- it needs `data/`,
which a clone does not have, so it is opt-in: `pytest -m data`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT
from src.metrics.session import compute_metrics
from src.report import session as report
from src.report.pairs import ReportError
from src.validate.session import validate_session
from tests import synthetic_session as syn


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """The designed session, taken all the way through to processed artefacts."""
    root = tmp_path_factory.mktemp("findings")
    roots = syn.write_full_session(root)
    processed = root / "processed" / "synthetic"
    compute_metrics(roots["grid_root"], roots["microsector_root"], roots["snapshot_root"],
                    roots["aligned_root"], out_root=processed)
    validate_session(roots["snapshot_root"], roots["aligned_root"], roots["grid_root"], processed,
                     out_root=processed, min_laps=2)
    return {**roots, "processed": processed, "out": root / "findings"}


@pytest.fixture(scope="module")
def built(synthetic: dict) -> report.FindingsResult:
    return report.build_findings(
        synthetic["processed"], synthetic["grid_root"], synthetic["microsector_root"],
        ("AAA", 1), ("BBB", 1), out_root=synthetic["out"], render=False)


class TestItRunsOnADesignedSession:
    def test_it_produces_a_decomposition_that_sums_to_its_own_sectors(self, built) -> None:
        total = built.decomposition.sectors.loc[~built.decomposition.sectors["partial"], "delta_s"].sum()
        assert built.report.decomposed_total_s == pytest.approx(total)

    def test_every_lap_pairing_is_compared(self, built) -> None:
        assert len(built.pairings) == 4, "two drivers, two laps each"

    def test_it_picks_a_focus_corner_and_can_defend_or_fail_to_defend_it(self, built) -> None:
        assert built.report.focus_event in set(built.consistency.index)
        assert built.report.focus_pairings == len(built.pairings)
        assert isinstance(built.report.claim_ok, bool)

    def test_the_control_is_a_driver_against_himself(self, built) -> None:
        assert built.report.control["pair"].startswith("AAA L")
        assert built.report.control["pair"].count("AAA") == 2

    def test_the_speed_window_is_the_focus_corner_s_own_sectors(self, built) -> None:
        """Not a window chosen afterwards to make the difference look bigger."""
        sectors = built.decomposition.sectors
        focus = sectors[sectors["corners"].astype(str) == built.report.focus_event]
        start, end = built.report.focus_deficit["window_m"]
        assert start == pytest.approx(float(focus["start_m"].min()))
        assert end == pytest.approx(float(focus["end_m"].max()))

    def test_the_reconciliation_covers_all_three_official_sectors(self, built) -> None:
        assert list(built.reconciliation.index) == ["s1", "s2", "s3"]
        assert built.reconciliation["unexplained_s"].notna().all()

    def test_every_frame_is_written_next_to_the_summary(self, built, synthetic) -> None:
        for name in ("pair_sectors", "pairings", "consistency", "sector_reconciliation",
                     "braking", "trace_window"):
            assert (synthetic["out"] / f"{name}.parquet").exists(), name

    def test_the_summary_is_json_and_carries_the_checks(self, built, synthetic) -> None:
        payload = json.loads((synthetic["out"] / "summary.json").read_text(encoding="utf-8"))
        assert set(payload["checks"]) == {"nothing_decomposes", "claim_repeats",
                                          "sector_reconciliation", "braking_within_window", "all"}
        assert payload["pair"] == "AAA L1 vs BBB L1"

    def test_running_it_twice_gives_the_same_answer(self, synthetic) -> None:
        again = report.build_findings(
            synthetic["processed"], synthetic["grid_root"], synthetic["microsector_root"],
            ("AAA", 1), ("BBB", 1), out_root=synthetic["out"] / "again", render=False)
        first = json.loads((synthetic["out"] / "summary.json").read_text(encoding="utf-8"))
        second = again.report.to_dict()
        for key in ("decomposed_total_s", "focus_event", "focus_median_s", "max_unexplained_s"):
            assert first[key] == second[key], key

    def test_a_missing_artefact_says_which_one(self, tmp_path: Path, synthetic) -> None:
        with pytest.raises(ReportError, match="missing artefacts"):
            report.build_findings(tmp_path, synthetic["grid_root"], synthetic["microsector_root"],
                                  ("AAA", 1), ("BBB", 1), render=False)


class TestChoosingTheFocus:
    def test_a_corner_that_always_points_the_same_way_wins(self) -> None:
        import pandas as pd
        frame = pd.DataFrame({
            "median_s": [0.20, 0.12], "always_same_sign": [False, True],
        }, index=pd.Index(["T9", "T5"], name="corners"))
        assert report.choose_focus(frame) == "T5", "consistency beats size"

    def test_with_nothing_consistent_it_falls_back_to_the_largest(self) -> None:
        import pandas as pd
        frame = pd.DataFrame({
            "median_s": [0.20, 0.12], "always_same_sign": [False, False],
        }, index=pd.Index(["T9", "T5"], name="corners"))
        assert report.choose_focus(frame) == "T9"


# --------------------------------------------------------------------------- #
# The spec's acceptance table, against the real session. `pytest -m data`.
# --------------------------------------------------------------------------- #

#: The session's slug follows the F006/F015 convention -- the schedule's event
#: name plus the alignment method. The old `2024_Japan_Q` roots were retired
#: once the backfill re-ingested Suzuka under this one.
SUZUKA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "2024_Japanese-Grand-Prix_Q_projection"
SUZUKA_GRID = PROJECT_ROOT / "data" / "interim" / "grid" / "2024_Japanese-Grand-Prix_Q_projection"
SUZUKA_MICROSECTORS = PROJECT_ROOT / "data" / "interim" / "microsectors" / "2024_Japanese-Grand-Prix_Q_projection"


@pytest.fixture(scope="module")
def suzuka() -> report.FindingsResult:
    """The published run: the same call the document's numbers come from."""
    if not (SUZUKA_PROCESSED / "lap_summary.parquet").exists():
        pytest.skip("the Suzuka artefacts are not on this machine")
    return report.build_findings(SUZUKA_PROCESSED, SUZUKA_GRID, SUZUKA_MICROSECTORS,
                                 ("VER", 11), ("PER", 11))


@pytest.mark.data
class TestSuzukaAcceptance:
    def test_criterion_1_the_pair_decomposes_to_the_published_figures(self, suzuka) -> None:
        """Re-derived on 2026-09-05 after F015 placed the axis origin: the
        corner phases are unchanged to a thousandth, while the straights row
        and the total read from a grid zero that now sits tens of metres
        further down the road. The document carries the same figures."""
        assert suzuka.report.official_gap_s == pytest.approx(0.066, abs=0.001)
        assert suzuka.report.decomposed_total_s == pytest.approx(0.1882, abs=0.001)
        expected = {"braking": 0.1190, "entry": -0.1229, "apex": 0.1188,
                    "exit": 0.0121, "straight": 0.0613}
        for phase, delta in expected.items():
            assert suzuka.report.phases[phase]["delta_s"] == pytest.approx(delta, abs=0.001), phase

    def test_criterion_1_nothing_clears_two_sigma(self, suzuka) -> None:
        """The headline. If this ever fails, the write-up needs rewriting."""
        assert suzuka.report.max_phase_ratio < 2.0
        assert suzuka.report.max_event_ratio < 2.0
        assert suzuka.report.nothing_decomposes is True

    def test_criterion_2_the_focus_corner_repeats_on_every_pairing(self, suzuka) -> None:
        assert suzuka.report.focus_event == "T5"
        assert suzuka.report.focus_positive == 16
        assert suzuka.report.focus_pairings == 16
        assert suzuka.report.focus_median_s == pytest.approx(0.122, abs=0.001)

    def test_criterion_3_the_speed_traces_agree(self, suzuka) -> None:
        deficit = suzuka.report.focus_deficit
        assert deficit["all_slower_on_average"] is True
        assert deficit["mean_kmh_smallest"] <= -2.0
        assert deficit["share_slower_min"] >= 0.7

    def test_criterion_3_the_control_shows_nothing_there(self, suzuka) -> None:
        """The same driver against his own lap does not show the difference."""
        assert abs(suzuka.report.control["focus_s"]) < 0.05

    def test_criterion_4_the_sector_disagreement_is_f010_s(self, suzuka) -> None:
        assert suzuka.report.max_unexplained_s <= 0.005
        for sector in ("s1", "s2", "s3"):
            row = suzuka.report.reconciliation[sector]
            assert abs(row["unexplained_s"]) <= 0.005, sector

    def test_criterion_5_the_braking_outliers_are_accounted_for(self, suzuka) -> None:
        braking = suzuka.report.braking
        assert braking["events"] == 6
        assert braking["metric_artefact"] == ["T8-T9"]
        assert braking["confirmed"] == ["T16-T17"]
        assert braking["definition_sensitive"] == ["T13-T14"]
        assert braking["metric_outliers_explained"] is True

    def test_criterion_5_the_dab_is_the_norm(self, suzuka) -> None:
        assert suzuka.report.dab_event == "T8-T9"
        assert suzuka.report.dab_laps == 71
        assert suzuka.report.dab_of == 74
        assert set(suzuka.report.dab_absent) == {"ALO L11", "PER L11", "VER L8"}

    def test_criterion_7_the_figures_are_written_and_small(self, suzuka) -> None:
        assert len(suzuka.report.figures) == 4
        total = 0
        for name in suzuka.report.figures:
            path = report.FIGURES_DIR / name
            assert path.exists(), name
            assert path.stat().st_size <= 300_000, name
            total += path.stat().st_size
        assert total <= 1_500_000

    def test_criterion_9_it_is_quick_enough_to_rerun_without_thinking(self, suzuka) -> None:
        assert suzuka.report.elapsed_s <= 10.0

    def test_every_check_passes(self, suzuka) -> None:
        assert suzuka.report.to_dict()["checks"]["all"] is True
