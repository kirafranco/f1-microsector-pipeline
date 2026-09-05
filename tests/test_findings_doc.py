"""The published document has to agree with the run that produced it.

A findings document drifts the moment someone edits a number in prose without
rebuilding, so the figures it links must exist and the numbers it quotes in its
result tables must appear in summary.json. The summary lives under data/, which
a clone does not have, so those checks skip rather than fail there.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT

DOC = PROJECT_ROOT / "docs" / "findings" / "2024-suzuka-q3-ver-per.md"
SUMMARY = PROJECT_ROOT / "data" / "processed" / "2024_Japan_Q_projection" / "findings" / "summary.json"

IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
#: Numbers written as a signed or unsigned quantity of seconds, e.g. "+0.122 s".
SECONDS = re.compile(r"([+-]?\d+\.\d+)\s*s\b")


def normalise(text: str) -> str:
    """Typography out, so the comparison is about numbers and not about dashes.

    The document is written for people: it uses a real minus sign and an en dash
    in corner names. The data uses ASCII. Neither should have to give way, so
    the match happens on a normalised copy.
    """
    return text.replace("\u2212", "-").replace("\u2013", "-")


@pytest.fixture(scope="module")
def document() -> str:
    return normalise(DOC.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prose() -> str:
    """The document as written, for checks about the words rather than the numbers."""
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def summary() -> dict:
    if not SUMMARY.exists():
        pytest.skip("no findings summary on this machine; run build_findings first")
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


class TestTheDocument:
    def test_it_exists_and_is_not_a_stub(self, prose: str) -> None:
        assert len(prose.split()) > 800, "a findings write-up is not a paragraph"

    def test_it_names_its_source_and_snapshot(self, document: str) -> None:
        """A result without a dated source cannot be reproduced or dated."""
        assert "FastF1" in document
        assert "2026-09-01" in document

    def test_it_names_the_file_its_numbers_come_from(self, document: str) -> None:
        assert "summary.json" in document

    def test_it_says_how_to_rebuild_it(self, document: str) -> None:
        assert "build_findings" in document

    def test_every_figure_it_shows_exists(self, prose: str) -> None:
        for target in IMAGE.findall(prose):
            assert (DOC.parent / target).exists(), target

    def test_it_shows_the_four_figures(self, prose: str) -> None:
        assert len(IMAGE.findall(prose)) == 4


class TestItDoesNotOverclaim:
    """The point of the write-up is what it refuses to say."""

    def test_it_states_that_the_gap_cannot_be_attributed(self, document: str) -> None:
        head = document[:2_000].lower()
        assert "cannot be placed" in head or "does not decompose" in head

    def test_it_gives_the_resolution_of_the_method(self, document: str) -> None:
        assert "0.2 s" in document and "0.3 s" in document

    def test_it_says_a_p_value_would_be_wrong_here(self, document: str) -> None:
        assert "p-value" in document
        assert "not sixteen independent samples" in document.lower() or "independent samples" in document

    def test_it_admits_the_single_session_limit(self, document: str) -> None:
        assert "One session, one circuit, one pair" in document


class TestItAgreesWithTheRun:
    def test_the_headline_numbers_are_in_the_summary(self, document: str, summary: dict) -> None:
        assert f"{summary['official_gap_s']:.3f}" in document.replace("+", "")
        assert f"{summary['focus_median_s']:.3f}" in document.replace("+", "")
        assert summary["focus_event"] in document

    def test_the_phase_table_matches(self, document: str, summary: dict) -> None:
        for phase, row in summary["phases"].items():
            rendered = f"{row['delta_s']:+.3f} s"
            assert rendered in document, f"{phase}: {rendered}"
            assert f"{row['sigma_s']:.3f} s" in document, f"{phase} sigma"

    def test_the_sign_test_matches(self, document: str, summary: dict) -> None:
        focus = summary["consistency"][summary["focus_event"]]
        assert f"{focus['positive']} / {focus['pairings']}" in document
        assert summary["focus_deficit"]["all_slower_on_average"] is True

    def test_the_braking_verdicts_match(self, document: str, summary: dict) -> None:
        braking = summary["braking"]
        for corner in braking["metric_artefact"]:
            assert corner in document, corner
        for corner in braking["confirmed"]:
            assert corner in document, corner
        assert f"{summary['dab_laps']} of the session's {summary['dab_of']} laps" in document

    def test_the_reconciliation_matches(self, document: str, summary: dict) -> None:
        for sector, row in summary["reconciliation"].items():
            assert f"{row['official_gap_s']:+.3f} s" in document, sector
            assert f"{row['unexplained_s']:+.4f} s" in document, sector

    def test_it_quotes_no_second_figure_the_summary_does_not_know(self, document: str, summary: dict) -> None:
        """Every "0.123 s" in a result table has to be traceable to the run.

        Prose thresholds and figures quoted from earlier features are listed
        explicitly, so adding a new number to the document means either it comes
        from the summary or it gets justified here.
        """
        allowed = {
            # Thresholds and rules of thumb, stated as such in the prose.
            "2.0", "0.2", "0.3", "0.066", "0.0035",
            # Figures from F004/F010 quoted with attribution.
            "0.29", "0.49", "0.57", "0.473", "0.202", "0.119", "0.125",
        }
        for value in set(SECONDS.findall(document)):
            number = value.lstrip("+")
            if number in allowed or number.lstrip("-") in allowed:
                continue
            assert _appears_in(float(value), summary), f"{value} s is in the document but not in the run"


def _appears_in(value: float, summary: dict, tolerance: float = 5e-4) -> bool:
    """Whether a number the document prints is one the run produced."""
    return any(abs(value - candidate) <= tolerance for candidate in _numbers(summary))


def _numbers(payload) -> list[float]:
    if isinstance(payload, bool):
        return []
    if isinstance(payload, (int, float)):
        return [float(payload)]
    if isinstance(payload, dict):
        return [number for value in payload.values() for number in _numbers(value)]
    if isinstance(payload, list):
        return [number for value in payload for number in _numbers(value)]
    return []
