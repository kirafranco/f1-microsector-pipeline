"""F015: the circuit table, and whether the data agrees with it.

The table is hand-entered reference data, so the interesting tests are not that
it parses. They are that it covers the calendar it claims to, and that every
figure in it is consistent with the laps actually driven at that circuit --
because a wrong entry looks exactly like a correct one until something measures
against it. A first pass at collecting these numbers returned Pescara's
22.835 km for Silverstone; that is what these tests exist to catch.
"""

from __future__ import annotations

import json

import pytest

from src.align.circuits import (
    OFFICIAL_LAP_LENGTH_M,
    OFFICIAL_LENGTH_BAND_PCT,
    UnknownCircuitError,
    official_lap_length_m,
    within_official_band,
)
from src.config import PROCESSED_ROOT

#: FastF1's `Location` for every round of the 2024 championship, in order.
CALENDAR_2024 = (
    "Sakhir", "Jeddah", "Melbourne", "Suzuka", "Shanghai", "Miami", "Imola",
    "Monaco", "Montréal", "Barcelona", "Spielberg", "Silverstone", "Budapest",
    "Spa-Francorchamps", "Zandvoort", "Monza", "Baku", "Marina Bay", "Austin",
    "Mexico City", "São Paulo", "Las Vegas", "Lusail", "Yas Island",
)


class TestCoverage:
    def test_every_2024_round_has_a_length(self) -> None:
        missing = [name for name in CALENDAR_2024 if name not in OFFICIAL_LAP_LENGTH_M]
        assert missing == [], f"the backfill would fail at {missing}"

    def test_the_table_holds_nothing_it_does_not_need(self) -> None:
        """A stray entry is a stray entry until a season needs it; keeping the
        table to the calendar keeps every figure checkable against real laps."""
        extra = sorted(set(OFFICIAL_LAP_LENGTH_M) - set(CALENDAR_2024))
        assert extra == []

    def test_suzuka_is_the_figure_f008_took_from_the_fia_specification(self) -> None:
        """The cross-check on the whole collection: the source used for the
        other 23 agrees exactly with the one number entered by hand."""
        assert OFFICIAL_LAP_LENGTH_M["Suzuka"] == 5807.0


class TestPlausibility:
    @pytest.mark.parametrize("location", CALENDAR_2024)
    def test_the_length_is_a_grand_prix_circuit(self, location: str) -> None:
        """Monaco at 3,337 m is the shortest ever raced and Spa at 7,004 m the
        longest on the calendar. Anything outside that is a parsing accident."""
        assert 3000.0 <= OFFICIAL_LAP_LENGTH_M[location] <= 7500.0, location

    def test_monaco_is_the_shortest_and_spa_the_longest(self) -> None:
        shortest = min(OFFICIAL_LAP_LENGTH_M, key=OFFICIAL_LAP_LENGTH_M.get)
        longest = max(OFFICIAL_LAP_LENGTH_M, key=OFFICIAL_LAP_LENGTH_M.get)
        assert shortest == "Monaco" and longest == "Spa-Francorchamps"

    def test_no_two_circuits_share_a_suspicious_length(self) -> None:
        """Miami and Sakhir genuinely share 5,412 m; a third collision would
        mean a row was matched twice."""
        counts: dict[float, list[str]] = {}
        for name, metres in OFFICIAL_LAP_LENGTH_M.items():
            counts.setdefault(metres, []).append(name)
        shared = {metres: names for metres, names in counts.items() if len(names) > 1}
        assert shared == {5412.0: ["Sakhir", "Miami"]}, shared


class TestLookup:
    def test_an_unknown_circuit_says_what_to_do_about_it(self) -> None:
        with pytest.raises(UnknownCircuitError, match="FIA circuit specification"):
            official_lap_length_m("Nürburgring")

    def test_the_band_is_asymmetric_because_a_racing_line_cuts_apexes(self) -> None:
        low, high = OFFICIAL_LENGTH_BAND_PCT
        assert low < 0 < high and abs(low) > high

    def test_within_band_accepts_a_shorter_lap_and_refuses_a_longer_one(self) -> None:
        official = OFFICIAL_LAP_LENGTH_M["Suzuka"]
        assert within_official_band(official * 0.99, "Suzuka"), "a racing line is shorter"
        assert not within_official_band(official * 1.01, "Suzuka"), "longer is a defect"


@pytest.mark.data
class TestAgainstTheLapsActuallyDriven:
    """Every entry checked against the circuit's own measured laps.

    This is what makes the table trustworthy rather than merely present: a
    figure copied from the wrong row would put the measured laps outside the
    band, and the session that used it would fail F010's distance check.
    """

    @staticmethod
    def measured() -> list[tuple[str, float, str]]:
        rows = []
        for report in sorted(PROCESSED_ROOT.glob("*/ground_truth_report.json")):
            payload = json.loads(report.read_text(encoding="utf-8"))
            driven = payload.get("acceptance", {}).get("driven_median_m")
            official = payload.get("official_lap_length_m")
            if driven is None or not official:
                continue
            location = next((name for name, metres in OFFICIAL_LAP_LENGTH_M.items()
                             if metres == official), None)
            if location:
                rows.append((location, float(driven), report.parent.name))
        return rows

    def test_there_is_something_to_check(self) -> None:
        if not self.measured():
            pytest.skip("no validated sessions on this machine")

    def test_every_measured_lap_agrees_with_its_table_entry(self) -> None:
        rows = self.measured()
        if not rows:
            pytest.skip("no validated sessions on this machine")
        wrong = [
            (session, location, round(driven, 1), OFFICIAL_LAP_LENGTH_M[location])
            for location, driven, session in rows
            if not within_official_band(driven, location)
        ]
        assert wrong == [], f"measured laps disagree with the table: {wrong}"
