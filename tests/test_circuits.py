"""F015 and F025: the circuit layout table, and whether the data agrees with it.

The table is hand-entered reference data, so the interesting tests are not that
it parses. They are that it covers the calendars it claims to, that every figure
is consistent with the laps actually driven at that circuit -- because a wrong
entry looks exactly like a correct one until something measures against it --
and, since F025, that it is keyed by something the source does not rename.

A first pass at collecting these numbers returned Pescara's 22.835 km for
Silverstone; that is what these tests exist to catch. F025 found the sequel: the
old location key resolved only 2 of 8 seasons FastF1 covers, and would have
answered 2025's Singapore with 2024's length rather than failing.
"""

from __future__ import annotations

import json

import pytest

from src.align.circuits import (
    LAYOUT_OVERRIDES,
    LAYOUTS,
    OFFICIAL_LENGTH_BAND_PCT,
    UnknownCircuitError,
    layouts_for,
    official_lap_length_m,
    resolve_layout,
    within_official_band,
)
from pathlib import Path

from src.config import FASTF1_RAW_ROOT, PROCESSED_ROOT

#: `(round, circuit_id)` for the 2024 championship, in order, with the length
#: the pipeline used for all 48 loaded sessions. This is F025's criterion 1:
#: the re-key must reproduce every one of these exactly.
CALENDAR_2024 = (
    ("bahrain", 5412.0), ("jeddah", 6174.0), ("albert_park", 5278.0), ("suzuka", 5807.0),
    ("shanghai", 5451.0), ("miami", 5412.0), ("imola", 4909.0), ("monaco", 3337.0),
    ("villeneuve", 4361.0), ("catalunya", 4657.0), ("red_bull_ring", 4318.0),
    ("silverstone", 5891.0), ("hungaroring", 4381.0), ("spa", 7004.0),
    ("zandvoort", 4259.0), ("monza", 5793.0), ("baku", 6003.0), ("marina_bay", 4940.0),
    ("americas", 5513.0), ("rodriguez", 4304.0), ("interlagos", 4309.0),
    ("vegas", 6201.0), ("losail", 5419.0), ("yas_marina", 5281.0),
)


class TestCriterion1The2024SeasonIsUnchanged:
    """Nothing that worked before F025 may resolve differently after it."""

    @pytest.mark.parametrize("circuit_id,length_m", CALENDAR_2024)
    def test_each_2024_round_resolves_to_the_length_it_was_loaded_with(
            self, circuit_id: str, length_m: float) -> None:
        assert official_lap_length_m(2024, circuit_id) == length_m

    def test_the_calendar_is_the_whole_season(self) -> None:
        assert len(CALENDAR_2024) == 24


class TestCriterion4TwoLayoutsOfOneCircuit:
    """2020 ran Bahrain twice, on different tracks, under one circuit_id.

    A table keyed by circuit could not express this at all, which is why the
    key is a layout.
    """

    def test_the_bahrain_grand_prix_ran_the_grand_prix_circuit(self) -> None:
        assert official_lap_length_m(2020, "bahrain", "Bahrain Grand Prix") == 5412.0

    def test_the_sakhir_grand_prix_ran_the_outer_loop(self) -> None:
        assert official_lap_length_m(2020, "bahrain", "Sakhir Grand Prix") == 3543.0

    def test_the_outer_loop_is_never_chosen_by_default(self) -> None:
        """Otherwise every Bahrain event in 2020 would be ambiguous."""
        assert LAYOUTS["bahrain_outer"].override_only is True
        assert official_lap_length_m(2020, "bahrain") == 5412.0

    def test_every_override_names_a_layout_that_exists(self) -> None:
        for (season, event), layout_id in LAYOUT_OVERRIDES.items():
            assert layout_id in LAYOUTS, f"({season}, {event!r}) -> {layout_id!r}"
            assert LAYOUTS[layout_id].covers(season)


class TestCriterion3And5TheRenamesAndTheChanges:
    """The three renames and the 2025 changes that motivated F025."""

    @pytest.mark.parametrize("season,circuit_id,length_m", [
        # 2025's three problems: one rename, two silent wrong lengths.
        (2025, "miami", 5412.0),
        (2025, "marina_bay", 4927.0),
        (2025, "red_bull_ring", 4326.0),
        # The same circuits under their earlier names and layouts.
        (2024, "marina_bay", 4940.0),
        (2018, "marina_bay", 5063.0),
        (2024, "red_bull_ring", 4318.0),
        (2018, "monaco", 3337.0),
        (2018, "yas_marina", 5554.0),
        (2024, "yas_marina", 5281.0),
        (2019, "albert_park", 5303.0),
        (2024, "albert_park", 5278.0),
        (2021, "catalunya", 4675.0),
        (2018, "catalunya", 4655.0),
        (2024, "catalunya", 4657.0),
        (2021, "losail", 5380.0),
        (2024, "losail", 5419.0),
    ])
    def test_the_season_selects_the_layout_that_was_raced(
            self, season: int, circuit_id: str, length_m: float) -> None:
        assert official_lap_length_m(season, circuit_id) == length_m

    @pytest.mark.parametrize("circuit_id,season,length_m", [
        # Each with a season F1 actually raced there -- the table is keyed by
        # layout validity, so asking for a year the circuit was not on the
        # calendar is refused, and should be.
        ("hockenheimring", 2019, 4574.0),
        ("istanbul", 2020, 5338.0),
        ("mugello", 2020, 5245.0),
        ("nurburgring", 2020, 5148.0),
        ("portimao", 2020, 4653.0),
        ("ricard", 2019, 5842.0),
        ("sochi", 2019, 5848.0),
    ])
    def test_the_circuits_f1_has_left_are_still_resolvable(
            self, circuit_id: str, season: int, length_m: float) -> None:
        """A season is only reproducible if its circuits outlive the calendar."""
        assert official_lap_length_m(season, circuit_id) == length_m


class TestCriterion6TheTableIsSelfConsistent:
    def test_every_layout_has_a_length_a_validity_and_a_source(self) -> None:
        for layout_id, layout in LAYOUTS.items():
            assert layout.length_m > 1000.0, layout_id
            assert layout.first_season >= 1950, layout_id
            assert layout.last_season is None or layout.last_season >= layout.first_season, layout_id
            assert layout.source, f"{layout_id} does not say where its length came from"

    def test_the_id_matches_its_key(self) -> None:
        for layout_id, layout in LAYOUTS.items():
            assert layout.layout_id == layout_id

    def test_no_two_layouts_of_one_circuit_claim_the_same_season(self) -> None:
        """The ambiguity `resolve_layout` refuses, asserted up front."""
        for circuit_id in {layout.circuit_id for layout in LAYOUTS.values()}:
            selectable = [l for l in layouts_for(circuit_id) if not l.override_only]
            for season in range(2018, 2027):
                claiming = [l.layout_id for l in selectable if l.covers(season)]
                assert len(claiming) <= 1, f"{circuit_id} in {season}: {claiming}"

    def test_the_length_is_a_grand_prix_circuit(self) -> None:
        for layout_id, layout in LAYOUTS.items():
            assert 3000.0 < layout.length_m < 8000.0, f"{layout_id}: {layout.length_m} m"

    def test_monaco_is_the_shortest_and_spa_the_longest(self) -> None:
        current = {i: l for i, l in LAYOUTS.items() if l.last_season is None and not l.override_only}
        assert min(current, key=lambda i: current[i].length_m) == "monaco"
        assert max(current, key=lambda i: current[i].length_m) == "spa"


class TestCriterion7TheErrorSaysWhatToDo:
    def test_an_unknown_circuit_names_the_season_and_what_is_known(self) -> None:
        with pytest.raises(UnknownCircuitError) as raised:
            official_lap_length_m(2026, "nordschleife", "Eifel Grand Prix")
        message = str(raised.value)
        assert "nordschleife" in message and "2026" in message
        assert "Eifel Grand Prix" in message
        assert "FIA circuit specification" in message
        assert "suzuka" in message, "the message should list what it does know"

    def test_a_season_before_a_layout_existed_is_refused(self) -> None:
        with pytest.raises(UnknownCircuitError):
            official_lap_length_m(2018, "vegas")

    def test_the_band_is_asymmetric_because_a_racing_line_cuts_apexes(self) -> None:
        low, high = OFFICIAL_LENGTH_BAND_PCT
        assert low < 0 < high and abs(low) > abs(high)

    def test_within_band_accepts_a_shorter_lap_and_refuses_a_longer_one(self) -> None:
        official = official_lap_length_m(2024, "suzuka")
        assert within_official_band(official * 0.99, official)
        assert not within_official_band(official * 1.01, official)


@pytest.mark.data
class TestAgainstTheLapsActuallyDriven:
    """Every entry checked against the circuit's own measured laps.

    This is what makes the table trustworthy rather than merely present: a
    figure copied from the wrong row would put the measured laps outside the
    band, and the session that used it would fail F010's distance check.
    """

    @staticmethod
    def measured() -> list[tuple[str, float, float]]:
        rows = []
        for report in sorted(PROCESSED_ROOT.glob("*/ground_truth_report.json")):
            payload = json.loads(report.read_text(encoding="utf-8"))
            driven = payload.get("acceptance", {}).get("driven_median_m")
            official = payload.get("official_lap_length_m")
            if driven is None or not official:
                continue
            rows.append((report.parent.name, float(driven), float(official)))
        return rows

    def test_there_is_something_to_check(self) -> None:
        if not self.measured():
            pytest.skip("no validated sessions on this machine")

    def test_every_measured_lap_agrees_with_the_length_it_was_loaded_with(self) -> None:
        rows = self.measured()
        if not rows:
            pytest.skip("no validated sessions on this machine")
        wrong = [(session, round(driven, 1), official)
                 for session, driven, official in rows
                 if not within_official_band(driven, official)]
        assert wrong == [], f"measured laps outside the official band: {wrong}"

    def test_criterion_1_every_loaded_session_still_resolves_the_same(self) -> None:
        """The re-key must not move a single one of the 48 loaded lengths.

        The length each session was loaded with is in its own report; the
        layout table must answer with that number for that season and circuit.
        """
        import pandas as pd

        from src.reference.session import load_reference

        rows = self.measured()
        if not rows:
            pytest.skip("no validated sessions on this machine")
        events: dict[int, pd.DataFrame] = {}
        moved = []
        for session, _driven, official in rows:
            # The report names its own snapshot, and the snapshot is where the
            # season and round live -- the processed layer does not repeat them.
            report = json.loads((PROCESSED_ROOT / session / "ground_truth_report.json")
                                .read_text(encoding="utf-8"))
            snapshot = Path(report["snapshot"])
            meta_path = FASTF1_RAW_ROOT / snapshot.parts[-2] / snapshot.name / "session_meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            season, round_number = int(meta["season"]), int(meta["round_number"])
            if season not in events:
                events[season] = load_reference(season).get("dim_event", pd.DataFrame())
            table = events[season]
            if table.empty:
                pytest.skip(f"no reference table for {season}")
            match = table[table["round"].astype(int) == round_number]
            if match.empty:
                continue
            resolved = official_lap_length_m(season, str(match.iloc[0]["circuit_id"]),
                                             meta.get("event_name"))
            if resolved != official:
                moved.append((session, official, resolved))
        assert moved == [], f"the re-key changed a loaded session's length: {moved}"
