"""F025 criterion 2: every season FastF1 covers resolves to a layout.

The one place this project deliberately depends on live external data, and the
reason is that the failure it catches cannot be caught any other way. FastF1
renames circuits between seasons -- "Miami" became "Miami Gardens" for 2025 --
and a schedule is published months before the first session. This test fails
the day the rename appears, not the day someone tries to ingest the race.

Opt-in: `pytest -m network`. Both sources are free and official-or-established
(global CLAUDE.md 4.0): FastF1 for the calendar, Jolpica for the stable circuit
id. Both are cached under `data/cache/`, so a re-run costs no requests.

Before F025 this resolved **2 of 8 seasons and 150 of 173 events**.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
import warnings
from pathlib import Path

import pytest

from src.align.circuits import UnknownCircuitError, resolve_layout
from src.config import DATA_ROOT, FASTF1_CACHE_DIR

pytestmark = pytest.mark.network

#: FastF1 carries telemetry from 2018; earlier seasons have nothing to resample.
FIRST_SEASON = 2018
LAST_SEASON = 2025

JOLPICA_CACHE = DATA_ROOT / "cache" / "jolpica"


def jolpica_races(season: int) -> list[dict]:
    """That season's races, cached. Jolpica is rate-limited; be polite."""
    JOLPICA_CACHE.mkdir(parents=True, exist_ok=True)
    path = JOLPICA_CACHE / f"{season}_races.json"
    if not path.exists():
        url = f"https://api.jolpi.ca/ergast/f1/{season}/races.json?limit=100"
        with urllib.request.urlopen(url, timeout=30) as response:
            path.write_bytes(response.read())
        time.sleep(1.0)
    return json.loads(path.read_text(encoding="utf-8"))["MRData"]["RaceTable"]["Races"]


@pytest.fixture(scope="module")
def schedules() -> dict[int, object]:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    import fastf1

    FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))
    return {season: fastf1.get_event_schedule(season, include_testing=False)
            for season in range(FIRST_SEASON, LAST_SEASON + 1)}


@pytest.fixture(scope="module")
def circuit_ids() -> dict[tuple[int, int], tuple[str, str]]:
    """`(season, round)` -> `(circuit_id, race_name)`, from Jolpica."""
    mapping = {}
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        for race in jolpica_races(season):
            mapping[(season, int(race["round"]))] = (race["Circuit"]["circuitId"], race["raceName"])
    return mapping


class TestEverySeasonResolves:
    def test_the_sources_agree_on_how_many_rounds_each_season_had(
            self, schedules: dict, circuit_ids: dict) -> None:
        """If they disagree the join below is meaningless, so check it first."""
        mismatched = []
        for season, schedule in schedules.items():
            jolpica = sum(1 for (s, _r) in circuit_ids if s == season)
            if len(schedule) != jolpica:
                mismatched.append((season, len(schedule), jolpica))
        assert mismatched == [], f"(season, fastf1, jolpica): {mismatched}"

    def test_every_event_in_every_season_resolves_to_a_layout(
            self, schedules: dict, circuit_ids: dict) -> None:
        """Criterion 2. Was 2/8 seasons and 150/173 events before F025."""
        unresolved: list[str] = []
        resolved = 0
        for season, schedule in schedules.items():
            for _, event in schedule.iterrows():
                round_number = int(event["RoundNumber"])
                known = circuit_ids.get((season, round_number))
                if known is None:
                    unresolved.append(f"{season} R{round_number}: not in Jolpica")
                    continue
                circuit_id, race_name = known
                try:
                    layout = resolve_layout(season, circuit_id, race_name)
                except UnknownCircuitError as exc:
                    unresolved.append(f"{season} R{round_number} {event['Location']}: {exc}"[:150])
                    continue
                if layout.length_m <= 0:
                    unresolved.append(f"{season} R{round_number}: no length")
                    continue
                resolved += 1
        assert unresolved == [], f"{len(unresolved)} events do not resolve: {unresolved[:6]}"
        assert resolved >= 160, f"only {resolved} events checked; the sweep is too small"

    def test_the_2020_sakhir_grand_prix_gets_the_outer_loop(self, circuit_ids: dict) -> None:
        """The case a circuit-keyed table could not express, on live data."""
        sakhir = [(r, name) for (s, r), (cid, name) in circuit_ids.items()
                  if s == 2020 and cid == "bahrain"]
        assert len(sakhir) == 2, f"2020 should have two Bahrain events, found {sakhir}"
        lengths = {name: resolve_layout(2020, "bahrain", name).length_m for _r, name in sakhir}
        assert lengths == {"Bahrain Grand Prix": 5412.0, "Sakhir Grand Prix": 3543.0}

    def test_a_rename_does_not_change_the_circuit_id(self, circuit_ids: dict) -> None:
        """The property the whole re-key rests on, asserted against the source."""
        for cid, seasons in (("miami", (2024, 2025)), ("monaco", (2018, 2024)),
                             ("marina_bay", (2018, 2024)), ("yas_marina", (2018, 2024))):
            for season in seasons:
                assert any(c == cid for (s, _r), (c, _n) in circuit_ids.items() if s == season), \
                    f"{cid} absent from {season}"
