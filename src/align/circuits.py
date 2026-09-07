"""Official circuit lengths, keyed by layout rather than by a display string.

Deliberately *not* derived from the telemetry. Acceptance criterion 2 checks
measured lap length against these numbers, so deriving them from the data being
validated would make the check circular.

**Why layouts and not locations (F025).** Until this feature the table was keyed
by FastF1's ``Location`` string, and that turned out to be one rename away from
failing: FastF1 called Miami "Miami" in 2024 and "Miami Gardens" in 2025, and
called Monaco "Monte Carlo", Marina Bay "Singapore" and Yas Island "Yas Marina"
in earlier seasons. Only 2023 and 2024 resolved. The key is now a `layout_id`
this project owns, built on Jolpica's ``circuitId``, which was checked
unchanged for every circuit F1 visited from 2018 to 2025.

A circuit is not enough either. In 2020 the Bahrain Grand Prix ran the 5,412 m
grand prix circuit and the Sakhir Grand Prix ran the 3,543 m outer loop, same
locality and same ``circuitId``. A length per *layout*, with the seasons it was
used, is the smallest thing that can say that -- and `LAYOUT_OVERRIDES` names
the event that picked the unusual one.

**Sources.** Suzuka was entered by hand from the FIA circuit specification when
F008 was built. The rest come from Wikipedia's circuit articles, read from the
infobox `layoutN` / `lengthN_km` fields through the MediaWiki API (2026-09-05
for the current layouts, 2026-09-07 for the historical ones), which global
CLAUDE.md 4.6 allows for reference data of this kind. The cross-check is that
its Suzuka figure agrees exactly with the FIA number already here, and that
`tests/test_circuits.py` asserts every entry lies inside F008's racing-line band
of that circuit's own measured laps for every session ingested.

A caution worth leaving in place: a first, naive parse of that table returned
Pescara's 22.835 km for both Silverstone and Las Vegas, and Silverstone's
5.891 km for Bahrain, because it matched on any link in the row. A second naive
parse, for F025, matched race-report citations instead of layout fields. Both
were caught by reading the output rather than trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Where a length came from, so a wrong number can be traced to its source.
FIA = "FIA circuit specification (F008, entered by hand)"
WIKI = "en.wikipedia.org circuit infobox, MediaWiki API, 2026-09-05"
WIKI_HISTORIC = "en.wikipedia.org circuit infobox, MediaWiki API, 2026-09-07 (F025)"


@dataclass(frozen=True)
class Layout:
    """One configuration of one circuit, and the seasons it was raced on.

    `circuit_id` is Jolpica's, which is stable across FastF1's renames.
    `last_season` is None while the layout is current.
    """

    layout_id: str
    circuit_id: str
    name: str
    length_m: float
    first_season: int
    last_season: int | None
    source: str
    #: A layout only one event ever used, which shares its `circuit_id` with
    #: the circuit's usual one. It must never be chosen by the season default,
    #: or every event at that circuit in that season would be ambiguous -- so
    #: it is reachable through `LAYOUT_OVERRIDES` alone.
    override_only: bool = False

    def covers(self, season: int) -> bool:
        return self.first_season <= season and (self.last_season is None or season <= self.last_season)


def _layout(layout_id, circuit_id, name, length_m, first_season, last_season=None,
            source=WIKI, override_only=False):
    return Layout(layout_id, circuit_id, name, length_m, first_season, last_season,
                  source, override_only)


#: Every layout F1 has raced from 2018, keyed by `layout_id`. Circuits whose
#: configuration never changed in that window have a single entry with
#: `first_season` set to when the current layout appeared.
LAYOUTS: dict[str, Layout] = {layout.layout_id: layout for layout in (
    # --- unchanged through 2018-2025 -------------------------------------
    _layout("suzuka", "suzuka", "Grand Prix Circuit", 5807.0, 2003, source=FIA),
    _layout("bahrain", "bahrain", "Grand Prix Circuit", 5412.0, 2005),
    _layout("jeddah", "jeddah", "Grand Prix Circuit", 6174.0, 2021),
    _layout("shanghai", "shanghai", "Grand Prix Circuit", 5451.0, 2004),
    _layout("miami", "miami", "Grand Prix Circuit", 5412.0, 2022),
    _layout("imola", "imola", "Grand Prix Circuit", 4909.0, 2008),
    _layout("monaco", "monaco", "Grand Prix Circuit", 3337.0, 2015),
    _layout("villeneuve", "villeneuve", "Grand Prix Circuit", 4361.0, 2002),
    _layout("silverstone", "silverstone", "Arena Grand Prix Circuit", 5891.0, 2011),
    _layout("hungaroring", "hungaroring", "Grand Prix Circuit", 4381.0, 2003),
    _layout("spa", "spa", "Grand Prix Circuit", 7004.0, 2007),
    _layout("zandvoort", "zandvoort", "Grand Prix Circuit", 4259.0, 2020),
    _layout("monza", "monza", "Modern Grand Prix Circuit", 5793.0, 2000),
    _layout("baku", "baku", "Grand Prix Circuit", 6003.0, 2016),
    _layout("americas", "americas", "Grand Prix Circuit", 5513.0, 2012),
    _layout("rodriguez", "rodriguez", "Grand Prix Circuit", 4304.0, 2015),
    _layout("interlagos", "interlagos", "Grand Prix Circuit (5th Variation)", 4309.0, 1999),
    _layout("vegas", "vegas", "Grand Prix Circuit", 6201.0, 2023),

    # --- raced before 2022 only ------------------------------------------
    _layout("hockenheimring", "hockenheimring", "Grand Prix Circuit", 4574.0, 2002, source=WIKI_HISTORIC),
    _layout("istanbul", "istanbul", "Grand Prix Circuit", 5338.0, 2005, source=WIKI_HISTORIC),
    _layout("mugello", "mugello", "Grand Prix Circuit", 5245.0, 1974, source=WIKI_HISTORIC),
    _layout("nurburgring", "nurburgring", "GP-Strecke", 5148.0, 2001, source=WIKI_HISTORIC),
    _layout("portimao", "portimao", "Grand Prix Circuit", 4653.0, 2020, source=WIKI_HISTORIC),
    _layout("ricard", "ricard", "Current layout with Mistral chicane (1C-V2)", 5842.0, 2005, source=WIKI_HISTORIC),
    _layout("sochi", "sochi", "Grand Prix Circuit", 5848.0, 2014, 2023, source=WIKI_HISTORIC),

    # --- changed inside 2018-2025 ----------------------------------------
    # Melbourne was reprofiled for 2022; 2020 and 2021 were not raced.
    _layout("albert_park_1996", "albert_park", "Grand Prix Circuit", 5303.0, 1996, 2020, source=WIKI_HISTORIC),
    _layout("albert_park", "albert_park", "Grand Prix Circuit", 5278.0, 2021),
    # Barcelona: the chicane was used in 2021-2022 and removed for 2023.
    _layout("catalunya_2007", "catalunya", "Grand Prix Circuit", 4655.0, 2007, 2020, source=WIKI_HISTORIC),
    _layout("catalunya_chicane", "catalunya", "Grand Prix Circuit with Chicane", 4675.0, 2021, 2022, source=WIKI_HISTORIC),
    _layout("catalunya", "catalunya", "Grand Prix Circuit without Chicane", 4657.0, 2023),
    # Singapore was shortened for 2023 and changed again for 2025.
    _layout("marina_bay_2018", "marina_bay", "Revised Circuit, Turns 16-17 reprofiled", 5063.0, 2018, 2022, source=WIKI_HISTORIC),
    _layout("marina_bay_2023", "marina_bay", "Revised Circuit, new straight Turns 15-16", 4940.0, 2023, 2024),
    _layout("marina_bay", "marina_bay", "Grand Prix Circuit", 4927.0, 2025, source=WIKI_HISTORIC),
    # Abu Dhabi was reprofiled for 2021.
    _layout("yas_marina_2009", "yas_marina", "Grand Prix Circuit", 5554.0, 2009, 2020, source=WIKI_HISTORIC),
    _layout("yas_marina", "yas_marina", "Grand Prix Circuit", 5281.0, 2021),
    # Qatar: 2021 ran the original circuit; F1 returned in 2023 to the revised one.
    _layout("losail_2004", "losail", "Original Grand Prix Circuit", 5380.0, 2004, 2022, source=WIKI_HISTORIC),
    _layout("losail", "losail", "Grand Prix Circuit", 5419.0, 2023),
    # Spielberg gained 8 m for 2025.
    _layout("red_bull_ring_2016", "red_bull_ring", "Grand Prix Circuit", 4318.0, 2016, 2024),
    _layout("red_bull_ring", "red_bull_ring", "Grand Prix Circuit", 4326.0, 2025, source=WIKI_HISTORIC),

    # --- a layout used by one event only ---------------------------------
    # The 2020 Sakhir Grand Prix, on Bahrain's outer loop. Same circuit_id as
    # the grand prix circuit, so only LAYOUT_OVERRIDES can select it.
    _layout("bahrain_outer", "bahrain", "Outer Circuit", 3543.0, 2020, 2020,
            source=WIKI_HISTORIC, override_only=True),
)}


#: `(season, event_name)` -> `layout_id`, for events that did not run their
#: circuit's usual layout. Looked up before the season default, so the
#: exception is data rather than a branch.
LAYOUT_OVERRIDES: dict[tuple[int, str], str] = {
    (2020, "Sakhir Grand Prix"): "bahrain_outer",
}


#: Criterion 2a: how far the aligned axis may drift from speed-integrated
#: distance on the same laps. Both measure the driven path, so this isolates
#: scale error from racing-line geometry.
MAX_SCALE_ERROR_PCT: float = 1.0

#: Criterion 2b: signed band for measured length against the official figure.
#: Asymmetric on purpose -- a racing line cuts apexes and is always shorter than
#: the centreline the official figure measures, so a *longer* result is a defect
#: (scale error, or a wrap double-counting a section), not a driving style.
OFFICIAL_LENGTH_BAND_PCT: tuple[float, float] = (-3.0, 0.2)


class UnknownCircuitError(KeyError):
    """No layout recorded for this circuit in this season."""


def layouts_for(circuit_id: str) -> list[Layout]:
    return [layout for layout in LAYOUTS.values() if layout.circuit_id == circuit_id]


def resolve_layout(season: int, circuit_id: str, event_name: str | None = None) -> Layout:
    """The layout an event raced, from its season and Jolpica circuit id.

    An override wins over the season default, which is how the 2020 Sakhir
    Grand Prix reaches the outer loop while the Bahrain Grand Prix that year
    reaches the grand prix circuit.
    """
    override = LAYOUT_OVERRIDES.get((int(season), str(event_name))) if event_name else None
    if override:
        return LAYOUTS[override]

    candidates = [layout for layout in layouts_for(circuit_id)
                  if layout.covers(int(season)) and not layout.override_only]
    if not candidates:
        known = sorted({layout.circuit_id for layout in LAYOUTS.values()})
        raise UnknownCircuitError(
            f"no layout recorded for circuit_id {circuit_id!r} in {season}"
            f"{f' ({event_name})' if event_name else ''}; add it to LAYOUTS from the FIA "
            f"circuit specification rather than deriving it from telemetry. "
            f"Known circuit ids: {known}"
        )
    if len(candidates) > 1:
        raise UnknownCircuitError(
            f"{len(candidates)} layouts of {circuit_id!r} claim {season}: "
            f"{[c.layout_id for c in candidates]}; their seasons overlap, which is a "
            f"defect in LAYOUTS, or the event needs a LAYOUT_OVERRIDES entry"
        )
    return candidates[0]


def official_lap_length_m(season: int, circuit_id: str, event_name: str | None = None) -> float:
    """The official length of the layout that event raced."""
    return resolve_layout(season, circuit_id, event_name).length_m


def within_official_band(measured_m: float, official_m: float) -> bool:
    """Whether a measured lap length is consistent with the official figure."""
    low, high = OFFICIAL_LENGTH_BAND_PCT
    percent = (measured_m - official_m) / official_m * 100.0
    return low <= percent <= high
