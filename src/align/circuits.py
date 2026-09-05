"""Official circuit lengths.

Deliberately *not* derived from the telemetry. Acceptance criterion 2 checks
measured lap length against these numbers, so deriving them from the data being
validated would make the check circular.

Suzuka was entered by hand from the FIA circuit specification when F008 was
built. The rest were collected for F015 from Wikipedia's *List of Formula One
circuits* (current-layout length, read from the wikitext via the MediaWiki API
on 2026-09-05), with Bahrain taken from its own article's infobox because its
row in that table is formatted differently. Global CLAUDE.md 4.6 allows
Wikipedia for reference data of this kind; the cross-check is that its Suzuka
figure agrees exactly with the FIA number already here.

A caution worth leaving in place: a first, naive parse of that table returned
Pescara's 22.835 km for both Silverstone and Las Vegas, and Silverstone's
5.891 km for Bahrain, because it matched on any link in the row. Every figure
below is therefore also checked against the data -- `tests/test_circuits.py`
asserts each one lies within F008's racing-line band of that circuit's own
measured laps, for every session ingested.
"""

from __future__ import annotations

#: Location (as reported by FastF1's event data) -> official lap length in
#: metres. Keyed by `Location` because that is what `session_meta.json` carries
#: and what `load_track_reference` looks up.
OFFICIAL_LAP_LENGTH_M: dict[str, float] = {
    # 2024 calendar, in round order.
    "Sakhir": 5412.0,
    "Jeddah": 6174.0,
    "Melbourne": 5278.0,
    "Suzuka": 5807.0,
    "Shanghai": 5451.0,
    "Miami": 5412.0,
    "Imola": 4909.0,
    "Monaco": 3337.0,
    "Montréal": 4361.0,
    "Barcelona": 4657.0,
    "Spielberg": 4318.0,
    "Silverstone": 5891.0,
    "Budapest": 4381.0,
    "Spa-Francorchamps": 7004.0,
    "Zandvoort": 4259.0,
    "Monza": 5793.0,
    "Baku": 6003.0,
    "Marina Bay": 4940.0,
    "Austin": 5513.0,
    "Mexico City": 4304.0,
    "São Paulo": 4309.0,
    "Las Vegas": 6201.0,
    "Lusail": 5419.0,
    "Yas Island": 5281.0,
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
    """No official length recorded for this circuit."""


def official_lap_length_m(location: str) -> float:
    try:
        return OFFICIAL_LAP_LENGTH_M[location]
    except KeyError as exc:
        raise UnknownCircuitError(
            f"no official lap length recorded for {location!r}; add it to "
            "OFFICIAL_LAP_LENGTH_M from the FIA circuit specification rather "
            "than deriving it from telemetry"
        ) from exc


def within_official_band(measured_m: float, location: str) -> bool:
    """Whether a measured lap length is consistent with the official figure."""
    official = official_lap_length_m(location)
    low, high = OFFICIAL_LENGTH_BAND_PCT
    percent = (measured_m - official) / official * 100.0
    return low <= percent <= high
