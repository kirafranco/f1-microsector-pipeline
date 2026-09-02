"""Official circuit lengths.

Hand-entered from FIA circuit specifications, deliberately *not* derived from
the telemetry. Acceptance criterion 2 checks measured lap length against these
numbers, so deriving them from the data being validated would make the check
circular.
"""

from __future__ import annotations

#: Location (as reported by FastF1's event data) -> official lap length in metres.
OFFICIAL_LAP_LENGTH_M: dict[str, float] = {
    "Suzuka": 5807.0,
}


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
