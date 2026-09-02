"""Unit conventions for position data.

FastF1 delivers X/Y/Z position in units of 1/10 metre, while the corner
`distance` channel is in metres. The raw layer (F002) stores both exactly as
delivered, which is correct for an immutable snapshot; converting is this
layer's job, and it must happen on *both* sides — lap positions and corner
positions — or residuals are compared across a factor of ten.

Measured on Suzuka 2024 Q, comparing X/Y path length against speed-integrated
distance over 40 laps: 10.0003 units per metre (min 9.9656, max 10.0535).
"""

from __future__ import annotations

import pandas as pd

#: FastF1 position units per metre.
POSITION_UNITS_PER_METRE: float = 10.0

POSITION_COLUMNS = ("x", "y", "z")


def positions_to_metres(frame: pd.DataFrame, columns=POSITION_COLUMNS) -> pd.DataFrame:
    """Convert position columns from FastF1 units to metres. Does not mutate."""
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].astype(float) / POSITION_UNITS_PER_METRE
    return out
