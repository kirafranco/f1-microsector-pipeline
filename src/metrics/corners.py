"""Per-lap corner metrics through the F009 event windows.

V_min and braking point are found inside each event's window, never by reading
a grid point's phase label (the boundaries are session medians). The braking
point carries the F003 ``source_gap_m`` at that grid point as its uncertainty:
D6's "about +/-20 m" measured per row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.grid.resample import GRID_SPACING_M
from src.segment.validation import per_lap_apex, per_lap_brake

CORNER_METRICS_SCHEMA: dict[str, str] = {
    "driver": "string",
    "lap_number": "Int16",
    "event_id": "Int16",
    "corners": "string",
    "marginal": "boolean",
    "v_min_kmh": "float32",
    "v_min_m": "float32",
    "apex_dev_m": "float32",
    "brake_on_m": "float32",
    "brake_dev_m": "float32",
    "brake_gap_m": "float32",
}


def corner_metrics(
    grid: pd.DataFrame, events: pd.DataFrame, lap_length_m: float, grid_m: float = GRID_SPACING_M
) -> pd.DataFrame:
    """One row per lap and event, typed per :data:`CORNER_METRICS_SCHEMA`."""
    if events.empty:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in CORNER_METRICS_SCHEMA.items()})

    apex = per_lap_apex(grid, events, lap_length_m)
    brake = per_lap_brake(grid, events, lap_length_m)
    keys = ["driver", "lap_number", "event_id"]
    for frame in (apex, brake):
        frame["driver"] = frame["driver"].astype(str)
        frame["lap_number"] = frame["lap_number"].astype(int)
        frame["event_id"] = frame["event_id"].astype(int)

    out = apex.merge(brake, on=keys, how="left")
    ev = events.set_index(events["event_id"].astype(int))
    out["corners"] = out["event_id"].map(ev["corners"])
    out["marginal"] = out["event_id"].map(ev["marginal"]).astype("boolean")
    out["v_min_m"] = (out["event_id"].map(ev["apex_m"]).astype(float) + out["apex_dev_m"]) % lap_length_m
    out["brake_on_m"] = (out["event_id"].map(ev["brake_on_m"]).astype(float) + out["brake_dev_m"]) % lap_length_m

    gap = grid[["driver", "lap_number", "grid_index", "source_gap_m"]].copy()
    gap["driver"] = gap["driver"].astype(str)
    gap["lap_number"] = gap["lap_number"].astype(int)
    gap["grid_index"] = gap["grid_index"].astype(int)
    out["grid_index"] = np.where(
        np.isfinite(out["brake_on_m"].to_numpy(dtype=float)),
        np.rint(out["brake_on_m"].to_numpy(dtype=float) / grid_m),
        -1,
    ).astype(int)
    out = out.merge(gap, on=["driver", "lap_number", "grid_index"], how="left").rename(columns={"source_gap_m": "brake_gap_m"})
    out = out.sort_values(keys).reset_index(drop=True)
    return out[list(CORNER_METRICS_SCHEMA)].astype(CORNER_METRICS_SCHEMA)
