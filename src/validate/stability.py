"""V_min repeatability per driver, compound and corner (F010 criterion 4).

A driver on one compound should reach nearly the same minimum speed at a given
corner lap after lap. The spread across those laps is a physical consistency
measure, and it is also the tightest available check that the corner windows
from F009 are picking out the same place every time.
"""

from __future__ import annotations

import pandas as pd

STABILITY_SCHEMA: dict[str, str] = {
    "driver": "string",
    "compound": "string",
    "event_id": "Int16",
    "corners": "string",
    "n_laps": "Int16",
    "v_min_mean_kmh": "float32",
    "v_min_std_kmh": "float32",
}

DEFAULT_MIN_LAPS = 3


def v_min_stability(
    corner_metrics: pd.DataFrame, laps: pd.DataFrame, min_laps: int = DEFAULT_MIN_LAPS
) -> pd.DataFrame:
    """One row per ``(driver, compound, event)`` with at least ``min_laps`` laps."""
    tyres = laps[["driver", "lap_number", "compound"]].copy()
    tyres["driver"] = tyres["driver"].astype(str)
    tyres["lap_number"] = tyres["lap_number"].astype(int)

    metrics = corner_metrics.copy()
    metrics["driver"] = metrics["driver"].astype(str)
    metrics["lap_number"] = metrics["lap_number"].astype(int)
    joined = metrics.merge(tyres, on=["driver", "lap_number"], how="left")

    grouped = joined.groupby(["driver", "compound", "event_id"], observed=True, dropna=False)
    out = grouped.agg(
        corners=("corners", "first"),
        n_laps=("v_min_kmh", "size"),
        v_min_mean_kmh=("v_min_kmh", "mean"),
        v_min_std_kmh=("v_min_kmh", "std"),
    ).reset_index()
    out = out[out["n_laps"] >= min_laps].sort_values(["driver", "compound", "event_id"]).reset_index(drop=True)
    return out[list(STABILITY_SCHEMA)].astype(STABILITY_SCHEMA)
