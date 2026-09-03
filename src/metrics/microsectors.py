"""Micro-sector times per lap and their spread across laps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics.delta import lap_lengths, reference_curves
from src.metrics.reference import lap_index

TIMES_SCHEMA: dict[str, str] = {
    "driver": "string",
    "lap_number": "Int16",
    "grain": "string",
    "microsector_id": "Int16",
    "time_s": "float32",
    "delta_s": "float32",
    "partial": "boolean",
    "length_m": "float32",
}

SUMMARY_SCHEMA: dict[str, str] = {
    "grain": "string",
    "microsector_id": "Int16",
    "phase": "string",
    "event_id": "Int16",
    "corners": "string",
    "length_m": "float32",
    "n_laps": "Int16",
    "ref_s": "float32",
    "mean_s": "float32",
    "std_s": "float32",
    "min_s": "float32",
    "p10_s": "float32",
    "within_driver_std_s": "float32",
}


def _sector_matrix(curves: pd.DataFrame, microsectors: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(laps x sectors)`` time and partial flag.

    ``time = t(min(end_index, n - 1)) - t(start_index)``; NaN when the lap
    ends before the sector starts; ``partial`` when it ends inside the sector.
    """
    values = curves.to_numpy()
    n = lap_lengths(curves).to_numpy()
    start = microsectors["start_index"].to_numpy(dtype=int)
    end = microsectors["end_index"].to_numpy(dtype=int)
    if (end > values.shape[1]).any():
        raise ValueError("microsector end_index beyond the grid")

    last = (n - 1)[:, None]
    end_clipped = np.minimum(end[None, :], last)
    rows = np.arange(len(values))[:, None]
    valid = start[None, :] < n[:, None]
    start_b = np.broadcast_to(start[None, :], end_clipped.shape)
    time = values[rows, end_clipped] - values[rows, np.where(valid, start_b, 0)]
    time = np.where(valid, time, np.nan)
    # Flagged whenever the lap ends before the sector does -- including the
    # case where it ends before the sector even starts and the time is NaN.
    partial = end[None, :] > last
    return time, partial


def microsector_times(
    curves: pd.DataFrame, microsectors: pd.DataFrame, reference: pd.Series
) -> pd.DataFrame:
    """Long table typed per :data:`TIMES_SCHEMA`, one row per lap and sector."""
    time, partial = _sector_matrix(curves, microsectors)
    ref_rows = lap_index(reference.reindex(curves.index).tolist())
    ref_pos = curves.index.get_indexer(ref_rows)
    if (ref_pos < 0).any():
        raise ValueError("reference lap not on the grid")
    delta = time - time[ref_pos]

    n_laps, n_sec = time.shape
    long = pd.DataFrame(
        {
            "driver": np.repeat(curves.index.get_level_values("driver").to_numpy(), n_sec),
            "lap_number": np.repeat(curves.index.get_level_values("lap_number").to_numpy(), n_sec),
            "grain": np.tile(microsectors["grain"].to_numpy(), n_laps),
            "microsector_id": np.tile(microsectors["microsector_id"].to_numpy(), n_laps),
            "time_s": time.ravel(),
            "delta_s": delta.ravel(),
            "partial": partial.ravel(),
            "length_m": np.tile((microsectors["end_m"] - microsectors["start_m"]).to_numpy(dtype=float), n_laps),
        }
    )
    return long.astype(TIMES_SCHEMA)


def summarise_microsectors(
    times: pd.DataFrame, microsectors: pd.DataFrame, reference_key: tuple | None
) -> pd.DataFrame:
    """Per-sector spread across laps, typed per :data:`SUMMARY_SCHEMA`.

    ``std_s`` is the micro-sector variance metric of the brief;
    ``within_driver_std_s`` is the median across drivers of each driver's own
    std (drivers with one lap contribute nothing). ``ref_s`` is the time on
    ``reference_key`` -- for ``driver_best`` the caller passes the session
    fastest so the column stays single-valued.
    """
    complete = times[times["time_s"].notna() & ~times["partial"].fillna(False).astype(bool)]
    keys = ["grain", "microsector_id"]
    stats = complete.groupby(keys, observed=True)["time_s"].agg(
        n_laps="size",
        mean_s="mean",
        std_s="std",
        min_s="min",
        p10_s=lambda s: float(np.percentile(s.to_numpy(dtype=float), 10)),
    )
    per_driver = complete.groupby(["driver", *keys], observed=True)["time_s"].agg(["std", "size"])
    per_driver = per_driver[per_driver["size"] >= 2]["std"]
    within = per_driver.groupby(level=keys).median().rename("within_driver_std_s")

    if reference_key is not None:
        driver, lap_number = str(reference_key[0]), int(reference_key[1])
        ref = times[(times["driver"] == driver) & (times["lap_number"] == lap_number)]
        ref_s = ref.set_index(keys)["time_s"].rename("ref_s")
    else:
        ref_s = pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=keys), name="ref_s")

    meta = microsectors.set_index(keys)[["phase", "event_id", "corners", "start_m", "end_m"]]
    out = meta.join(stats, how="left").join(within, how="left").join(ref_s, how="left")
    out["length_m"] = out["end_m"] - out["start_m"]
    out = out.drop(columns=["start_m", "end_m"]).reset_index()
    out["n_laps"] = out["n_laps"].fillna(0)
    return out[list(SUMMARY_SCHEMA)].astype(SUMMARY_SCHEMA)
