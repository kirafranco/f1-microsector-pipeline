"""Lap and sector times reconstructed from the grid, against official timing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics.reference import lap_index

RECONSTRUCTION_COLUMNS = (
    "lap_grid_s",
    "s1_grid_s",
    "s2_grid_s",
    "s3_grid_s",
    "start_extrap_m",
    "end_extrap_m",
)


def _time_at(distance: np.ndarray, time_s: np.ndarray, speed_ms: np.ndarray, target: float) -> float:
    """Time at a distance, extending beyond the grid at the end sample's speed."""
    if target < distance[0]:
        return float(time_s[0] - (distance[0] - target) / speed_ms[0])
    if target > distance[-1]:
        return float(time_s[-1] + (target - distance[-1]) / speed_ms[-1])
    return float(np.interp(target, distance, time_s))


def reconstruct_laps(
    curves: pd.DataFrame,
    speeds: pd.DataFrame,
    laps: pd.DataFrame,
    d_start: float,
    d_end: float,
    s1_m: float,
    s2_m: float,
    grid_m: float = 10.0,
) -> pd.DataFrame:
    """Grid lap and sector times between the timing-line positions, with residuals.

    ``curves`` is F004's wide ``t_s`` table and ``speeds`` the matching wide
    speed table, both indexed by lap key. Sector splits use F008's S1/S2
    positions on the same axis.
    """
    official = laps.copy()
    official["driver"] = official["driver"].astype(str)
    official["lap_number"] = official["lap_number"].astype(int)
    official = official.set_index(["driver", "lap_number"]).reindex(curves.index)

    values = curves.to_numpy()
    speed_values = speeds.reindex(index=curves.index, columns=curves.columns).to_numpy() / 3.6
    distance = curves.columns.to_numpy(dtype=float) * grid_m

    rows: list[dict] = []
    for i, key in enumerate(curves.index):
        finite = np.isfinite(values[i])
        n = int(finite.sum())
        if n < 2:
            continue
        d = distance[:n]
        t = values[i, :n]
        v = np.where(speed_values[i, :n] > 0, speed_values[i, :n], np.nan)
        v = np.nan_to_num(v, nan=float(np.nanmedian(v)) if np.isfinite(v).any() else 1.0)

        t_start = _time_at(d, t, v, d_start)
        t_end = _time_at(d, t, v, d_end)
        t_s1 = _time_at(d, t, v, s1_m)
        t_s2 = _time_at(d, t, v, s2_m)
        record = official.loc[key]
        rows.append(
            {
                "driver": key[0],
                "lap_number": key[1],
                "lap_grid_s": t_end - t_start,
                "s1_grid_s": t_s1 - t_start,
                "s2_grid_s": t_s2 - t_s1,
                "s3_grid_s": t_end - t_s2,
                "lap_time_s": float(record.get("lap_time", np.nan)),
                "s1_official_s": float(record.get("sector1_time", np.nan)),
                "s2_official_s": float(record.get("sector2_time", np.nan)),
                "s3_official_s": float(record.get("sector3_time", np.nan)),
                "n_points": n,
                "start_extrap_m": max(0.0, d[0] - d_start),
                "end_extrap_m": max(0.0, d_end - d[-1]),
            }
        )

    out = pd.DataFrame(rows)
    out["lap_residual_s"] = out["lap_grid_s"] - out["lap_time_s"]
    for name in ("s1", "s2", "s3"):
        out[f"{name}_residual_s"] = out[f"{name}_grid_s"] - out[f"{name}_official_s"]
    return out


def closure_residuals(reconstructed: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """Δt closure: grid lap-time difference minus the official one, per lap.

    NaN on a lap that is its own reference -- the residual is zero by
    construction there and would flatter the statistics.
    """
    indexed = reconstructed.set_index(lap_index(zip(reconstructed["driver"], reconstructed["lap_number"])))
    ref_keys = lap_index(reference.reindex(indexed.index).tolist())
    ref_grid = indexed["lap_grid_s"].reindex(ref_keys).to_numpy(dtype=float)
    ref_official = indexed["lap_time_s"].reindex(ref_keys).to_numpy(dtype=float)

    is_reference = np.array([k == r for k, r in zip(indexed.index, ref_keys)])
    residual = (indexed["lap_grid_s"].to_numpy(dtype=float) - ref_grid) - (
        indexed["lap_time_s"].to_numpy(dtype=float) - ref_official
    )
    out = reconstructed.copy()
    out["closure_residual_s"] = np.where(is_reference, np.nan, residual)
    out["is_reference"] = is_reference
    return out
