"""Registration measurement at held-out corners.

The alignment forces the lap's distance axis onto the reference distances of
the corners it anchors on, so measuring registration at those corners measures
nothing. These functions measure it at corners the alignment never saw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.align.track_reference import TrackReference


def measure_registration(
    laps: "list[pd.DataFrame]",
    reference: TrackReference,
    corner_numbers: "list[int]",
    column: str,
) -> pd.DataFrame:
    """Where each lap's closest approach to each corner lands on `column`.

    `column` is `distance_aligned` for the aligned axis, or `distance_raw` for
    the unaligned baseline. Both are measured against the corner's reference
    distance, so the two are directly comparable.
    """
    corners = reference.corners[reference.corners["number"].isin(corner_numbers)]
    corners = corners.sort_values("distance")

    rows: list[dict] = []
    for frame in laps:
        if frame.empty:
            continue
        xs = frame["x"].to_numpy(dtype=float)
        ys = frame["y"].to_numpy(dtype=float)
        values = frame[column].to_numpy(dtype=float)
        driver = frame["driver"].iloc[0]
        lap_number = int(frame["lap_number"].iloc[0])

        for corner in corners.itertuples(index=False):
            dx = xs - float(corner.x)
            dy = ys - float(corner.y)
            squared = dx * dx + dy * dy
            index = int(np.argmin(squared))
            rows.append(
                {
                    "driver": driver,
                    "lap_number": lap_number,
                    "corner_number": int(corner.number),
                    "d_ref": float(corner.distance),
                    "observed": float(values[index]),
                    "error_m": float(values[index]) - float(corner.distance),
                    "xy_residual_m": float(np.sqrt(squared[index])),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "driver",
            "lap_number",
            "corner_number",
            "d_ref",
            "observed",
            "error_m",
            "xy_residual_m",
        ],
    )


def summarise_registration(observations: pd.DataFrame) -> pd.DataFrame:
    """Per-corner spread across laps: the 'same physical place' statistic."""
    if observations.empty:
        return pd.DataFrame(
            columns=["corner_number", "n_laps", "max_abs_error_m", "spread_m", "std_m"]
        )

    grouped = observations.groupby("corner_number")
    summary = pd.DataFrame(
        {
            "n_laps": grouped.size(),
            "max_abs_error_m": grouped["error_m"].apply(lambda s: s.abs().max()),
            "spread_m": grouped["observed"].max() - grouped["observed"].min(),
            "std_m": grouped["observed"].std(ddof=0),
        }
    ).reset_index()
    return summary.sort_values("corner_number").reset_index(drop=True)
