"""Circuit-info corners placed on the aligned distance axis.

Corner X/Y arrive in FastF1's 1/10 m units and in a frame offset from the
position stream (F008 findings). Both corrections are applied here, then each
corner is snapped to the nearest grid point of the session's reference lap and
takes that point's ``distance_m``. The raw circuit ``distance`` channel was
measured to disagree with the aligned axis by -2 to -64 m at Suzuka, so it is
carried for reference only and never used for positioning.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.align.frame import RigidTransform
from src.align.units import positions_to_metres

CORNER_SCHEMA: dict[str, str] = {
    "number": "Int16",
    "letter": "string",
    "distance_m": "float32",
    "line_offset_m": "float32",
    "raw_distance_m": "float32",
}

_REFERENCE_LAP = re.compile(r"^(?P<driver>\S+) L(?P<lap>\d+)$")


class CornerError(ValueError):
    """Corners cannot be placed on the aligned axis."""


def load_frame(meta: dict) -> RigidTransform:
    """Rebuild the F008 frame transform from ``alignment_meta.json``."""
    try:
        frame = meta["frame"]
        return RigidTransform(
            rotation_rad=float(np.deg2rad(frame["rotation_deg"])),
            translation=np.asarray(frame["translation_m"], dtype=float),
            median_residual_m=float(frame["median_residual_m"]),
            iterations=int(frame["iterations"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CornerError(f"alignment_meta.json has no usable frame: {exc}") from exc


def parse_reference_lap(label: str) -> tuple[str, int]:
    """``"VER L11"`` -> ``("VER", 11)``, the F008 reference-line lap."""
    match = _REFERENCE_LAP.match(label.strip())
    if not match:
        raise CornerError(f"cannot parse reference lap label {label!r}")
    return match.group("driver"), int(match.group("lap"))


def corner_positions(
    grid: pd.DataFrame,
    corners: pd.DataFrame,
    frame: RigidTransform,
    reference_lap: tuple[str, int],
) -> pd.DataFrame:
    """Aligned distance of every circuit-info corner, one row per corner.

    ``grid`` is F003 output; ``corners`` is F002's ``circuit_corners.parquet``
    exactly as stored (1/10 m units).
    """
    driver, lap_number = reference_lap
    ref = grid[(grid["driver"] == driver) & (grid["lap_number"] == lap_number)]
    if len(ref) < 2:
        raise CornerError(f"reference lap {driver} L{lap_number} not found in grid")
    ref = ref.sort_values("grid_index")

    required = {"number", "x", "y", "distance"}
    missing = required - set(corners.columns)
    if missing:
        raise CornerError(f"corners frame is missing columns {sorted(missing)}")
    if corners.empty:
        raise CornerError("corners frame is empty")

    corner_xy = frame.apply(positions_to_metres(corners)[["x", "y"]].to_numpy(dtype=float))
    ref_xy = ref[["x", "y"]].to_numpy(dtype=float)
    squared = ((corner_xy[:, None, :] - ref_xy[None, :, :]) ** 2).sum(-1)
    nearest = squared.argmin(axis=1)

    out = pd.DataFrame(
        {
            "number": corners["number"].to_numpy(),
            "letter": corners["letter"].fillna("").to_numpy() if "letter" in corners else "",
            "distance_m": ref["distance_m"].to_numpy(dtype=float)[nearest],
            "line_offset_m": np.sqrt(squared[np.arange(len(corners)), nearest]),
            "raw_distance_m": corners["distance"].to_numpy(dtype=float),
        }
    )
    out = out.sort_values("distance_m").reset_index(drop=True)
    return out.astype(CORNER_SCHEMA)


def corner_label(numbers: list[int], letters: list[str] | None = None) -> str | None:
    """``[1, 2]`` -> ``"T1-T2"``; empty -> ``None``."""
    if not numbers:
        return None
    letters = letters or [""] * len(numbers)
    return "-".join(f"T{n}{letter or ''}" for n, letter in zip(numbers, letters))


_LABEL_PART = re.compile(r"^T(?P<number>\d+)(?P<letter>[A-Za-z]*)$")


def parse_corner_label(label: object) -> list[int]:
    """``"T1-T2"`` -> ``[1, 2]``; null or empty -> ``[]``."""
    if label is None or (isinstance(label, float) and np.isnan(label)) or label is pd.NA:
        return []
    numbers = []
    for part in str(label).split("-"):
        match = _LABEL_PART.match(part.strip())
        if not match:
            raise CornerError(f"malformed corner label {label!r}")
        numbers.append(int(match.group("number")))
    return numbers
