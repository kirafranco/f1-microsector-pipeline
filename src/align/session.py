"""Session-level alignment: raw snapshot in, aligned interim layer out."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.align import validation
from src.align.circuits import official_lap_length_m
from src.align.merge import MergeError, build_lap_frame
from src.align.track_reference import (
    DEFAULT_MIN_ANCHORS,
    AlignedLap,
    TrackReference,
    align_lap,
    split_anchor_and_holdout,
)
from src.align.units import positions_to_metres
from src.config import INTERIM_ROOT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignmentResult:
    root: Path
    laps_total: int
    laps_aligned: int
    laps_rejected: int
    median_residual_m: float
    median_lap_length_m: float
    lap_length_error_pct: float
    registration_aligned: pd.DataFrame
    registration_raw: pd.DataFrame

    @property
    def aligned_fraction(self) -> float:
        return self.laps_aligned / self.laps_total if self.laps_total else 0.0


def load_track_reference(snapshot_root: Path) -> TrackReference:
    corners = pd.read_parquet(snapshot_root / "circuit_corners.parquet")
    meta = json.loads((snapshot_root / "session_meta.json").read_text(encoding="utf-8"))
    location = meta.get("location") or meta.get("event_name") or "unknown"
    return TrackReference(
        circuit=location,
        # Corner X/Y arrive in the same 1/10 m units as position data; the
        # distance channel is already metres and is left alone.
        corners=positions_to_metres(corners),
        lap_length_m=official_lap_length_m(location),
    )


def select_flying_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Accurate, non-in/out laps — the only ones worth aligning."""
    mask = (
        laps["is_accurate"].fillna(False).astype(bool)
        & laps["lap_time"].notna()
        & laps["pit_in_time"].isna()
        & laps["pit_out_time"].isna()
    )
    return laps.loc[mask].reset_index(drop=True)


def align_session(
    snapshot_root: Path,
    out_root: Path | None = None,
    min_anchors: int = DEFAULT_MIN_ANCHORS,
) -> AlignmentResult:
    """Align every flying lap in a raw snapshot and write the interim layer."""
    reference = load_track_reference(snapshot_root)
    anchor_numbers, holdout_numbers = split_anchor_and_holdout(reference)
    logger.info(
        "alignment_start circuit=%s corners=%d anchors=%d holdout=%d",
        reference.circuit,
        len(reference.corners),
        len(anchor_numbers),
        len(holdout_numbers),
    )

    laps = select_flying_laps(pd.read_parquet(snapshot_root / "laps.parquet"))
    car = pd.read_parquet(snapshot_root / "car_telemetry.parquet")
    pos = pd.read_parquet(snapshot_root / "pos_data.parquet")

    car_groups = dict(tuple(car.groupby(["driver", "lap_number"], observed=True)))
    pos_groups = dict(tuple(pos.groupby(["driver", "lap_number"], observed=True)))

    aligned_frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    anchor_rows: list[pd.DataFrame] = []
    rejected: list[dict] = []

    for lap in laps.itertuples(index=False):
        key = (lap.driver, lap.lap_number)
        if key not in car_groups or key not in pos_groups:
            rejected.append({"driver": lap.driver, "lap_number": lap.lap_number,
                             "reason": "no telemetry for this lap"})
            continue

        try:
            frame = build_lap_frame(car_groups[key], pos_groups[key])
        except MergeError as exc:
            rejected.append({"driver": lap.driver, "lap_number": lap.lap_number,
                             "reason": f"merge failed: {exc}"})
            continue

        result: AlignedLap = align_lap(
            frame, reference, min_anchors=min_anchors, anchor_corner_numbers=anchor_numbers
        )
        if result.rejected:
            rejected.append({"driver": lap.driver, "lap_number": lap.lap_number,
                             "reason": result.reject_reason})
            continue

        aligned_frames.append(result.telemetry)
        raw_frames.append(frame)
        anchors = result.anchors.copy()
        anchors.insert(0, "lap_number", lap.lap_number)
        anchors.insert(0, "driver", lap.driver)
        anchor_rows.append(anchors)

    if not aligned_frames:
        raise RuntimeError(f"{snapshot_root}: no lap aligned ({len(rejected)} rejected)")

    telemetry = pd.concat(aligned_frames, ignore_index=True)
    anchors_out = pd.concat(anchor_rows, ignore_index=True)
    rejected_out = pd.DataFrame(rejected, columns=["driver", "lap_number", "reason"])

    lap_lengths = np.array([float(f["distance_aligned"].iloc[-1]) for f in aligned_frames])
    median_length = float(np.median(lap_lengths))

    registration_aligned = validation.measure_registration(
        aligned_frames, reference, holdout_numbers, "distance_aligned"
    )
    registration_raw = validation.measure_registration(
        raw_frames, reference, holdout_numbers, "distance_raw"
    )

    out_root = out_root or (INTERIM_ROOT / "aligned" / snapshot_root.name)
    out_root.mkdir(parents=True, exist_ok=True)
    telemetry.to_parquet(out_root / "telemetry_aligned.parquet", index=False)
    anchors_out.to_parquet(out_root / "anchors.parquet", index=False)
    rejected_out.to_parquet(out_root / "rejected_laps.parquet", index=False)

    result = AlignmentResult(
        root=out_root,
        laps_total=len(laps),
        laps_aligned=len(aligned_frames),
        laps_rejected=len(rejected),
        median_residual_m=float(anchors_out["residual_m"].median()),
        median_lap_length_m=median_length,
        lap_length_error_pct=100.0
        * abs(median_length - reference.lap_length_m)
        / reference.lap_length_m,
        registration_aligned=validation.summarise_registration(registration_aligned),
        registration_raw=validation.summarise_registration(registration_raw),
    )

    logger.info(
        "alignment_complete aligned=%d/%d rejected=%d median_residual_m=%.2f "
        "median_lap_length_m=%.1f error_pct=%.3f",
        result.laps_aligned,
        result.laps_total,
        result.laps_rejected,
        result.median_residual_m,
        result.median_lap_length_m,
        result.lap_length_error_pct,
    )
    return result
