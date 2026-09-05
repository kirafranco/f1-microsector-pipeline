"""Session-level alignment: raw snapshot in, aligned interim layer out."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.align import validation
from src.align.circuits import (
    MAX_SCALE_ERROR_PCT,
    OFFICIAL_LENGTH_BAND_PCT,
    official_lap_length_m,
)
from src.align.centreline import ReferenceLine, build_reference_line, project_lap
from src.align.frame import RigidTransform, fit_corner_frame
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

#: Laps pooled to form the driven-line cloud the frame is fitted against.
FRAME_FIT_LAPS = 8

#: Default distance method.
#:
#: "projection" gives arc length along one shared reference line, so the axis is
#: metrically true and independent of FastF1's corner Distance values -- which
#: were measured to disagree with real arc length by std 21 m. "anchors" maps
#: onto those corner distances instead; it scores comparably on cross-lap
#: consistency because the corner error is systematic and every lap inherits it
#: identically, but the resulting axis is a warp of true distance rather than
#: distance. Both are kept so the comparison stays reproducible.
DEFAULT_METHOD = "projection"


@dataclass(frozen=True)
class LengthChecks:
    """Criterion 2, split so each half tests one thing.

    2a compares the aligned axis against speed-integrated distance -- both
    measure the driven path, so it isolates scale error. 2b is a loose,
    deliberately asymmetric sanity check against the official centreline figure.
    """

    aligned_m: float
    driven_m: float
    official_m: float

    @property
    def scale_error_pct(self) -> float:
        return 100.0 * abs(self.aligned_m - self.driven_m) / self.driven_m

    @property
    def official_error_pct(self) -> float:
        """Signed: negative means shorter than official, which is expected."""
        return 100.0 * (self.aligned_m - self.official_m) / self.official_m

    @property
    def scale_ok(self) -> bool:
        return self.scale_error_pct <= MAX_SCALE_ERROR_PCT

    @property
    def official_ok(self) -> bool:
        low, high = OFFICIAL_LENGTH_BAND_PCT
        return low <= self.official_error_pct <= high

    @property
    def ok(self) -> bool:
        return self.scale_ok and self.official_ok


@dataclass(frozen=True)
class AlignmentResult:
    root: Path
    laps_total: int
    laps_aligned: int
    laps_rejected: int
    median_residual_m: float
    median_lap_length_m: float
    length: LengthChecks
    frame: RigidTransform
    method: str
    reference_line_lap: str
    sector_aligned: pd.DataFrame
    sector_raw: pd.DataFrame

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


def calibrate_frame(
    reference: TrackReference,
    frames: "list[pd.DataFrame]",
    anchor_numbers: "list[int]",
) -> tuple[TrackReference, RigidTransform]:
    """Put corner coordinates into the telemetry's own frame.

    Fitted on anchor corners only, so the held-out corners contribute nothing
    to the calibration that their registration is later used to validate.
    """
    path = np.vstack([f[["x", "y"]].to_numpy(dtype=float) for f in frames[:FRAME_FIT_LAPS]])
    anchors = reference.corners[reference.corners["number"].isin(anchor_numbers)]
    transform = fit_corner_frame(anchors[["x", "y"]].to_numpy(dtype=float), path)

    corrected = reference.corners.copy()
    corrected[["x", "y"]] = transform.apply(corrected[["x", "y"]].to_numpy(dtype=float))
    calibrated = TrackReference(
        circuit=reference.circuit, corners=corrected, lap_length_m=reference.lap_length_m
    )
    return calibrated, transform


def build_session_reference_line(
    laps: pd.DataFrame, merged: "list[tuple]"
) -> tuple[ReferenceLine, str]:
    """Reference line from the session's fastest clean lap.

    Any single consistent line satisfies the requirement -- what matters is that
    every lap uses the same one -- but the fastest lap is deterministic and is
    the least likely to contain an off-track excursion.
    """
    fastest = laps.loc[laps["lap_time"].idxmin()]
    key = (fastest["driver"], fastest["lap_number"])
    seed = next(
        (f for row, f in merged if (row.driver, row.lap_number) == key), merged[0][1]
    )
    label = f"{key[0]} L{int(key[1])}"
    return build_reference_line(seed[["x", "y"]].to_numpy(dtype=float)), label


#: How far past the reference line's own origin the axis may be moved to find
#: one every lap covers. A session where some lap's telemetry opens later than
#: this has a lap with a real coverage problem, which F010 flags rather than
#: the axis accommodating it; 150 m is about 1.9 s at racing speed.
MAX_ORIGIN_SHIFT_M = 150.0


def place_axis_origin(frames: "list[pd.DataFrame]") -> "tuple[list[pd.DataFrame], float]":
    """Shift the shared axis so distance zero is covered by nearly every lap.

    The resampler grids from distance zero, and interpolating a lap's time
    channel there needs a real sample at or before it. The reference line's own
    origin is wherever the seed lap's telemetry happened to open, which is an
    accident of that one lap: at Suzuka it opened 30 m past the timing line and
    every other lap therefore had data to spare, while at Bahrain it opened
    2 m past, most laps began 12 m further on, and the first two grid points
    were extrapolated -- a -0.19 s bias on every reconstructed lap time.

    So the origin is placed rather than inherited: distance zero moves to the
    *latest* opening in the session, so that every lap has a real sample at or
    before it. Not a high quantile -- 95 % was tried, and the remaining 5 % of
    laps in the Australian Grand Prix produced a time channel that did not
    increase across their first grid point, which is both a false lap time and
    an F011 contract failure that blocks the whole session from loading.

    Nothing else about the axis changes: it is one subtraction, so every
    distance *between* two points is what it was, and so is every lap and
    sector time built on them.
    """
    if not frames:
        return frames, 0.0
    firsts = np.array([float(f["distance_aligned"].iloc[0]) for f in frames])
    shift = float(firsts.max())
    if shift > MAX_ORIGIN_SHIFT_M:
        logger.warning("axis_origin_shift_capped wanted_m=%.1f cap_m=%.1f", shift, MAX_ORIGIN_SHIFT_M)
        shift = MAX_ORIGIN_SHIFT_M
    if abs(shift) < 1e-9:
        return frames, 0.0

    shifted = []
    for frame in frames:
        out = frame.copy()
        out["distance_aligned"] = out["distance_aligned"] - shift
        shifted.append(out)
    covered = int((firsts <= shift + 1e-9).sum())
    logger.info("axis_origin_placed shift_m=%.1f covered=%d/%d", shift, covered, len(frames))
    return shifted, shift


def align_session(
    snapshot_root: Path,
    out_root: Path | None = None,
    min_anchors: int = DEFAULT_MIN_ANCHORS,
    method: str = DEFAULT_METHOD,
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

    rejected: list[dict] = []
    merged: list[tuple] = []

    # Pass 1: merge channels. The frame fit needs a driven-line cloud, so every
    # lap must exist before any lap can be aligned.
    for lap in laps.itertuples(index=False):
        key = (lap.driver, lap.lap_number)
        if key not in car_groups or key not in pos_groups:
            rejected.append({"driver": lap.driver, "lap_number": lap.lap_number,
                             "reason": "no telemetry for this lap"})
            continue
        try:
            merged.append((lap, build_lap_frame(car_groups[key], pos_groups[key])))
        except MergeError as exc:
            rejected.append({"driver": lap.driver, "lap_number": lap.lap_number,
                             "reason": f"merge failed: {exc}"})

    if not merged:
        raise RuntimeError(f"{snapshot_root}: no lap survived channel merge")

    calibrated, transform = calibrate_frame(
        reference, [frame for _, frame in merged], anchor_numbers
    )

    # Pass 2: build the distance axis.
    reference_line, reference_line_lap = build_session_reference_line(laps, merged)
    aligned_frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    anchor_rows: list[pd.DataFrame] = []

    for lap, frame in merged:
        if method == "projection":
            aligned = project_lap(frame, reference_line)
            anchors = pd.DataFrame(
                columns=["corner_number", "sample_index", "d_raw", "d_ref", "residual_m"]
            )
        elif method == "anchors":
            result: AlignedLap = align_lap(
                frame, calibrated, min_anchors=min_anchors, anchor_corner_numbers=anchor_numbers
            )
            if result.rejected:
                rejected.append({"driver": lap.driver, "lap_number": lap.lap_number,
                                 "reason": result.reject_reason})
                continue
            aligned, anchors = result.telemetry, result.anchors
        else:
            raise ValueError(f"unknown method {method!r}; expected 'projection' or 'anchors'")

        aligned_frames.append(aligned)
        raw_frames.append(frame)
        tagged = anchors.copy()
        tagged.insert(0, "lap_number", lap.lap_number)
        tagged.insert(0, "driver", lap.driver)
        anchor_rows.append(tagged)

    if not aligned_frames:
        raise RuntimeError(f"{snapshot_root}: no lap aligned ({len(rejected)} rejected)")

    origin_shift_m = 0.0
    if method == "projection":
        aligned_frames, origin_shift_m = place_axis_origin(aligned_frames)

    telemetry = pd.concat(aligned_frames, ignore_index=True)
    anchors_out = pd.concat(anchor_rows, ignore_index=True)
    rejected_out = pd.DataFrame(rejected, columns=["driver", "lap_number", "reason"])

    lap_lengths = np.array([
        float(f["distance_aligned"].iloc[-1] - f["distance_aligned"].iloc[0])
        for f in aligned_frames
    ])
    median_length = float(np.median(lap_lengths))
    driven_lengths = np.array([float(f["distance_raw"].iloc[-1]) for f in raw_frames])
    length = LengthChecks(
        aligned_m=median_length,
        driven_m=float(np.median(driven_lengths)),
        official_m=reference.lap_length_m,
    )

    if method == "projection":
        median_residual = float(np.median(telemetry["line_offset_m"]))
    else:
        median_residual = float(anchors_out["residual_m"].median())

    sector_aligned = validation.summarise_sector_crossings(
        validation.measure_sector_crossings(aligned_frames, laps, "distance_aligned")
    )
    sector_raw = validation.summarise_sector_crossings(
        validation.measure_sector_crossings(raw_frames, laps, "distance_raw")
    )

    out_root = out_root or (INTERIM_ROOT / "aligned" / snapshot_root.name)
    out_root.mkdir(parents=True, exist_ok=True)
    telemetry.to_parquet(out_root / "telemetry_aligned.parquet", index=False)
    anchors_out.to_parquet(out_root / "anchors.parquet", index=False)
    rejected_out.to_parquet(out_root / "rejected_laps.parquet", index=False)
    (out_root / "alignment_meta.json").write_text(
        json.dumps(
            {
                "method": method,
                "reference_line_lap": reference_line_lap,
                "reference_line_length_m": reference_line.total_length_m,
                "official_lap_length_m": reference.lap_length_m,
                "length_checks": {
                    "aligned_m": length.aligned_m,
                    "driven_m": length.driven_m,
                    "scale_error_pct": length.scale_error_pct,
                    "official_error_pct": length.official_error_pct,
                    "scale_ok": length.scale_ok,
                    "official_ok": length.official_ok,
                },
                "frame": {
                    "rotation_deg": transform.rotation_deg,
                    "translation_m": transform.translation.tolist(),
                    "median_residual_m": transform.median_residual_m,
                    "iterations": transform.iterations,
                },
                "sector_consistency": sector_aligned.to_dict("records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = AlignmentResult(
        root=out_root,
        laps_total=len(laps),
        laps_aligned=len(aligned_frames),
        laps_rejected=len(rejected),
        median_residual_m=median_residual,
        median_lap_length_m=median_length,
        length=length,
        frame=transform,
        method=method,
        reference_line_lap=reference_line_lap,
        sector_aligned=sector_aligned,
        sector_raw=sector_raw,
    )

    logger.info(
        "alignment_complete method=%s aligned=%d/%d rejected=%d median_residual_m=%.2f "
        "median_lap_length_m=%.1f scale_error_pct=%.3f official_error_pct=%+.3f",
        method,
        result.laps_aligned,
        result.laps_total,
        result.laps_rejected,
        result.median_residual_m,
        result.median_lap_length_m,
        length.scale_error_pct,
        length.official_error_pct,
    )
    return result
