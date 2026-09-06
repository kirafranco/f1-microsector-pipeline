"""Session-level ground-truth validation: the end-to-end test of slice 1."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.align.circuits import OFFICIAL_LENGTH_BAND_PCT
from src.config import PROCESSED_ROOT
from src.grid.resample import GRID_SPACING_M
from src.metrics.delta import time_curves
from src.metrics.reference import ReferenceSpec, lap_label, resolve_reference
from src.segment.validation import Spread
from src.validate.anomalies import lap_outliers, sector_split_outliers
from src.validate.closure import closure_residuals, reconstruct_laps
from src.validate.stability import DEFAULT_MIN_LAPS, DEFAULT_PUSH_FRACTION, push_laps, v_min_stability
from src.validate.timing_line import line_crossings, session_line_positions, start_stretch

logger = logging.getLogger(__name__)

#: Thresholds from the spec, set one notch above the measured post-fix figures.
LAP_STD_MAX_S = 0.12
LAP_P95_MAX_S = 0.20
#: Closure is the lap residual minus the reference lap's own: per session the
#: two spreads agree to three decimals. So its p95 is the lap check again --
#: F004's original 0.35 s, which F010 tightened to 0.20 on one session -- and
#: the only thing closure adds is whether the reference lap is itself odd,
#: which `reference_offset_s` reports and `LAP_P95_MAX_S` bounds (F022). The
#: p50 rule this replaces measured which lap F004 happened to pick: it
#: correlated 0.972 with that lap's own offset and failed 9 sessions.
CLOSURE_P95_MAX_S = 0.35
SECTOR_MEDIAN_MAX_S = 0.03

#: A sector time is bounded by two timing-loop crossings exactly as a lap is,
#: and the measured spreads agree: sector-to-lap std ratios of 1.01, 0.93 and
#: 0.96 across the season. So the ceiling *is* the lap ceiling, derived here
#: rather than typed, and the two cannot drift apart again. The 0.10/0.08/0.08
#: this replaces was one notch above Suzuka's own and failed the four circuits
#: whose timing loops sit at 150-210 km/h, where 4 m of registration is 0.1 s.
SECTOR_STD_MAX_S = {name: LAP_STD_MAX_S for name in ("s1", "s2", "s3")}

#: Anomalous laps are excluded like coverage-poor ones; what stays gated is how
#: many. This is a broken-pipeline bound -- alignment wrong for every lap, not
#: one in a hundred -- not a precision claim. The season's worst is 8.4 %
#: (Monaco Q) against a 0.88 % median.
ANOMALY_FRACTION_MAX = 0.10

LINE_POSITION_STD_MAX_M = 8.0

#: Lap-to-lap spread in driven distance, as a percentage of the official lap
#: length (F019). The absolute 10 m this replaces was Suzuka qualifying's own
#: figure and did not travel: the spread is a property of the racing line and
#: scales with the circuit, measuring 0.170 % of lap length at the median in
#: both qualifying and races across the 2024 season. At 0.25 % the only
#: sessions that fail are Montreal R, Sao Paulo Q and Spa Q -- each with rain
#: in its own weather snapshot.
DISTANCE_STD_MAX_PCT = 0.25

#: V_min repeatability, over push laps only (see `validate.stability`). The
#: 3.0 / 6.5 these replace were read off one dry qualifying session; the
#: season's push-lap distribution is a median of 2.46 km/h and a p95 of 5.69 in
#: qualifying, 2.94 and 6.42 in races. At 4.5 / 10.0 the only sessions that
#: fail are Silverstone Q, Silverstone R and Spa Q -- again, all three wet.
V_MIN_STD_MEDIAN_MAX_KMH = 4.5
V_MIN_STD_P95_MAX_KMH = 10.0

#: A lap whose telemetry window opens later than the session p95 by more than
#: this is missing real data, not misaligned; it is reported, not gated on.
COVERAGE_SLACK_S = 0.25

GROUND_TRUTH_COLUMNS = (
    "driver", "lap_number", "compound", "lap_time_s", "lap_grid_s", "lap_residual_s",
    "s1_residual_s", "s2_residual_s", "s3_residual_s", "closure_residual_s",
    "driven_m", "driven_pct_of_official", "line_start_m", "line_end_m",
    "window_open_s", "window_close_s", "start_extrap_m", "end_extrap_m",
    "start_coverage_poor", "end_coverage_poor", "distance_excursion", "is_reference",
    "start_offset_s", "start_stretch_s", "start_from_samples",
    "sector_split_outlier", "lap_outlier",
)



def sector_registration_m(
    grid: pd.DataFrame, sector_std_s: dict[str, float], s1_m: float, s2_m: float, grid_m: float
) -> dict[str, float]:
    """Each sector's spread expressed as metres of per-crossing registration.

    Reported, never gated (F022). A sector is bounded by two timing loops, so a
    spatial noise of ``x`` metres at each end gives a time spread of
    ``x * sqrt(1/va^2 + 1/vb^2)``; inverting puts every circuit on one scale.
    In seconds the season's sector spreads vary by 30 %, in metres by 25 % around
    a common 3.6-4.0 m -- and the sessions that look worst in seconds are those
    whose loops sit at 150-210 km/h, where 4 m is 0.1 s. Grid 0 stands in for the
    line: since F015 it is 35-188 m further along the same straight.
    """
    speeds = grid.groupby("grid_index")["speed"].median()

    def at(distance_m: float) -> float:
        index = int(round(max(distance_m, 0.0) / grid_m))
        window = speeds.loc[speeds.index.intersection(range(index - 2, index + 3))]
        return float(window.median()) / 3.6 if len(window) else float("nan")

    v_line, v_s1, v_s2 = at(0.0), at(s1_m), at(s2_m)
    bounds = {"s1": (v_line, v_s1), "s2": (v_s1, v_s2), "s3": (v_s2, v_line)}
    out: dict[str, float] = {}
    for name, (first, second) in bounds.items():
        if not (np.isfinite(first) and np.isfinite(second)) or first <= 0 or second <= 0:
            out[name] = float("nan")
            continue
        out[name] = float(sector_std_s[name] / np.sqrt(1.0 / first**2 + 1.0 / second**2))
    return out


def distance_excursions(
    driven_pct_of_official: pd.Series, band: tuple[float, float] = OFFICIAL_LENGTH_BAND_PCT
) -> pd.Series:
    """Laps whose driven distance falls outside the band: off the road and back on.

    Twenty laps of the 2024 season's 26,671 qualify (0.075 %), on accurate,
    green-flag laps with no pit time -- Piastri at -5.05 % and Albon at +4.64 %
    are the extremes. They are excluded from the gated statistics and named in
    the report rather than failing the session, the way a coverage-poor lap
    already is (global section 3.1). A lap with no distance is not an excursion.
    """
    low, high = band
    value = driven_pct_of_official.astype(float)
    return ((value < low) | (value > high)).fillna(False)

@dataclass(frozen=True)
class ValidationReport:
    laps: int
    laps_gated: int
    flagged: list[str]
    reference_label: str
    line_start_m: float
    line_end_m: float
    line_start_std_m: float
    line_end_std_m: float
    window_open_median_s: float
    window_close_median_s: float
    lap_residual_median_s: float
    lap_residual_std_s: float
    lap_residual: Spread
    closure: Spread
    sector_median_s: dict[str, float]
    sector_std_s: dict[str, float]
    driven_median_m: float
    driven_std_m: float
    driven_std_pct: float
    driven_pct_min: float
    driven_pct_max: float
    excursions: list[str]
    anomalies: list[str]
    anomaly_fraction: float
    reference_offset_s: float
    reference_flagged: bool
    registration_m: dict[str, float]
    push_laps: int
    laps_from_samples: int
    v_min_groups: int
    v_min_std_median_kmh: float
    v_min_std_p95_kmh: float
    compounds: dict[str, int]
    start_extrap_max_m: float
    end_extrap_max_m: float

    @property
    def lap_ok(self) -> bool:
        return self.lap_residual_std_s <= LAP_STD_MAX_S and self.lap_residual.p95 <= LAP_P95_MAX_S

    @property
    def closure_ok(self) -> bool:
        """The spread, plus whether the reference lap is a reasonable one.

        A closure residual is a lap's residual minus the reference lap's, so the
        spread is the lap spread and the median is that one lap's registration
        offset. Gating the median therefore gated a coin flip; gating the offset
        asks the question that was meant. When the reference lap is itself
        excluded the offset cannot be measured and is not held against the
        session -- two sessions of the 2024 season, named in the report.
        """
        reference_ok = self.reference_flagged or abs(self.reference_offset_s) <= LAP_P95_MAX_S
        return self.closure.p95 <= CLOSURE_P95_MAX_S and reference_ok

    @property
    def anomaly_fraction_ok(self) -> bool:
        return self.anomaly_fraction <= ANOMALY_FRACTION_MAX

    @property
    def sectors_ok(self) -> bool:
        return all(
            abs(self.sector_median_s[name]) <= SECTOR_MEDIAN_MAX_S
            and self.sector_std_s[name] <= SECTOR_STD_MAX_S[name]
            for name in ("s1", "s2", "s3")
        )

    @property
    def distance_ok(self) -> bool:
        """Spread of the laps that stayed on the road, plus the band on those laps.

        A lap outside the band drove somewhere else -- off the road and back on
        -- and is excluded as an excursion before the statistics, the way a
        coverage-poor lap already is. Global section 3.1: an anomalous record is
        logged and skipped, the batch does not stop. The excursions are named in
        `excursions` so they stay visible.
        """
        low, high = OFFICIAL_LENGTH_BAND_PCT
        return (
            self.driven_std_pct <= DISTANCE_STD_MAX_PCT
            and self.driven_pct_min >= low
            and self.driven_pct_max <= high
        )

    @property
    def stability_ok(self) -> bool:
        if self.v_min_groups == 0:
            return True
        return (
            self.v_min_std_median_kmh <= V_MIN_STD_MEDIAN_MAX_KMH
            and self.v_min_std_p95_kmh <= V_MIN_STD_P95_MAX_KMH
        )

    @property
    def line_ok(self) -> bool:
        return self.line_start_std_m <= LINE_POSITION_STD_MAX_M and self.line_end_std_m <= LINE_POSITION_STD_MAX_M

    @property
    def ok(self) -> bool:
        return (self.lap_ok and self.closure_ok and self.sectors_ok and self.distance_ok
                and self.stability_ok and self.line_ok and self.anomaly_fraction_ok)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["checks"] = {
            "lap_reconstruction": self.lap_ok,
            "delta_closure": self.closure_ok,
            "sector_times": self.sectors_ok,
            "driven_distance": self.distance_ok,
            "v_min_stability": self.stability_ok,
            "timing_line_spread": self.line_ok,
            "anomaly_fraction": self.anomaly_fraction_ok,
            "all": self.ok,
        }
        return out


@dataclass(frozen=True)
class ValidationResult:
    root: Path
    ground_truth: pd.DataFrame
    stability: pd.DataFrame
    report: ValidationReport
    elapsed_s: float


def validate_session(
    snapshot_root: Path,
    aligned_root: Path,
    grid_root: Path,
    processed_root: Path,
    out_root: Path | None = None,
    reference: ReferenceSpec = ReferenceSpec(),
    grid_m: float = GRID_SPACING_M,
    min_laps: int = DEFAULT_MIN_LAPS,
    push_fraction: float = DEFAULT_PUSH_FRACTION,
) -> ValidationResult:
    """Reconstruct official timing from the pipeline output and score every criterion."""
    started = time.perf_counter()
    aligned = pd.read_parquet(aligned_root / "telemetry_aligned.parquet")
    grid = pd.read_parquet(grid_root / "grid.parquet")
    laps = pd.read_parquet(snapshot_root / "laps.parquet")
    corner_metrics = pd.read_parquet(processed_root / "corner_metrics.parquet")
    meta = json.loads((aligned_root / "alignment_meta.json").read_text(encoding="utf-8"))

    line_length_m = float(meta["reference_line_length_m"])
    official_length_m = float(meta["official_lap_length_m"])
    boundaries = {row["boundary"]: float(row["median_m"]) for row in meta.get("sector_consistency", [])}
    s1_m, s2_m = boundaries.get("S1", np.nan), boundaries.get("S2", np.nan)

    crossings = line_crossings(aligned, laps, line_length_m)
    d_start, d_end = session_line_positions(crossings)
    logger.info(
        "timing_line start_m=%.2f (std %.2f) end_m=%.2f (std %.2f) window_open_s=%+.3f window_close_s=%+.3f",
        d_start, float(crossings["line_start_m"].std()), d_end, float(crossings["line_end_m"].std()),
        float(crossings["window_open_s"].median()), float(crossings["window_close_s"].median()),
    )

    curves = time_curves(grid)
    speeds = grid.pivot_table(index=["driver", "lap_number"], columns="grid_index", values="speed", aggfunc="first")
    speeds.index = curves.index
    ref = resolve_reference(laps, curves.index, reference)
    stretch = start_stretch(crossings, d_start)
    reconstructed = reconstruct_laps(curves, speeds, laps, d_start, d_end, s1_m, s2_m, grid_m, start_stretch=stretch)
    reconstructed = closure_residuals(reconstructed, ref)

    table = reconstructed.merge(crossings.drop(columns=["start_extrapolated", "end_extrapolated"]),
                                on=["driver", "lap_number"], how="left")
    tyres = laps[["driver", "lap_number", "compound"]].copy()
    tyres["driver"] = tyres["driver"].astype(str)
    tyres["lap_number"] = tyres["lap_number"].astype(int)
    table = table.merge(tyres, on=["driver", "lap_number"], how="left")
    table["driven_pct_of_official"] = 100.0 * (table["driven_m"] - official_length_m) / official_length_m

    open_limit = float(table["window_open_s"].quantile(0.95)) + COVERAGE_SLACK_S
    close_limit = float(table["window_close_s"].quantile(0.05)) - COVERAGE_SLACK_S
    table["start_coverage_poor"] = table["window_open_s"] > open_limit
    table["end_coverage_poor"] = table["window_close_s"] < close_limit
    table["distance_excursion"] = distance_excursions(table["driven_pct_of_official"])
    # Per-lap anomalies in the reconstructed timing (F022): the source split a
    # lap's sectors elsewhere, or the lap is simply far out. Both are excluded
    # from the gated statistics and reported; their fraction is what is gated.
    table["sector_split_outlier"] = sector_split_outliers(table)
    table["lap_outlier"] = lap_outliers(table)
    anomaly_mask = table["sector_split_outlier"] | table["lap_outlier"]
    coverage_mask = table["start_coverage_poor"].fillna(False) | table["end_coverage_poor"].fillna(False)
    flagged_mask = coverage_mask | table["distance_excursion"] | anomaly_mask
    flagged = [f"{r.driver} L{int(r.lap_number)}" for r in table[coverage_mask].itertuples()]
    excursions = [
        f"{r.driver} L{int(r.lap_number)} {r.driven_pct_of_official:+.2f}%"
        for r in table[table["distance_excursion"]].itertuples()
    ]
    if flagged:
        logger.warning("coverage_flagged laps=%s (excluded from gated statistics)", flagged)
    anomalies = [
        f"{r.driver} L{int(r.lap_number)} "
        f"{'split' if r.sector_split_outlier else ''}{'+' if r.sector_split_outlier and r.lap_outlier else ''}"
        f"{'lap' if r.lap_outlier else ''}"
        for r in table[anomaly_mask].itertuples()
    ]
    if excursions:
        logger.warning("distance_excursion laps=%s (off the road and back; excluded from gated statistics)", excursions)
    if anomalies:
        logger.warning("timing_anomaly laps=%s (excluded from gated statistics)", anomalies)

    gated = table[~flagged_mask]
    anomaly_fraction = float(anomaly_mask.sum()) / len(table) if len(table) else 0.0

    # The reference lap's own registration offset is the constant every closure
    # residual carries. When that lap is itself excluded the offset cannot be
    # measured, and the session is not judged on it.
    reference_row = gated[gated["is_reference"].fillna(False).astype(bool)]
    reference_flagged = bool(reference_row.empty)
    gated_lap_median = float(gated["lap_residual_s"].median())
    reference_offset_s = (
        float("nan") if reference_flagged
        else float(reference_row["lap_residual_s"].iloc[0]) - gated_lap_median
    )
    reference_label = lap_label(ref.iloc[0]) if reference.kind != "driver_best" else reference.label
    if reference_flagged:
        logger.warning("reference_lap_flagged label=%s (closure offset unavailable, not gated)", reference_label)

    sector_std = {name: float(gated[f"{name}_residual_s"].std()) for name in ("s1", "s2", "s3")}
    registration = sector_registration_m(grid, sector_std, s1_m, s2_m, grid_m)
    stability = v_min_stability(corner_metrics, laps, min_laps, push_fraction)
    lap_residual = gated["lap_residual_s"].dropna()
    closure = gated["closure_residual_s"].dropna()
    std_values = stability["v_min_std_kmh"].dropna()

    report = ValidationReport(
        laps=int(len(table)),
        laps_gated=int(len(gated)),
        flagged=flagged,
        reference_label=reference_label,
        line_start_m=d_start,
        line_end_m=d_end,
        line_start_std_m=float(crossings["line_start_m"].std()),
        line_end_std_m=float(crossings["line_end_m"].std()),
        window_open_median_s=float(crossings["window_open_s"].median()),
        window_close_median_s=float(crossings["window_close_s"].median()),
        lap_residual_median_s=float(lap_residual.median()) if len(lap_residual) else float("nan"),
        lap_residual_std_s=float(lap_residual.std()) if len(lap_residual) > 1 else 0.0,
        lap_residual=Spread.of(lap_residual.to_numpy(dtype=float)),
        closure=Spread.of(closure.to_numpy(dtype=float)),
        sector_median_s={n: float(gated[f"{n}_residual_s"].median()) for n in ("s1", "s2", "s3")},
        sector_std_s=sector_std,
        driven_median_m=float(gated["driven_m"].median()),
        driven_std_m=float(gated["driven_m"].std()),
        driven_std_pct=100.0 * float(gated["driven_m"].std()) / official_length_m,
        driven_pct_min=float(gated["driven_pct_of_official"].min()),
        driven_pct_max=float(gated["driven_pct_of_official"].max()),
        excursions=excursions,
        anomalies=anomalies,
        anomaly_fraction=anomaly_fraction,
        reference_offset_s=reference_offset_s,
        reference_flagged=reference_flagged,
        registration_m=registration,
        push_laps=int(len(push_laps(laps, push_fraction))),
        laps_from_samples=int(table["start_from_samples"].fillna(False).sum()),
        v_min_groups=int(len(stability)),
        v_min_std_median_kmh=float(std_values.median()) if len(std_values) else float("nan"),
        v_min_std_p95_kmh=float(std_values.quantile(0.95)) if len(std_values) else float("nan"),
        compounds={str(k): int(v) for k, v in stability["compound"].value_counts().items()},
        start_extrap_max_m=float(table["start_extrap_m"].max()),
        end_extrap_max_m=float(table["end_extrap_m"].max()),
    )

    out_root = out_root or (PROCESSED_ROOT / grid_root.name)
    out_root.mkdir(parents=True, exist_ok=True)
    ground_truth = table[list(GROUND_TRUTH_COLUMNS)]
    ground_truth.to_parquet(out_root / "ground_truth.parquet", index=False)
    stability.to_parquet(out_root / "v_min_stability.parquet", index=False)

    elapsed = time.perf_counter() - started
    (out_root / "ground_truth_report.json").write_text(
        json.dumps(
            {
                "snapshot": str(snapshot_root),
                "aligned": str(aligned_root),
                "grid": str(grid_root),
                "processed": str(processed_root),
                "reference": {**asdict(reference), "label": report.reference_label},
                "line_length_m": line_length_m,
                "official_lap_length_m": official_length_m,
                "sector_boundaries_m": {"S1": s1_m, "S2": s2_m},
                "elapsed_s": elapsed,
                "acceptance": report.to_dict(),
                "limitation": (
                    "Residual lap-closure noise is the source's floor: the interior timing spread "
                    "plus ~4 m of timing-versus-telemetry registration at each line crossing. Any "
                    "comparison between a grid time and an official time inherits ~0.1 s, and any "
                    "delta-t between two laps ~0.15 s at p95. Flagged laps are missing telemetry, "
                    "not misaligned, and are reported rather than gated."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "validation_complete laps=%d gated=%d flagged=%d excursions=%d push_laps=%d lap_median_s=%+.3f lap_std_s=%.3f "
        "closure_p50_s=%.3f closure_p95_s=%.3f s1=%+.3f s2=%+.3f s3=%+.3f v_min_std_median=%.2f "
        "acceptance_ok=%s elapsed_s=%.2f",
        report.laps, report.laps_gated, len(flagged), len(report.excursions), report.push_laps,
        report.lap_residual_median_s, report.lap_residual_std_s,
        report.closure.p50, report.closure.p95, report.sector_median_s["s1"], report.sector_median_s["s2"],
        report.sector_median_s["s3"], report.v_min_std_median_kmh, report.ok, elapsed,
    )
    return ValidationResult(root=out_root, ground_truth=ground_truth, stability=stability, report=report, elapsed_s=elapsed)
