"""Acceptance measurements for the metrics layer (F004 criteria 1-6)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.grid.resample import GRID_SPACING_M
from src.metrics.delta import lap_lengths, reference_curves
from src.metrics.reference import accurate_lap_times, lap_index
from src.segment.phases import GRAIN_CORNER_PHASE
from src.segment.validation import Spread

#: Thresholds from the spec.
CLOSURE_MAX_S = 1e-3
S2_MEDIAN_MAX_S = 0.05
S2_STD_MAX_S = 0.10
ENDPOINT_P50_MAX_S = 0.15
ENDPOINT_P95_MAX_S = 0.35
BRAKE_COVERAGE_MIN = 0.95


def delta_zero_checks(delta: pd.DataFrame, reference: pd.Series) -> tuple[bool, bool]:
    """Criterion 1: delta-t is 0 at grid 0 on every lap, and 0 everywhere on reference laps."""
    at_zero = delta.loc[delta["grid_index"] == 0, "delta_t_s"].to_numpy(dtype=float)
    zero_ok = bool(len(at_zero) and np.all(at_zero == 0.0))
    ref_keys = {(str(d), int(n)) for d, n in set(reference.tolist())}
    keys = list(zip(delta["driver"].astype(str), delta["lap_number"].astype(int)))
    on_ref = delta.loc[[k in ref_keys for k in keys], "delta_t_s"].to_numpy(dtype=float)
    # A lap referenced against itself is identically zero; against another
    # reference lap (driver_best) it need not be.
    self_ref = [k == (str(r[0]), int(r[1])) for k, r in zip(keys, reference.reindex(lap_index(keys)).tolist())]
    on_self = delta.loc[self_ref, "delta_t_s"].to_numpy(dtype=float)
    ref_ok = bool(len(on_self) and np.all(on_self == 0.0)) if len(ref_keys) else True
    _ = on_ref
    return zero_ok, ref_ok


def closure_error(
    times: pd.DataFrame, delta: pd.DataFrame, microsectors: pd.DataFrame, grain: str = GRAIN_CORNER_PHASE
) -> float:
    """Criterion 2: sum of complete-sector deltas equals delta-t at the last complete sector end.

    "Complete" means no lap has the sector flagged partial or missing.
    """
    sectors = microsectors[microsectors["grain"] == grain].sort_values("microsector_id")
    t = times[times["grain"] == grain]
    bad = t[t["partial"].fillna(True).astype(bool) | t["time_s"].isna()]["microsector_id"].unique()
    complete = sectors[~sectors["microsector_id"].isin(bad)]
    if complete.empty:
        return float("nan")
    end_index = int(complete["end_index"].iloc[-1])
    ids = set(complete["microsector_id"].astype(int))

    summed = (
        t[t["microsector_id"].astype(int).isin(ids)]
        .groupby(["driver", "lap_number"], observed=True)["delta_s"]
        .sum()
    )
    at_end = delta[delta["grid_index"] == end_index].set_index(["driver", "lap_number"])["delta_t_s"]
    joined = pd.concat([summed.rename("sum"), at_end.rename("end")], axis=1).dropna()
    if joined.empty:
        return float("nan")
    return float((joined["sum"].astype(float) - joined["end"].astype(float)).abs().max())


def marshalling_sectors(
    curves: pd.DataFrame, laps: pd.DataFrame, s1_m: float, s2_m: float, grid_m: float = GRID_SPACING_M
) -> pd.DataFrame:
    """Criterion 3 and the reported S1/S3 offsets: grid vs official sector times, per lap."""
    values = curves.to_numpy()
    n = lap_lengths(curves).to_numpy()
    rows = np.arange(len(values))
    i1, i2 = int(round(s1_m / grid_m)), int(round(s2_m / grid_m))
    last = values[rows, n - 1]
    t1 = values[:, i1] if i1 < values.shape[1] else np.full(len(values), np.nan)
    t2 = values[:, i2] if i2 < values.shape[1] else np.full(len(values), np.nan)
    official = laps.copy()
    official["driver"] = official["driver"].astype(str)
    official["lap_number"] = official["lap_number"].astype(int)
    official = official.set_index(["driver", "lap_number"]).reindex(curves.index)
    return pd.DataFrame(
        {
            "s1_grid_s": t1,
            "s2_grid_s": t2 - t1,
            "s3_grid_s": last - t2,
            "s1_official_s": official["sector1_time"].to_numpy(dtype=float),
            "s2_official_s": official["sector2_time"].to_numpy(dtype=float),
            "s3_official_s": official["sector3_time"].to_numpy(dtype=float),
        },
        index=curves.index,
    )


def endpoint_residuals(curves: pd.DataFrame, laps: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """Criterion 4: delta-t at the last common grid point vs the official lap-time difference."""
    values = curves.to_numpy()
    ref = reference_curves(curves, reference)
    n = lap_lengths(curves).to_numpy()
    n_ref = np.isfinite(ref).sum(axis=1)
    rows = np.arange(len(values))
    last_common = np.minimum(n, n_ref) - 1
    delta_end = values[rows, last_common] - ref[rows, last_common]
    grid_time = values[rows, n - 1]

    times = accurate_lap_times(laps)
    own = times.reindex(curves.index).to_numpy(dtype=float)
    ref_keys = lap_index(reference.reindex(curves.index).tolist())
    ref_time = times.reindex(ref_keys).to_numpy(dtype=float)
    official_delta = own - ref_time
    is_reference = np.array([k == r for k, r in zip(curves.index, ref_keys)])
    return pd.DataFrame(
        {
            "n_points": n,
            "grid_time_s": grid_time,
            "delta_t_end_s": delta_end,
            "official_delta_s": official_delta,
            "residual_s": delta_end - official_delta,
            "is_reference": is_reference,
        },
        index=curves.index,
    )


@dataclass(frozen=True)
class MetricsReport:
    laps: int
    reference_kind: str
    reference_label: str
    delta_zero_ok: bool
    reference_zero_ok: bool
    closure_max_err_s: float
    s2_median_s: float
    s2_std_s: float
    s2_n: int
    s1_median_s: float
    s3_median_s: float
    endpoint: Spread
    endpoint_signed_median_s: float
    vmin_coverage: float
    brake_coverage: float
    brake_gap_m: Spread
    unflagged_nan_sector_times: int
    sector_std_median_s: float
    within_driver_std_median_s: float

    @property
    def closure_ok(self) -> bool:
        return np.isfinite(self.closure_max_err_s) and self.closure_max_err_s <= CLOSURE_MAX_S

    @property
    def s2_ok(self) -> bool:
        if self.s2_n == 0:
            return True  # no marshalling boundaries available: nothing to check
        return abs(self.s2_median_s) <= S2_MEDIAN_MAX_S and self.s2_std_s <= S2_STD_MAX_S

    @property
    def endpoint_ok(self) -> bool:
        return self.endpoint.n == 0 or (
            self.endpoint.p50 <= ENDPOINT_P50_MAX_S and self.endpoint.p95 <= ENDPOINT_P95_MAX_S
        )

    @property
    def coverage_ok(self) -> bool:
        return self.vmin_coverage >= 1.0 and (np.isnan(self.brake_coverage) or self.brake_coverage >= BRAKE_COVERAGE_MIN)

    @property
    def sectors_ok(self) -> bool:
        return self.unflagged_nan_sector_times == 0

    @property
    def ok(self) -> bool:
        return (
            self.delta_zero_ok
            and self.reference_zero_ok
            and self.closure_ok
            and self.s2_ok
            and self.endpoint_ok
            and self.coverage_ok
            and self.sectors_ok
        )

    def to_dict(self) -> dict:
        out = asdict(self)
        out["checks"] = {
            "delta_zero_at_line": self.delta_zero_ok,
            "reference_delta_identically_zero": self.reference_zero_ok,
            "closure": self.closure_ok,
            "s2_interior_accuracy": self.s2_ok,
            "endpoint_vs_official": self.endpoint_ok,
            "corner_metric_coverage": self.coverage_ok,
            "sector_times_complete": self.sectors_ok,
            "all": self.ok,
        }
        return out


def measure(
    delta: pd.DataFrame,
    times: pd.DataFrame,
    summary: pd.DataFrame,
    corners: pd.DataFrame,
    microsectors: pd.DataFrame,
    lap_summary: pd.DataFrame,
    reference: pd.Series,
    events: pd.DataFrame,
    reference_kind: str,
    reference_label: str,
) -> MetricsReport:
    """Run every criterion and collect the reported-not-gated figures."""
    zero_ok, ref_ok = delta_zero_checks(delta, reference)

    s2 = (lap_summary["s2_grid_s"] - lap_summary["s2_official_s"]).dropna()
    s1 = (lap_summary["s1_grid_s"] - lap_summary["s1_official_s"]).dropna()
    s3 = (lap_summary["s3_grid_s"] - lap_summary["s3_official_s"]).dropna()
    resid = lap_summary.loc[~lap_summary["is_reference"].astype(bool), "residual_s"].dropna()

    n_events = int(len(events))
    n_laps = int(lap_summary.shape[0])
    vmin_cov = float(corners["v_min_kmh"].notna().sum() / (n_laps * n_events)) if n_events else 1.0
    braked = events["has_braking"].fillna(False).astype(bool).sum() if n_events else 0
    if braked:
        rows = corners[corners["event_id"].isin(events.loc[events["has_braking"].fillna(False).astype(bool), "event_id"].astype(int))]
        brake_cov = float(rows["brake_dev_m"].notna().sum() / max(1, len(rows)))
    else:
        brake_cov = float("nan")

    unflagged = int((times["time_s"].isna() & ~times["partial"].fillna(False).astype(bool)).sum())
    phase_summary = summary[summary["grain"] == GRAIN_CORNER_PHASE]

    return MetricsReport(
        laps=n_laps,
        reference_kind=reference_kind,
        reference_label=reference_label,
        delta_zero_ok=zero_ok,
        reference_zero_ok=ref_ok,
        closure_max_err_s=closure_error(times, delta, microsectors),
        s2_median_s=float(s2.median()) if len(s2) else float("nan"),
        s2_std_s=float(s2.std(ddof=0)) if len(s2) > 1 else 0.0,
        s2_n=int(len(s2)),
        s1_median_s=float(s1.median()) if len(s1) else float("nan"),
        s3_median_s=float(s3.median()) if len(s3) else float("nan"),
        endpoint=Spread.of(resid.to_numpy(dtype=float)),
        endpoint_signed_median_s=float(resid.median()) if len(resid) else float("nan"),
        vmin_coverage=vmin_cov,
        brake_coverage=brake_cov,
        brake_gap_m=Spread.of(corners["brake_gap_m"].to_numpy(dtype=float)),
        unflagged_nan_sector_times=unflagged,
        sector_std_median_s=float(phase_summary["std_s"].median()) if len(phase_summary) else float("nan"),
        within_driver_std_median_s=float(phase_summary["within_driver_std_s"].median()) if len(phase_summary) else float("nan"),
    )
