"""Acceptance measurements for the segmentation (F009 criteria 1-8).

Every function compares the session-level tables against the per-lap grid they
were derived from, so the same code serves the tests and the report written
next to the parquet output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.grid.resample import GRID_SPACING_M
from src.segment.corners import parse_corner_label
from src.segment.events import EventParams, detect_events, median_traces
from src.segment.phases import GRAIN_CORNER_PHASE, GRAIN_FIXED_100M, check_partition

#: Thresholds from the spec. The per-lap ones sit at 30 m rather than the
#: measured 20 m because D6 bounds the source resolution at roughly +/-20 m.
APEX_DEV_P95_M = 30.0
BRAKE_DEV_P95_M = 30.0
JACKKNIFE_SHIFT_P95_M = 10.0
CORNER_OFFSET_MAX_M = 10.0

APEX_WINDOW_MARGIN_M = 30.0
BRAKE_LOOKBACK_M = 50.0


def _in_window(distance: np.ndarray, start: float, end: float, lap_length: float) -> np.ndarray:
    """``start <= d < end`` on the circular lap; ``start`` may be negative."""
    start %= lap_length
    end %= lap_length
    if end > start:
        return (distance >= start) & (distance < end)
    return (distance >= start) | (distance < end)


def per_lap_apex(grid: pd.DataFrame, events: pd.DataFrame, lap_length_m: float) -> pd.DataFrame:
    """Criterion 3: each lap's speed minimum inside every event window.

    Window is ``[apex_start - 30, apex_end + 30)``. Returns one row per
    lap-event with the deviation from the session apex and the lap's V_min.
    """
    rows = []
    for (driver, lap_number), lap in grid.groupby(["driver", "lap_number"], observed=True):
        d = lap["distance_m"].to_numpy(dtype=float)
        v = lap["speed"].to_numpy(dtype=float)
        for _, e in events.iterrows():
            mask = _in_window(d, float(e["apex_start_m"]) - APEX_WINDOW_MARGIN_M,
                              float(e["apex_end_m"]) + APEX_WINDOW_MARGIN_M, lap_length_m)
            if not mask.any():
                continue
            idx = np.flatnonzero(mask)
            i = idx[np.nanargmin(v[idx])]
            dev = d[i] - float(e["apex_m"])
            dev -= lap_length_m * np.rint(dev / lap_length_m)
            rows.append(dict(driver=driver, lap_number=lap_number, event_id=int(e["event_id"]),
                             apex_dev_m=dev, v_min_kmh=v[i]))
    return pd.DataFrame(rows, columns=["driver", "lap_number", "event_id", "apex_dev_m", "v_min_kmh"])


def per_lap_brake(grid: pd.DataFrame, events: pd.DataFrame, lap_length_m: float) -> pd.DataFrame:
    """Criterion 4: each lap's first brake application in every braked event.

    Window is ``[brake_on - 50, apex_start)``. A lap with no brake sample in the
    window gets NaN -- the driver braked outside the shared zone -- and is
    reported rather than dropped.
    """
    braked = events[events["has_braking"].fillna(False).astype(bool)]
    rows = []
    for (driver, lap_number), lap in grid.groupby(["driver", "lap_number"], observed=True):
        d = lap["distance_m"].to_numpy(dtype=float)
        brake = lap["brake"].fillna(False).to_numpy(dtype=bool)
        for _, e in braked.iterrows():
            mask = _in_window(d, float(e["brake_on_m"]) - BRAKE_LOOKBACK_M, float(e["apex_start_m"]), lap_length_m)
            idx = np.flatnonzero(mask & brake)
            if len(idx):
                dev = d[idx[0]] - float(e["brake_on_m"])
                dev -= lap_length_m * np.rint(dev / lap_length_m)
            else:
                dev = np.nan
            rows.append(dict(driver=driver, lap_number=lap_number, event_id=int(e["event_id"]), brake_dev_m=dev))
    return pd.DataFrame(rows, columns=["driver", "lap_number", "event_id", "brake_dev_m"])


def jackknife(
    grid: pd.DataFrame,
    corners: pd.DataFrame | None,
    events: pd.DataFrame,
    params: EventParams = EventParams(),
    trials: int = 10,
    seed: int = 0,
    grid_m: float = GRID_SPACING_M,
) -> pd.DataFrame:
    """Criteria 5-6: re-detect events on random half-sessions.

    One row per trial and event with the shift of apex, brake-on and exit-end
    against the full-session table; a trial whose event count differs is
    recorded once with ``count_match=False`` and no shifts.
    """
    keys = grid[["driver", "lap_number"]].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(trials):
        pick = rng.choice(len(keys), size=max(1, len(keys) // 2), replace=False)
        chosen = pd.MultiIndex.from_arrays([keys[pick, 0], keys[pick, 1]])
        half = grid.set_index(["driver", "lap_number"]).loc[chosen].reset_index()
        found = detect_events(median_traces(half), corners, params, grid_m)
        if len(found) != len(events):
            rows.append(dict(trial=trial, event_id=pd.NA, count_match=False,
                             apex_shift_m=np.nan, brake_on_shift_m=np.nan, exit_end_shift_m=np.nan))
            continue
        for k in range(len(events)):
            rows.append(dict(
                trial=trial, event_id=k, count_match=True,
                apex_shift_m=float(found["apex_m"].iloc[k] - events["apex_m"].iloc[k]),
                brake_on_shift_m=float(found["brake_on_m"].iloc[k] - events["brake_on_m"].iloc[k]),
                exit_end_shift_m=float(found["exit_end_m"].iloc[k] - events["exit_end_m"].iloc[k]),
            ))
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Spread:
    p50: float
    p95: float
    max: float
    n: int
    n_missing: int = 0

    @staticmethod
    def of(values: np.ndarray) -> "Spread":
        values = np.asarray(values, dtype=float)
        finite = np.abs(values[np.isfinite(values)])
        missing = int(len(values) - len(finite))
        if len(finite) == 0:
            return Spread(float("nan"), float("nan"), float("nan"), 0, missing)
        return Spread(float(np.percentile(finite, 50)), float(np.percentile(finite, 95)),
                      float(finite.max()), int(len(finite)), missing)


@dataclass(frozen=True)
class SegmentationReport:
    lap_length_m: float
    events: int
    partition_ok: bool
    fixed_bins_ok: bool
    fixed_bins: int
    apex_dev: Spread
    brake_dev: Spread
    jackknife_trials: int
    jackknife_count_matches: int
    apex_shift: Spread
    brake_on_shift: Spread
    exit_end_shift: Spread
    corner_offset_max_m: float
    events_without_corners: int
    corners_without_event: int
    v_min_std_all_drivers_kmh: dict[int, float]
    v_min_std_within_driver_kmh: dict[int, float]

    @property
    def apex_ok(self) -> bool:
        return self.apex_dev.p95 <= APEX_DEV_P95_M

    @property
    def brake_ok(self) -> bool:
        return self.brake_dev.n == 0 or self.brake_dev.p95 <= BRAKE_DEV_P95_M

    @property
    def jackknife_ok(self) -> bool:
        shifts_ok = all(
            s.n == 0 or s.p95 <= JACKKNIFE_SHIFT_P95_M
            for s in (self.apex_shift, self.brake_on_shift, self.exit_end_shift)
        )
        return self.jackknife_count_matches == self.jackknife_trials and shifts_ok

    @property
    def corners_ok(self) -> bool:
        return self.corner_offset_max_m <= CORNER_OFFSET_MAX_M and self.events_without_corners == 0

    @property
    def ok(self) -> bool:
        return self.partition_ok and self.fixed_bins_ok and self.apex_ok and self.brake_ok and self.jackknife_ok and self.corners_ok

    def to_dict(self) -> dict:
        out = asdict(self)
        out["v_min_std_all_drivers_kmh"] = {str(k): v for k, v in self.v_min_std_all_drivers_kmh.items()}
        out["v_min_std_within_driver_kmh"] = {str(k): v for k, v in self.v_min_std_within_driver_kmh.items()}
        out["checks"] = {
            "partition": self.partition_ok,
            "fixed_bins": self.fixed_bins_ok,
            "apex_position": self.apex_ok,
            "braking_point": self.brake_ok,
            "jackknife": self.jackknife_ok,
            "corners": self.corners_ok,
            "all": self.ok,
        }
        return out


def measure_session(
    grid: pd.DataFrame,
    corners: pd.DataFrame | None,
    events: pd.DataFrame,
    sectors: pd.DataFrame,
    lap_length_m: float,
    params: EventParams = EventParams(),
    grid_m: float = GRID_SPACING_M,
    jackknife_trials: int = 10,
) -> SegmentationReport:
    """Run every criterion and collect the reported-not-gated figures."""
    phases = sectors[sectors["grain"] == GRAIN_CORNER_PHASE]
    bins = sectors[sectors["grain"] == GRAIN_FIXED_100M]

    apex = per_lap_apex(grid, events, lap_length_m)
    brake = per_lap_brake(grid, events, lap_length_m)
    jack = jackknife(grid, corners, events, params, trials=jackknife_trials, grid_m=grid_m)
    matches = int(jack.groupby("trial")["count_match"].all().sum()) if len(jack) else 0
    good = jack[jack["count_match"].fillna(False).astype(bool)] if len(jack) else jack

    std_all = apex.groupby("event_id")["v_min_kmh"].std().fillna(0.0) if len(apex) else pd.Series(dtype=float)
    std_within = (
        apex.groupby(["driver", "event_id"])["v_min_kmh"].std().groupby("event_id").median().fillna(0.0)
        if len(apex) else pd.Series(dtype=float)
    )

    if corners is not None and len(corners):
        offset_max = float(corners["line_offset_m"].max())
        labelled: set[int] = set()
        for label in events["corners"].dropna():
            labelled.update(parse_corner_label(label))
        unassigned = int((~corners["number"].astype(int).isin(labelled)).sum())
    else:
        offset_max, unassigned = float("nan"), 0

    return SegmentationReport(
        lap_length_m=float(lap_length_m),
        events=int(len(events)),
        partition_ok=check_partition(phases, lap_length_m, grid_m),
        fixed_bins_ok=check_partition(bins, lap_length_m, grid_m) and len(bins) == int(np.ceil(lap_length_m / 100.0 - 1e-9)),
        fixed_bins=int(len(bins)),
        apex_dev=Spread.of(apex["apex_dev_m"].to_numpy(dtype=float)),
        brake_dev=Spread.of(brake["brake_dev_m"].to_numpy(dtype=float)),
        jackknife_trials=jackknife_trials,
        jackknife_count_matches=matches,
        apex_shift=Spread.of(good["apex_shift_m"].to_numpy(dtype=float) if len(good) else np.empty(0)),
        brake_on_shift=Spread.of(good["brake_on_shift_m"].to_numpy(dtype=float) if len(good) else np.empty(0)),
        exit_end_shift=Spread.of(good["exit_end_shift_m"].to_numpy(dtype=float) if len(good) else np.empty(0)),
        corner_offset_max_m=offset_max,
        events_without_corners=int(events["corners"].isna().sum()) if corners is not None and len(corners) else 0,
        corners_without_event=unassigned,
        v_min_std_all_drivers_kmh={int(k): float(v) for k, v in std_all.items()},
        v_min_std_within_driver_kmh={int(k): float(v) for k, v in std_within.items()},
    )
