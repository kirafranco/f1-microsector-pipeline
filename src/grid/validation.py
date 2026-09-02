"""Acceptance measurements for the distance grid (F003 criteria 1-7).

Everything here compares a resampled lap against the source lap it came from,
so the same functions serve the unit tests, the integration test and the
per-session report written next to the parquet output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.grid.resample import (
    DISCRETE_CHANNELS,
    DISTANCE_COLUMN,
    GRID_SPACING_M,
    grid_point_count,
)

#: Thresholds from the spec, each measured on the Suzuka 2024 Q prototype
#: before being written down.
SPEED_P95_KMH = 2.0
SPEED_P99_KMH = 4.0
THROTTLE_P95_PCT = 6.0
RPM_P95 = 200.0
BRAKE_EDGE_P95_M = 10.0

ROUND_TRIP_CHANNELS = ("speed", "throttle", "rpm")


def check_grid_structure(
    grid: pd.DataFrame, lap_length_m: float, grid_m: float = GRID_SPACING_M
) -> bool:
    """Criterion 1: exact count, uniform spacing, monotone index and distance."""
    if len(grid) != grid_point_count(lap_length_m, grid_m):
        return False
    index = grid["grid_index"].to_numpy(dtype=np.int64)
    distance = grid["distance_m"].to_numpy(dtype=float)
    if not np.array_equal(index, np.arange(len(grid))):
        return False
    # float4 storage: compare at float32 precision, not exactly.
    expected = (np.arange(len(grid)) * grid_m).astype(np.float32)
    return np.array_equal(distance.astype(np.float32), expected)


def round_trip_error(lap: pd.DataFrame, grid: pd.DataFrame, channel: str) -> np.ndarray:
    """Criteria 2-4: absolute error when the grid is read back at source distances.

    Only source samples inside the grid's range are compared; a sample beyond
    the last grid point has nothing to be read back from.
    """
    source_d = lap[DISTANCE_COLUMN].to_numpy(dtype=float)
    source_v = lap[channel].to_numpy(dtype=float)
    grid_d = grid["distance_m"].to_numpy(dtype=float)
    grid_v = grid[channel].to_numpy(dtype=float)

    inside = (source_d >= grid_d[0]) & (source_d <= grid_d[-1])
    back = np.interp(source_d[inside], grid_d, grid_v)
    return np.abs(back - source_v[inside])


def invented_values(lap: pd.DataFrame, grid: pd.DataFrame, channel: str) -> set:
    """Criterion 5: grid values of a discrete channel absent from the source lap."""
    source = set(lap[channel].dropna().unique().tolist())
    on_grid = set(grid[channel].dropna().unique().tolist())
    return on_grid - source


def _rising_edges(distance: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Distances at which a boolean state goes False -> True."""
    on = state.astype(bool)
    edge = np.zeros(len(on), dtype=bool)
    edge[1:] = on[1:] & ~on[:-1]
    return distance[edge]


def brake_edge_displacement(
    lap: pd.DataFrame, grid: pd.DataFrame, channel: str = "brake"
) -> np.ndarray:
    """Criterion 6: distance from each source brake-on edge to the nearest grid edge.

    A brake application too short to reach a grid point has no grid edge of
    its own and is matched to whatever grid edge is nearest -- that is the
    documented worst case, and it is reported rather than hidden.
    """
    source_edges = _rising_edges(
        lap[DISTANCE_COLUMN].to_numpy(dtype=float),
        lap[channel].fillna(False).to_numpy(),
    )
    grid_edges = _rising_edges(
        grid["distance_m"].to_numpy(dtype=float),
        grid[channel].fillna(False).to_numpy(),
    )
    if len(source_edges) == 0:
        return np.empty(0)
    if len(grid_edges) == 0:
        return np.full(len(source_edges), np.nan)
    nearest = grid_edges[np.abs(source_edges[:, None] - grid_edges[None, :]).argmin(axis=1)]
    return np.abs(source_edges - nearest)


def elapsed_time_strictly_increasing(grid: pd.DataFrame) -> bool:
    """Criterion 7."""
    elapsed = grid["elapsed_time"].to_numpy(dtype=float)
    return bool((np.diff(elapsed) > 0).all())


def empty_bin_fraction(lap: pd.DataFrame, grid_m: float = GRID_SPACING_M) -> float:
    """Share of grid bins ``[i*grid_m, (i+1)*grid_m)`` holding no source sample.

    The documented limitation, measured rather than asserted.
    """
    distance = lap[DISTANCE_COLUMN].to_numpy(dtype=float)
    n_bins = grid_point_count(float(distance[-1]), grid_m)
    occupied = np.unique(np.floor(distance / grid_m).astype(np.int64))
    occupied = occupied[(occupied >= 0) & (occupied < n_bins)]
    return 1.0 - len(occupied) / n_bins


@dataclass(frozen=True)
class ChannelErrors:
    p95: float
    p99: float
    max: float
    n: int

    @staticmethod
    def of(errors: np.ndarray) -> "ChannelErrors":
        finite = errors[np.isfinite(errors)]
        if len(finite) == 0:
            return ChannelErrors(p95=float("nan"), p99=float("nan"), max=float("nan"), n=0)
        return ChannelErrors(
            p95=float(np.percentile(finite, 95)),
            p99=float(np.percentile(finite, 99)),
            max=float(finite.max()),
            n=int(len(finite)),
        )


@dataclass(frozen=True)
class GridReport:
    """Acceptance table for one session, plus the reported-not-gated figures."""

    grid_m: float
    laps: int
    structure_ok: bool
    elapsed_ok: bool
    speed: ChannelErrors
    throttle: ChannelErrors
    rpm: ChannelErrors
    invented: dict[str, int]
    brake_edge: ChannelErrors
    empty_bin_fraction: float
    source_gap: ChannelErrors
    source_gap_median_m: float

    @property
    def speed_ok(self) -> bool:
        return self.speed.p95 <= SPEED_P95_KMH and self.speed.p99 <= SPEED_P99_KMH

    @property
    def throttle_ok(self) -> bool:
        return self.throttle.p95 <= THROTTLE_P95_PCT

    @property
    def rpm_ok(self) -> bool:
        return self.rpm.p95 <= RPM_P95

    @property
    def discrete_ok(self) -> bool:
        return all(count == 0 for count in self.invented.values())

    @property
    def brake_edge_ok(self) -> bool:
        return self.brake_edge.n == 0 or self.brake_edge.p95 <= BRAKE_EDGE_P95_M

    @property
    def ok(self) -> bool:
        return (
            self.structure_ok
            and self.elapsed_ok
            and self.speed_ok
            and self.throttle_ok
            and self.rpm_ok
            and self.discrete_ok
            and self.brake_edge_ok
        )

    def to_dict(self) -> dict:
        out = asdict(self)
        out["checks"] = {
            "structure": self.structure_ok,
            "elapsed_time_increasing": self.elapsed_ok,
            "speed": self.speed_ok,
            "throttle": self.throttle_ok,
            "rpm": self.rpm_ok,
            "discrete_no_invented_values": self.discrete_ok,
            "brake_edge": self.brake_edge_ok,
            "all": self.ok,
        }
        return out


def measure_session(
    pairs: Iterable[tuple[pd.DataFrame, pd.DataFrame]], grid_m: float = GRID_SPACING_M
) -> GridReport:
    """Run every criterion over ``(source lap, resampled lap)`` pairs."""
    structure_ok = True
    elapsed_ok = True
    errors: dict[str, list[np.ndarray]] = {c: [] for c in ROUND_TRIP_CHANNELS}
    invented: dict[str, int] = {c: 0 for c in DISCRETE_CHANNELS}
    edges: list[np.ndarray] = []
    empty: list[float] = []
    gaps: list[np.ndarray] = []
    laps = 0

    for lap, grid in pairs:
        laps += 1
        lap_length = float(lap[DISTANCE_COLUMN].to_numpy(dtype=float)[-1])
        structure_ok &= check_grid_structure(grid, lap_length, grid_m)
        elapsed_ok &= elapsed_time_strictly_increasing(grid)
        for channel in ROUND_TRIP_CHANNELS:
            errors[channel].append(round_trip_error(lap, grid, channel))
        for channel in DISCRETE_CHANNELS:
            invented[channel] += len(invented_values(lap, grid, channel))
        edges.append(brake_edge_displacement(lap, grid))
        empty.append(empty_bin_fraction(lap, grid_m))
        gaps.append(grid["source_gap_m"].to_numpy(dtype=float))

    def stack(parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts) if parts else np.empty(0)

    gap_all = stack(gaps)
    gap_finite = gap_all[np.isfinite(gap_all)]
    return GridReport(
        grid_m=grid_m,
        laps=laps,
        structure_ok=bool(structure_ok),
        elapsed_ok=bool(elapsed_ok),
        speed=ChannelErrors.of(stack(errors["speed"])),
        throttle=ChannelErrors.of(stack(errors["throttle"])),
        rpm=ChannelErrors.of(stack(errors["rpm"])),
        invented=invented,
        brake_edge=ChannelErrors.of(stack(edges)),
        empty_bin_fraction=float(np.mean(empty)) if empty else float("nan"),
        source_gap=ChannelErrors.of(gap_all),
        source_gap_median_m=float(np.median(gap_finite)) if len(gap_finite) else float("nan"),
    )
