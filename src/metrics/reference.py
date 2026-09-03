"""Reference-lap selection for delta-t (D7: the reference is a parameter).

Selection always uses F002's official ``lap_time``, never grid time, so the
choice of reference cannot depend on how the telemetry window was sliced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

REFERENCE_KINDS = ("session_fastest", "lap", "driver_best")


class ReferenceError(ValueError):
    """No usable reference lap for the requested specification."""


@dataclass(frozen=True)
class ReferenceSpec:
    """Which lap every other lap is compared against.

    ``session_fastest``: the accurate lap with the smallest official lap time.
    ``lap``: the nominated ``(driver, lap_number)``.
    ``driver_best``: each lap against its own driver's fastest accurate lap.
    """

    kind: str = "session_fastest"
    driver: str | None = None
    lap_number: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in REFERENCE_KINDS:
            raise ReferenceError(f"unknown reference kind {self.kind!r}; expected one of {REFERENCE_KINDS}")
        if self.kind == "lap" and (self.driver is None or self.lap_number is None):
            raise ReferenceError("reference kind 'lap' needs driver and lap_number")

    @property
    def label(self) -> str:
        if self.kind == "lap":
            return f"lap {self.driver} L{int(self.lap_number)}"
        return self.kind


def lap_label(key: tuple) -> str:
    """``("VER", 11)`` -> ``"VER L11"``."""
    driver, lap_number = key
    return f"{driver} L{int(lap_number)}"


def lap_index(keys: Iterable[tuple]) -> pd.MultiIndex:
    """Normalised ``(driver, lap_number)`` index: plain str and int."""
    tuples = [(str(d), int(n)) for d, n in keys]
    return pd.MultiIndex.from_tuples(tuples, names=["driver", "lap_number"])


def accurate_lap_times(laps: pd.DataFrame) -> pd.Series:
    """Official lap time of every accurate, timed lap, indexed by lap key."""
    for column in ("driver", "lap_number", "lap_time", "is_accurate"):
        if column not in laps.columns:
            raise ReferenceError(f"laps frame is missing {column!r}")
    mask = laps["is_accurate"].fillna(False).astype(bool) & laps["lap_time"].notna()
    good = laps.loc[mask]
    return pd.Series(
        good["lap_time"].to_numpy(dtype=float),
        index=lap_index(zip(good["driver"], good["lap_number"])),
        name="lap_time",
    )


def resolve_reference(laps: pd.DataFrame, keys: Iterable[tuple], spec: ReferenceSpec) -> pd.Series:
    """Reference lap key for every lap in ``keys``.

    Returns a Series indexed by the normalised lap key whose values are the
    reference lap keys (tuples). Only laps present in ``keys`` -- i.e. on the
    grid -- can be references.
    """
    index = lap_index(keys)
    if len(index) == 0:
        raise ReferenceError("no laps to reference")
    times = accurate_lap_times(laps).reindex(index)

    if spec.kind == "session_fastest":
        if times.notna().sum() == 0:
            raise ReferenceError("no accurate timed lap on the grid to use as session fastest")
        reference = times.idxmin()
        return pd.Series([reference] * len(index), index=index, name="reference")

    if spec.kind == "lap":
        reference = (str(spec.driver), int(spec.lap_number))
        if reference not in index:
            raise ReferenceError(f"nominated reference {lap_label(reference)} is not on the grid")
        return pd.Series([reference] * len(index), index=index, name="reference")

    # driver_best
    best: dict[str, tuple] = {}
    missing: list[str] = []
    for driver, group in times.groupby(level="driver", sort=False):
        if group.notna().sum() == 0:
            missing.append(str(driver))
            continue
        best[str(driver)] = group.idxmin()
    if missing:
        raise ReferenceError(f"no accurate timed lap for driver(s) {missing}; cannot build driver_best")
    return pd.Series([best[d] for d in index.get_level_values("driver")], index=index, name="reference")
