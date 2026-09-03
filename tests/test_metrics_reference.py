"""F004 reference selection (D7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics.reference import (
    ReferenceError,
    ReferenceSpec,
    accurate_lap_times,
    lap_index,
    lap_label,
    resolve_reference,
)


def laps_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver": pd.array(["VER", "VER", "VER", "NOR", "NOR", "HAM"], dtype="string"),
            "lap_number": pd.array([8, 11, 14, 9, 12, 5], dtype="Int16"),
            "lap_time": [88.24, 88.197, np.nan, 88.489, 88.585, 87.0],
            "is_accurate": pd.array([True, True, True, True, True, False], dtype="boolean"),
        }
    )


KEYS = [("VER", 8), ("VER", 11), ("VER", 14), ("NOR", 9), ("NOR", 12), ("HAM", 5)]


class TestSpec:
    def test_defaults_to_session_fastest(self) -> None:
        assert ReferenceSpec().kind == "session_fastest"
        assert ReferenceSpec().label == "session_fastest"

    def test_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ReferenceError, match="unknown reference kind"):
            ReferenceSpec(kind="ideal")

    def test_lap_kind_needs_a_lap(self) -> None:
        with pytest.raises(ReferenceError, match="needs driver and lap_number"):
            ReferenceSpec(kind="lap", driver="VER")
        assert ReferenceSpec(kind="lap", driver="VER", lap_number=11).label == "lap VER L11"

    def test_labels_and_index_normalise_types(self) -> None:
        assert lap_label(("VER", np.int16(11))) == "VER L11"
        index = lap_index([("VER", np.int16(11)), ("NOR", 9)])
        assert index.tolist() == [("VER", 11), ("NOR", 9)]
        assert index.names == ["driver", "lap_number"]


class TestAccurateLapTimes:
    def test_filters_inaccurate_and_untimed(self) -> None:
        times = accurate_lap_times(laps_table())
        assert times.index.tolist() == [("VER", 8), ("VER", 11), ("NOR", 9), ("NOR", 12)]
        assert times[("VER", 11)] == pytest.approx(88.197)

    def test_missing_column_is_an_error(self) -> None:
        with pytest.raises(ReferenceError, match="missing"):
            accurate_lap_times(laps_table().drop(columns=["is_accurate"]))


class TestResolve:
    def test_session_fastest_ignores_inaccurate_laps(self) -> None:
        ref = resolve_reference(laps_table(), KEYS, ReferenceSpec())
        assert set(ref.tolist()) == {("VER", 11)}  # HAM's 87.0 is not accurate
        assert ref.index.tolist() == lap_index(KEYS).tolist()

    def test_session_fastest_only_considers_laps_on_the_grid(self) -> None:
        ref = resolve_reference(laps_table(), [("NOR", 9), ("NOR", 12)], ReferenceSpec())
        assert set(ref.tolist()) == {("NOR", 9)}

    def test_session_fastest_with_no_timed_lap_is_an_error(self) -> None:
        with pytest.raises(ReferenceError, match="no accurate timed lap"):
            resolve_reference(laps_table(), [("HAM", 5), ("VER", 14)], ReferenceSpec())

    def test_nominated_lap(self) -> None:
        ref = resolve_reference(laps_table(), KEYS, ReferenceSpec("lap", "NOR", 12))
        assert set(ref.tolist()) == {("NOR", 12)}

    def test_nominated_lap_must_be_on_the_grid(self) -> None:
        with pytest.raises(ReferenceError, match="not on the grid"):
            resolve_reference(laps_table(), KEYS[:2], ReferenceSpec("lap", "NOR", 12))

    def test_driver_best(self) -> None:
        ref = resolve_reference(laps_table(), KEYS[:5], ReferenceSpec("driver_best"))
        assert ref[("VER", 8)] == ("VER", 11)
        assert ref[("VER", 14)] == ("VER", 11)
        assert ref[("NOR", 12)] == ("NOR", 9)

    def test_driver_best_without_a_timed_lap_is_an_error(self) -> None:
        with pytest.raises(ReferenceError, match="HAM"):
            resolve_reference(laps_table(), KEYS, ReferenceSpec("driver_best"))

    def test_empty_keys_is_an_error(self) -> None:
        with pytest.raises(ReferenceError):
            resolve_reference(laps_table(), [], ReferenceSpec())
