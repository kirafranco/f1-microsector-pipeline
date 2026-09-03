"""F009 corner placement on the aligned axis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.segment.corners import (
    CORNER_SCHEMA,
    CornerError,
    corner_label,
    corner_positions,
    load_frame,
    parse_corner_label,
    parse_reference_lap,
)
from tests import synthetic_session as syn


class TestFrameAndLabels:
    def test_load_frame_round_trips_the_meta(self) -> None:
        frame = load_frame(syn.frame_meta())
        assert frame.rotation_deg == pytest.approx(syn.FRAME_ROTATION_DEG)
        np.testing.assert_allclose(frame.translation, syn.FRAME_TRANSLATION_M)

    def test_load_frame_rejects_missing_frame(self) -> None:
        with pytest.raises(CornerError):
            load_frame({"reference_line_lap": "VER L11"})

    @pytest.mark.parametrize(("label", "expected"), [("VER L11", ("VER", 11)), (" HAM L3 ", ("HAM", 3))])
    def test_parse_reference_lap(self, label: str, expected: tuple[str, int]) -> None:
        assert parse_reference_lap(label) == expected

    @pytest.mark.parametrize("bad", ["VER 11", "L11", "", "VER L"])
    def test_parse_reference_lap_rejects_garbage(self, bad: str) -> None:
        with pytest.raises(CornerError):
            parse_reference_lap(bad)

    def test_corner_label_formatting(self) -> None:
        assert corner_label([1, 2]) == "T1-T2"
        assert corner_label([5]) == "T5"
        assert corner_label([8, 8], ["a", "b"]) == "T8a-T8b"
        assert corner_label([]) is None

    def test_parse_corner_label(self) -> None:
        assert parse_corner_label("T1-T2") == [1, 2]
        assert parse_corner_label("T8a-T8b") == [8, 8]
        assert parse_corner_label(None) == []
        assert parse_corner_label(pd.NA) == []
        with pytest.raises(CornerError):
            parse_corner_label("turn 3")


class TestCornerPositions:
    @pytest.fixture()
    def placed(self) -> pd.DataFrame:
        return corner_positions(syn.grid(), syn.raw_corners(), load_frame(syn.frame_meta()), syn.REFERENCE_LAP)

    def test_recovers_the_designed_distances(self, placed: pd.DataFrame) -> None:
        assert placed["number"].tolist() == [1, 2, 3]
        assert placed["distance_m"].tolist() == list(syn.CORNER_DISTANCES.values())

    def test_frame_correction_puts_corners_on_the_line(self, placed: pd.DataFrame) -> None:
        """Without the transform the corners would sit ~30 m off the driven line."""
        assert placed["line_offset_m"].max() < 0.5

    def test_raw_distance_is_carried_not_used(self, placed: pd.DataFrame) -> None:
        np.testing.assert_allclose(placed["raw_distance_m"], placed["distance_m"] - 30.0)

    def test_schema(self, placed: pd.DataFrame) -> None:
        assert list(placed.columns) == list(CORNER_SCHEMA)
        for column, dtype in CORNER_SCHEMA.items():
            assert str(placed[column].dtype) == dtype, column

    def test_missing_reference_lap_is_an_error(self) -> None:
        with pytest.raises(CornerError, match="reference lap"):
            corner_positions(syn.grid(), syn.raw_corners(), load_frame(syn.frame_meta()), ("ZZZ", 9))

    def test_missing_columns_is_an_error(self) -> None:
        with pytest.raises(CornerError, match="missing columns"):
            corner_positions(syn.grid(), syn.raw_corners().drop(columns=["x"]), load_frame(syn.frame_meta()), syn.REFERENCE_LAP)

    def test_output_is_sorted_by_distance(self) -> None:
        shuffled = syn.raw_corners().iloc[::-1].reset_index(drop=True)
        placed = corner_positions(syn.grid(), shuffled, load_frame(syn.frame_meta()), syn.REFERENCE_LAP)
        assert placed["distance_m"].is_monotonic_increasing
