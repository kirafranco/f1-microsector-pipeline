"""F010: locating the official timing line on the aligned axis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validate.timing_line import (
    CROSSING_SCHEMA,
    TimingLineError,
    line_crossings,
    session_line_positions,
)

LINE_LENGTH_M = 1000.0
SPEED_KMH = 180.0
SPEED_MS = SPEED_KMH / 3.6
STEP_M = 10.0
STEP_S = STEP_M / SPEED_MS


def aligned_lap(driver: str = "AAA", lap_number: int = 1, lead_m: float = 30.0, trail_m: float = 20.0,
                window_open_s: float = 0.0, window_close_s: float = 0.0) -> pd.DataFrame:
    """A constant-speed lap whose axis runs from -lead_m to LINE_LENGTH_M + trail_m.

    The telemetry window is trimmed by ``window_open_s`` at the start and
    ``window_close_s`` at the end, so the true line crossings sit outside it.
    """
    start = -lead_m + window_open_s * SPEED_MS
    end = LINE_LENGTH_M + trail_m - window_close_s * SPEED_MS
    n = int(round((end - start) / STEP_M)) + 1
    distance = start + np.arange(n) * STEP_M
    return pd.DataFrame(
        {
            "driver": driver,
            "lap_number": lap_number,
            "session_time": 100.0 + np.arange(n) * STEP_S,
            "speed": SPEED_KMH,
            "distance_raw": np.arange(n) * STEP_M,
            "distance_aligned": distance,
        }
    )


def official_row(driver: str = "AAA", lap_number: int = 1, lead_m: float = 30.0, trail_m: float = 20.0,
                 window_open_s: float = 0.0) -> dict:
    """Official timing consistent with ``aligned_lap``.

    The true line crossings sit at axis ``-lead_m`` and ``LINE_LENGTH_M + trail_m``
    -- i.e. where an untrimmed telemetry window would begin and end -- so the lap
    started ``window_open_s`` before the first sample.
    """
    return {
        "driver": driver,
        "lap_number": lap_number,
        "lap_start_time": 100.0 - window_open_s,
        "lap_time": (LINE_LENGTH_M + lead_m + trail_m) / SPEED_MS,
        "sector1_time": np.nan,
        "sector2_time": np.nan,
        "sector3_time": np.nan,
    }


class TestLineCrossings:
    def test_recovers_the_designed_line_positions(self) -> None:
        # Line at 0 m and 1000 m: lap opens 30 m early and closes 20 m late.
        lap = aligned_lap()
        laps = pd.DataFrame([official_row()])
        out = line_crossings(lap, laps, LINE_LENGTH_M)
        assert len(out) == 1
        assert out["line_start_m"].iloc[0] == pytest.approx(-30.0, abs=0.1)
        assert out["line_end_m"].iloc[0] == pytest.approx(LINE_LENGTH_M + 20.0, abs=0.1)
        assert out["driven_m"].iloc[0] == pytest.approx(LINE_LENGTH_M + 50.0, abs=0.1)

    def test_window_offsets_are_reported(self) -> None:
        # Trims are whole grid steps (0.2 s = 10 m here) so the lap is representable.
        lap = aligned_lap(window_open_s=STEP_S, window_close_s=STEP_S)
        laps = pd.DataFrame([official_row(window_open_s=STEP_S)])
        out = line_crossings(lap, laps, LINE_LENGTH_M)
        assert out["window_open_s"].iloc[0] == pytest.approx(STEP_S, abs=0.02)
        assert out["window_close_s"].iloc[0] == pytest.approx(-STEP_S, abs=0.02)
        assert bool(out["start_extrapolated"].iloc[0]) and bool(out["end_extrapolated"].iloc[0])

    def test_crossings_are_recovered_even_when_extrapolated(self) -> None:
        """The line is outside the window: extrapolation must still find it."""
        lap = aligned_lap(window_open_s=0.4, window_close_s=0.4)
        laps = pd.DataFrame([official_row(window_open_s=0.4)])
        out = line_crossings(lap, laps, LINE_LENGTH_M)
        assert out["line_start_m"].iloc[0] == pytest.approx(-30.0, abs=0.5)
        assert out["line_end_m"].iloc[0] == pytest.approx(LINE_LENGTH_M + 20.0, abs=0.5)

    def test_anchor_margin_keeps_clamped_samples_out(self) -> None:
        """A clamped run at the axis start must not become the transfer anchor."""
        lap = aligned_lap()
        lap.loc[:2, "distance_aligned"] = lap.loc[2, "distance_aligned"]
        out = line_crossings(lap, pd.DataFrame([official_row()]), LINE_LENGTH_M)
        assert out["line_start_m"].iloc[0] == pytest.approx(-30.0, abs=0.1)

    def test_schema_and_multiple_laps(self) -> None:
        frames = [aligned_lap("AAA", 1), aligned_lap("BBB", 2, lead_m=40.0)]
        laps = pd.DataFrame([official_row("AAA", 1), official_row("BBB", 2, lead_m=40.0)])
        out = line_crossings(pd.concat(frames, ignore_index=True), laps, LINE_LENGTH_M)
        assert len(out) == 2
        assert list(out.columns) == list(CROSSING_SCHEMA)
        for column, dtype in CROSSING_SCHEMA.items():
            assert str(out[column].dtype) == dtype, column

    def test_lap_without_official_timing_is_skipped(self) -> None:
        frames = pd.concat([aligned_lap("AAA", 1), aligned_lap("BBB", 2)], ignore_index=True)
        laps = pd.DataFrame([official_row("AAA", 1)])
        assert len(line_crossings(frames, laps, LINE_LENGTH_M)) == 1

    def test_missing_lap_time_is_skipped(self) -> None:
        row = official_row()
        row["lap_time"] = np.nan
        with pytest.raises(TimingLineError, match="no lap"):
            line_crossings(aligned_lap(), pd.DataFrame([row]), LINE_LENGTH_M)

    def test_missing_column_is_an_error(self) -> None:
        with pytest.raises(TimingLineError, match="missing"):
            line_crossings(aligned_lap().drop(columns=["distance_raw"]), pd.DataFrame([official_row()]), LINE_LENGTH_M)


class TestSessionPositions:
    def test_medians_across_laps(self) -> None:
        frames, rows = [], []
        for i, lead in enumerate((28.0, 30.0, 32.0), start=1):
            frames.append(aligned_lap("AAA", i, lead_m=lead))
            rows.append(official_row("AAA", i, lead_m=lead))
        out = line_crossings(pd.concat(frames, ignore_index=True), pd.DataFrame(rows), LINE_LENGTH_M)
        start, end = session_line_positions(out)
        assert start == pytest.approx(-30.0, abs=0.2)
        assert end == pytest.approx(LINE_LENGTH_M + 20.0, abs=0.2)

    def test_empty_is_an_error(self) -> None:
        with pytest.raises(TimingLineError):
            session_line_positions(pd.DataFrame(columns=list(CROSSING_SCHEMA)))
