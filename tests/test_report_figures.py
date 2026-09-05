"""F016 figures: they render, and a re-render of unchanged data is unchanged.

The document's figures are versioned PNGs. If matplotlib stamped a timestamp
into them, every rebuild would show up as a diff and the repository would churn
for no reason -- so byte stability is asserted, not assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.report import figures, traces

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

PHASES = pd.DataFrame({
    "sectors": [6, 4, 8, 8, 8],
    "length_m": [750.0, 250.0, 310.0, 700.0, 3490.0],
    "delta_s": [0.119, -0.125, 0.118, 0.017, -0.048],
    "sigma_s": [0.168, 0.107, 0.138, 0.183, 0.166],
}, index=pd.Index(["braking", "entry", "apex", "exit", "straight"], name="phase"))
PHASES["ratio"] = PHASES["delta_s"].abs() / PHASES["sigma_s"]

PAIRINGS = pd.DataFrame({
    "gap_s": [0.066, 0.408, 0.023, 0.365],
    "T5": [0.138, 0.210, 0.113, 0.185],
    "T11": [0.099, 0.205, -0.037, 0.069],
    "T1-T2": [-0.124, -0.093, -0.043, -0.012],
}, index=pd.Index(["VER L11 vs PER L11", "VER L11 vs PER L8",
                   "VER L8 vs PER L11", "VER L8 vs PER L8"], name="pair"))
NOISE = pd.Series({"T5": 0.090, "T11": 0.136, "T1-T2": 0.109}, name="sigma_s")

RECONCILIATION = pd.DataFrame({
    "official_gap_s": [0.069, -0.087, 0.084],
    "grid_gap_s": [0.013, -0.068, 0.142],
    "difference_s": [-0.056, 0.019, 0.058],
    "f010_residual_difference_s": [-0.059, 0.018, 0.061],
    "unexplained_s": [0.0034, 0.0011, -0.0035],
}, index=["s1", "s2", "s3"])


@pytest.fixture(scope="module")
def trace() -> pd.DataFrame:
    distance = np.arange(1280.0, 1420.0, 10.0)
    return pd.DataFrame({
        "distance_m": distance,
        "speed_a": 225.0 + np.linspace(0, 12, len(distance)),
        "speed_b": 220.0 + np.linspace(0, 12, len(distance)),
        "throttle_a": np.clip(np.linspace(50, 100, len(distance)), 0, 100),
        "throttle_b": np.clip(np.linspace(30, 100, len(distance)), 0, 100),
        "cumulative_delta_s": np.linspace(-0.05, 0.15, len(distance)),
    })


def render_all(directory: Path, trace: pd.DataFrame) -> list[Path]:
    return [
        figures.fig_phase_totals(PHASES, directory / "phases.png", label_a="VER L11",
                                 label_b="PER L11", official_gap_s=0.066, total_s=0.081),
        figures.fig_event_pairings(PAIRINGS, NOISE, directory / "pairings.png",
                                   driver_a="VER", driver_b="PER", highlight="T5"),
        figures.fig_trace_window(trace, directory / "trace.png", label_a="VER L11",
                                 label_b="PER L11", title="T5", subtitle="speed and throttle"),
        figures.fig_sector_reconciliation(RECONCILIATION, directory / "sectors.png",
                                          label_a="VER L11", label_b="PER L11"),
    ]


class TestRendering:
    def test_every_figure_is_written(self, tmp_path: Path, trace: pd.DataFrame) -> None:
        paths = render_all(tmp_path, trace)
        assert len(paths) == 4
        for path in paths:
            assert path.exists(), path.name
            assert path.stat().st_size > 5_000, f"{path.name} is suspiciously small"

    def test_they_are_png(self, tmp_path: Path, trace: pd.DataFrame) -> None:
        for path in render_all(tmp_path, trace):
            assert path.read_bytes()[:8] == PNG_MAGIC, path.name

    def test_none_is_large_enough_to_bloat_the_repository(self, tmp_path: Path, trace: pd.DataFrame) -> None:
        paths = render_all(tmp_path, trace)
        for path in paths:
            assert path.stat().st_size <= 300_000, f"{path.name} is {path.stat().st_size} bytes"
        assert sum(path.stat().st_size for path in paths) <= 1_500_000

    def test_a_rebuild_of_unchanged_data_produces_identical_bytes(self, tmp_path: Path, trace: pd.DataFrame) -> None:
        """Otherwise every re-run of the pipeline dirties the working tree."""
        first = {path.name: path.read_bytes() for path in render_all(tmp_path / "one", trace)}
        second = {path.name: path.read_bytes() for path in render_all(tmp_path / "two", trace)}
        assert first.keys() == second.keys()
        for name in first:
            assert first[name] == second[name], f"{name} is not reproducible"

    def test_a_directory_that_does_not_exist_yet_is_created(self, tmp_path: Path, trace: pd.DataFrame) -> None:
        path = figures.fig_phase_totals(PHASES, tmp_path / "deep" / "nested" / "phases.png",
                                        label_a="A", label_b="B", official_gap_s=0.0, total_s=0.0)
        assert path.exists()


class TestWhatTheFiguresShow:
    def test_a_delta_bar_is_coloured_by_who_lost_the_time(self) -> None:
        assert figures._delta_colours([0.1, -0.1]) == [figures.COLOUR_POSITIVE, figures.COLOUR_NEGATIVE]

    def test_the_backend_needs_no_display(self) -> None:
        """The figures have to render in a container and in CI."""
        import matplotlib
        assert matplotlib.get_backend().lower() == "agg"


class TestTraceWindow:
    """`fig_trace_window` reads columns that `traces.window` has to produce."""

    def test_the_window_frame_carries_what_the_figure_plots(self) -> None:
        grid = pd.concat([_synthetic_lap("AAA", 1, 0.0), _synthetic_lap("BBB", 1, -5.0)], ignore_index=True)
        frame = traces.window(grid, ("AAA", 1), ("BBB", 1), 100.0, 300.0)
        for column in ("distance_m", "speed_a", "speed_b", "throttle_a", "throttle_b", "cumulative_delta_s"):
            assert column in frame.columns, column

    def test_the_running_delta_starts_at_zero_inside_the_window(self) -> None:
        """Carrying in the delta accumulated before the window would make every
        corner look like the sum of the ones before it."""
        grid = pd.concat([_synthetic_lap("AAA", 1, 0.0), _synthetic_lap("BBB", 1, -5.0)], ignore_index=True)
        frame = traces.window(grid, ("AAA", 1), ("BBB", 1), 100.0, 300.0)
        assert frame["cumulative_delta_s"].iloc[0] == pytest.approx(0.0)
        assert frame["cumulative_delta_s"].iloc[-1] > 0, "the slower lap loses time across the window"

    def test_the_deficit_counts_points_as_well_as_averaging_them(self) -> None:
        grid = pd.concat([_synthetic_lap("AAA", 1, 0.0), _synthetic_lap("BBB", 1, -5.0)], ignore_index=True)
        deficit = traces.speed_deficit(grid, ("AAA", 1), ("BBB", 1), 100.0, 300.0)
        assert deficit.mean_kmh == pytest.approx(-5.0)
        assert deficit.share_slower == 1.0
        assert deficit.one_sided is True

    def test_a_window_off_the_end_of_the_lap_is_an_error(self) -> None:
        grid = pd.concat([_synthetic_lap("AAA", 1, 0.0), _synthetic_lap("BBB", 1, 0.0)], ignore_index=True)
        with pytest.raises(Exception, match="no shared grid points"):
            traces.window(grid, ("AAA", 1), ("BBB", 1), 9_000.0, 9_500.0)


def _synthetic_lap(driver: str, lap: int, speed_offset: float) -> pd.DataFrame:
    distance = np.arange(0.0, 500.0, 10.0)
    speed = 200.0 + speed_offset
    return pd.DataFrame({
        "driver": driver, "lap_number": lap, "grid_index": np.arange(len(distance)),
        "distance_m": distance, "speed": speed, "throttle": 100.0, "n_gear": 6,
        "brake": False, "source_gap_m": 8.0,
        "elapsed_time": distance / (speed / 3.6),
    })
