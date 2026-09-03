"""F004 micro-sector times: telescoping, partial flags, summary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics.delta import delta_t, time_curves
from src.metrics.microsectors import SUMMARY_SCHEMA, TIMES_SCHEMA, microsector_times, summarise_microsectors
from src.metrics.reference import lap_index
from src.metrics.validation import closure_error
from src.segment.events import detect_events
from src.segment.phases import build_corner_phases, build_fixed_bins
from tests import synthetic_session as syn


@pytest.fixture()
def microsectors() -> pd.DataFrame:
    events = detect_events(syn.traces(), None)
    return pd.concat(
        [build_corner_phases(events, syn.LAP_LENGTH_M), build_fixed_bins(syn.LAP_LENGTH_M)], ignore_index=True
    )


@pytest.fixture()
def curves() -> pd.DataFrame:
    return time_curves(syn.grid())


@pytest.fixture()
def reference(curves: pd.DataFrame) -> pd.Series:
    return pd.Series([("AAA", 1)] * len(curves), index=lap_index(curves.index))


class TestTimes:
    def test_sector_times_telescope_to_the_lap(self, curves, microsectors, reference) -> None:
        times = microsector_times(curves, microsectors, reference)
        for grain in ("corner_phase", "fixed_100m"):
            total = times[times["grain"] == grain].groupby(["driver", "lap_number"], observed=True)["time_s"].sum()
            last = curves.iloc[:, -1]
            np.testing.assert_allclose(total.to_numpy(dtype=float), last.to_numpy(dtype=float), atol=1e-3)

    def test_only_the_last_sector_is_partial(self, curves, microsectors, reference) -> None:
        times = microsector_times(curves, microsectors, reference)
        phase = times[times["grain"] == "corner_phase"]
        assert phase["partial"].sum() == len(curves)  # one per lap
        assert (phase.loc[phase["partial"].astype(bool), "microsector_id"] == phase["microsector_id"].max()).all()
        assert times["time_s"].notna().all()

    def test_shortened_lap_gets_nan_and_partial(self, microsectors, reference) -> None:
        g = syn.grid()
        short = g[~((g["driver"] == "BBB") & (g["lap_number"] == 2) & (g["grid_index"] >= 150))]
        times = microsector_times(time_curves(short), microsectors, reference)
        rows = times[(times["driver"] == "BBB") & (times["lap_number"] == 2) & (times["grain"] == "corner_phase")]
        assert rows["time_s"].isna().sum() > 0
        # Every sector the lap does not finish is flagged, whether it was entered or not.
        assert rows.loc[rows["time_s"].isna(), "partial"].astype(bool).all()
        cut = rows[rows["partial"].astype(bool) & rows["time_s"].notna()]
        assert len(cut) == 1  # exactly one sector is entered but not finished
        assert (times["time_s"].isna() <= times["partial"].astype(bool)).all()

    def test_delta_is_zero_on_the_reference_and_closes(self, curves, microsectors, reference) -> None:
        times = microsector_times(curves, microsectors, reference)
        ref_rows = times[(times["driver"] == "AAA") & (times["lap_number"] == 1)]
        assert (ref_rows["delta_s"] == 0.0).all()
        delta = delta_t(curves, reference, "session_fastest")
        assert closure_error(times, delta, microsectors) <= 1e-3

    def test_length_travels_with_every_row(self, curves, microsectors, reference) -> None:
        times = microsector_times(curves, microsectors, reference)
        first = times[(times["grain"] == "corner_phase") & (times["microsector_id"] == 0)]
        assert (first["length_m"] == 640.0).all()

    def test_schema(self, curves, microsectors, reference) -> None:
        times = microsector_times(curves, microsectors, reference)
        assert list(times.columns) == list(TIMES_SCHEMA)
        for column, dtype in TIMES_SCHEMA.items():
            assert str(times[column].dtype) == dtype, column
        assert len(times) == len(curves) * len(microsectors)


class TestSummary:
    def test_statistics(self, curves, microsectors, reference) -> None:
        times = microsector_times(curves, microsectors, reference)
        summary = summarise_microsectors(times, microsectors, ("AAA", 1))
        assert len(summary) == len(microsectors)
        phase = summary[summary["grain"] == "corner_phase"].set_index("microsector_id")
        assert (phase["n_laps"].iloc[:-1] == 4).all()
        assert phase["n_laps"].iloc[-1] == 0  # partial on every lap: excluded from the spread
        ref_first = times[(times["driver"] == "AAA") & (times["lap_number"] == 1) & (times["grain"] == "corner_phase") & (times["microsector_id"] == 0)]["time_s"].iloc[0]
        assert phase.loc[0, "ref_s"] == pytest.approx(float(ref_first))
        assert (phase["min_s"].iloc[:-1] <= phase["mean_s"].iloc[:-1] + 1e-6).all()
        assert phase["within_driver_std_s"].iloc[:-1].notna().all()
        assert phase.loc[0, "length_m"] == 640.0 and phase.loc[0, "phase"] == "straight"

    def test_identical_laps_have_zero_spread(self, microsectors) -> None:
        g = syn.grid()
        one = g[(g["driver"] == "AAA") & (g["lap_number"] == 1)]
        clones = pd.concat([one.assign(lap_number=np.int16(k)) for k in (1, 2, 3)], ignore_index=True).astype(g.dtypes.to_dict())
        curves = time_curves(clones)
        reference = pd.Series([("AAA", 1)] * 3, index=lap_index(curves.index))
        summary = summarise_microsectors(microsector_times(curves, microsectors, reference), microsectors, ("AAA", 1))
        phase = summary[(summary["grain"] == "corner_phase") & (summary["n_laps"] > 0)]
        assert (phase["std_s"] == 0.0).all() and (phase["within_driver_std_s"] == 0.0).all()

    def test_no_reference_key_gives_null_ref(self, curves, microsectors, reference) -> None:
        summary = summarise_microsectors(microsector_times(curves, microsectors, reference), microsectors, None)
        assert summary["ref_s"].isna().all()

    def test_schema(self, curves, microsectors, reference) -> None:
        summary = summarise_microsectors(microsector_times(curves, microsectors, reference), microsectors, ("AAA", 1))
        assert list(summary.columns) == list(SUMMARY_SCHEMA)
        for column, dtype in SUMMARY_SCHEMA.items():
            assert str(summary[column].dtype) == dtype, column
