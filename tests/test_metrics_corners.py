"""F004 corner metrics through the F009 windows, on the designed session."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics.corners import CORNER_METRICS_SCHEMA, corner_metrics
from src.segment.events import EVENT_SCHEMA, detect_events
from tests import synthetic_session as syn


@pytest.fixture()
def events() -> pd.DataFrame:
    return detect_events(syn.traces(), None).assign(corners=pd.array(["T1", "T2"], dtype="string"))


@pytest.fixture()
def metrics(events: pd.DataFrame) -> pd.DataFrame:
    return corner_metrics(syn.grid(), events, syn.LAP_LENGTH_M)


class TestCornerMetrics:
    def test_one_row_per_lap_and_event(self, metrics: pd.DataFrame) -> None:
        assert len(metrics) == 4 * 2
        assert not metrics.duplicated(["driver", "lap_number", "event_id"]).any()

    def test_v_min_matches_the_design(self, metrics: pd.DataFrame) -> None:
        a = metrics[metrics["event_id"] == 0]
        b = metrics[metrics["event_id"] == 1]
        np.testing.assert_allclose(a["v_min_kmh"], 100.0, atol=0.6)
        np.testing.assert_allclose(b["v_min_kmh"], 286.0, atol=0.6)
        assert (np.abs(a["v_min_m"].to_numpy(dtype=float) - 800.0) <= 30.0).all()
        assert (np.abs(b["v_min_m"].to_numpy(dtype=float) - 1800.0) <= 10.0).all()

    def test_braking_point_and_its_uncertainty(self, metrics: pd.DataFrame) -> None:
        a = metrics[metrics["event_id"] == 0].set_index("driver")
        assert (a.loc["AAA", "brake_dev_m"] == 0.0).all()
        assert (a.loc["BBB", "brake_dev_m"] == 10.0).all()
        assert (a.loc["AAA", "brake_on_m"] == 640.0).all() and (a.loc["BBB", "brake_on_m"] == 650.0).all()
        assert (a["brake_gap_m"] == 8.0).all()  # the synthetic grid's constant source gap

    def test_lift_only_event_has_no_braking_columns(self, metrics: pd.DataFrame) -> None:
        b = metrics[metrics["event_id"] == 1]
        assert b["brake_on_m"].isna().all() and b["brake_dev_m"].isna().all() and b["brake_gap_m"].isna().all()

    def test_labels_and_flags_come_from_the_events(self, metrics: pd.DataFrame) -> None:
        assert metrics.groupby("event_id")["corners"].first().tolist() == ["T1", "T2"]
        assert metrics.groupby("event_id")["marginal"].first().tolist() == [False, True]

    def test_schema(self, metrics: pd.DataFrame) -> None:
        assert list(metrics.columns) == list(CORNER_METRICS_SCHEMA)
        for column, dtype in CORNER_METRICS_SCHEMA.items():
            assert str(metrics[column].dtype) == dtype, column

    def test_no_events_gives_an_empty_typed_table(self) -> None:
        empty = pd.DataFrame({c: pd.Series(dtype=t) for c, t in EVENT_SCHEMA.items()})
        out = corner_metrics(syn.grid(), empty, syn.LAP_LENGTH_M)
        assert out.empty and list(out.columns) == list(CORNER_METRICS_SCHEMA)
