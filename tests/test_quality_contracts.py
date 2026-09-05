"""F011 contracts: the synthetic session passes clean, and each injected defect is caught.

The frames come from the designed session run through the real F009, F004 and
F010 runners, so the contracts are checked against the shapes the pipeline
actually produces rather than against hand-written stubs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.metrics.session import compute_metrics
from src.quality.contracts import CONTRACTS, OPTIONAL_TABLES, SESSION_TABLES
from src.quality.engine import validate_tables
from src.quality.session import load_artefacts
from src.validate.session import validate_session
from tests import synthetic_session as syn


@pytest.fixture(scope="module")
def frames(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    root = tmp_path_factory.mktemp("quality")
    roots = syn.write_full_session(root)
    processed = root / "processed" / "synthetic"
    compute_metrics(roots["grid_root"], roots["microsector_root"], roots["snapshot_root"], roots["aligned_root"],
                    out_root=processed)
    validate_session(roots["snapshot_root"], roots["aligned_root"], roots["grid_root"], processed,
                     out_root=processed, min_laps=2)
    return load_artefacts(roots["snapshot_root"], roots["aligned_root"], roots["grid_root"],
                          roots["microsector_root"], processed)


def corrupt(frames: dict[str, pd.DataFrame], table: str, mutate) -> dict[str, pd.DataFrame]:
    """A copy of the frames with one table damaged."""
    out = {name: value.copy(deep=True) for name, value in frames.items()}
    out[table] = mutate(out[table])
    return out


#: Reference tables (F012) are ingested per season, so a session's frames never
#: contain them; validating a session against them would report 5 missing tables.
SESSION_CONTRACTS = {name: CONTRACTS[name] for name in sorted(SESSION_TABLES)}


def findings_for(frames: dict[str, pd.DataFrame], table: str) -> list:
    return [f for f in validate_tables(frames, SESSION_CONTRACTS).findings if f.table == table]


class TestCleanSession:
    def test_every_artefact_is_present_and_contracted(self, frames) -> None:
        assert set(frames) == set(SESSION_TABLES), set(SESSION_TABLES) ^ set(frames)

    def test_the_designed_session_passes_with_no_error(self, frames) -> None:
        report = validate_tables(frames, SESSION_CONTRACTS)
        assert report.ok, [str(f) for f in report.errors]

    def test_contracts_only_reference_columns_that_exist(self, frames) -> None:
        report = validate_tables(frames, SESSION_CONTRACTS)
        inapplicable = [f for f in report.findings if "not applicable" in f.rule]
        assert not inapplicable, [f"{f.table}: {f.detail}" for f in inapplicable]

    def test_every_contract_has_a_key_and_rules(self) -> None:
        for name, contract in CONTRACTS.items():
            assert contract.key, name
            assert contract.rules, name
            assert contract.name == name

    def test_optional_tables_are_a_subset_of_the_registry(self) -> None:
        assert OPTIONAL_TABLES <= set(CONTRACTS)


class TestCorruptionMatrix:
    """Criterion 3: eleven defects, each caught by the rule that owns it."""

    def test_duplicate_business_key(self, frames) -> None:
        damaged = corrupt(frames, "grid", lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True))
        rules = [f.rule for f in findings_for(damaged, "grid")]
        assert "Unique" in rules

    def test_null_in_a_critical_column(self, frames) -> None:
        def blank_speed(f):
            f.loc[5, "speed"] = None
            return f

        findings = [f for f in findings_for(corrupt(frames, "grid", blank_speed), "grid") if f.rule == "NotNull"]
        assert len(findings) == 1 and findings[0].count == 1
        assert "speed" in findings[0].columns

    def test_orphan_foreign_key(self, frames) -> None:
        def relabel(f):
            f.loc[0, "driver"] = "ZZZ"
            return f

        findings = [f for f in findings_for(corrupt(frames, "delta_t", relabel), "delta_t")
                    if f.rule.startswith("ForeignKey")]
        assert len(findings) == 1 and findings[0].count == 1

    def test_speed_beyond_the_envelope(self, frames) -> None:
        def implausible(f):
            f.loc[3, "speed"] = 450.0
            return f

        findings = [f for f in findings_for(corrupt(frames, "grid", implausible), "grid")
                    if f.rule.startswith("Range") and f.columns == ("speed",)]
        assert len(findings) == 1 and findings[0].count == 1

    def test_undocumented_drs_code(self, frames) -> None:
        def odd_code(f):
            f.loc[2, "drs"] = 7
            return f

        findings = [f for f in findings_for(corrupt(frames, "grid", odd_code), "grid") if f.rule == "AllowedValues"]
        assert len(findings) == 1 and "7" in findings[0].detail

    def test_impossible_gear(self, frames) -> None:
        def ninth_gear(f):
            f.loc[1, "n_gear"] = 9
            return f

        findings = [f for f in findings_for(corrupt(frames, "grid", ninth_gear), "grid")
                    if f.rule.startswith("Range") and f.columns == ("n_gear",)]
        assert len(findings) == 1

    def test_distance_no_longer_matches_the_index(self, frames) -> None:
        def shift(f):
            f.loc[4, "distance_m"] = 999.0
            return f

        rules = [f.rule for f in findings_for(corrupt(frames, "grid", shift), "grid")]
        assert "Invariant(grid_distance_matches_index)" in rules

    def test_time_going_backwards_along_a_lap(self, frames) -> None:
        def stall(f):
            f.loc[6, "elapsed_time"] = f.loc[5, "elapsed_time"]
            return f

        findings = [f for f in findings_for(corrupt(frames, "grid", stall), "grid")
                    if f.rule == "Invariant(elapsed_time_increasing_per_lap)"]
        assert len(findings) == 1
        # The whole lap is flagged, so the samples point at a lap worth opening.
        assert findings[0].count > 1

    def test_a_gap_in_the_microsector_partition(self, frames) -> None:
        def open_a_gap(f):
            index = f.index[(f["grain"] == "corner_phase")][1]
            f.loc[index, "start_m"] = float(f.loc[index, "start_m"]) + 5.0
            return f

        rules = [f.rule for f in findings_for(corrupt(frames, "microsectors", open_a_gap), "microsectors")]
        assert "Invariant(microsectors_partition_the_lap)" in rules

    def test_delta_not_zero_at_the_line(self, frames) -> None:
        def nudge(f):
            index = f.index[f["grid_index"] == 0][0]
            f.loc[index, "delta_t_s"] = 0.4
            return f

        findings = [f for f in findings_for(corrupt(frames, "delta_t", nudge), "delta_t")
                    if f.rule == "Invariant(delta_is_zero_at_the_line)"]
        assert len(findings) == 1 and findings[0].count == 1

    def test_a_table_nobody_contracted(self, frames) -> None:
        damaged = dict(frames)
        damaged["surprise"] = pd.DataFrame({"x": [1, 2, 3]})
        findings = [f for f in validate_tables(damaged, SESSION_CONTRACTS).findings if f.table == "surprise"]
        assert len(findings) == 1 and findings[0].rule == "NoContract" and findings[0].is_error


class TestPermittedNulls:
    """Criterion 2: the intended nulls are silent, and only they."""

    def test_lift_only_events_carry_no_braking_point(self, frames) -> None:
        events = frames["events"]
        assert events["brake_on_m"].isna().any(), "the designed session needs a lift-only event"
        assert not [f for f in findings_for(frames, "events") if f.rule.startswith("NotNull")]

    def test_a_braked_event_missing_its_braking_point_is_a_warning(self, frames) -> None:
        def erase(f):
            braked = frames["events"].loc[frames["events"]["has_braking"].fillna(False).astype(bool), "event_id"]
            index = f.index[f["event_id"] == int(braked.iloc[0])][0]
            f.loc[index, ["brake_on_m", "brake_dev_m"]] = None
            return f

        findings = [f for f in findings_for(corrupt(frames, "corner_metrics", erase), "corner_metrics")
                    if f.rule.startswith("NotNull")]
        assert len(findings) == 1
        assert findings[0].severity == "warning" and findings[0].count == 1

    def test_straights_and_bins_belong_to_no_event(self, frames) -> None:
        sectors = frames["microsectors"]
        assert sectors["event_id"].isna().any()
        assert not [f for f in findings_for(frames, "microsectors") if f.rule.startswith("NotNull")]

    def test_the_reference_lap_has_no_closure_residual(self, frames) -> None:
        truth = frames["ground_truth"]
        assert truth["closure_residual_s"].isna().sum() == 1
        assert not [f for f in findings_for(frames, "ground_truth") if f.rule.startswith("NotNull")]

    def test_a_null_where_nothing_permits_it_is_still_an_error(self, frames) -> None:
        def blank_lap_time(f):
            accurate = f.index[f["is_accurate"].fillna(False).astype(bool)][0]
            f.loc[accurate, "lap_time"] = None
            return f

        findings = [f for f in findings_for(corrupt(frames, "laps", blank_lap_time), "laps")
                    if f.rule.startswith("NotNull")]
        assert len(findings) == 1 and findings[0].is_error
