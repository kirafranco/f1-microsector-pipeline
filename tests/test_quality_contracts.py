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
from src.config import DATA_ROOT
from src.quality.contracts import CONTRACTS, OPTIONAL_TABLES, SESSION_TABLES
from src.quality.rules import Invariant, NotNull
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

    def test_one_implausible_speed_is_tolerated(self, frames) -> None:
        """Per-sample channel bounds carry a glitch tolerance: the feed does
        emit occasional nonsense, and a race must not fail over a handful of
        samples (global CLAUDE.md 3.1)."""
        def implausible(f):
            f.loc[3, "speed"] = 450.0
            return f

        findings = [f for f in findings_for(corrupt(frames, "grid", implausible), "grid")
                    if f.rule.startswith("Range") and f.columns == ("speed",)]
        assert findings == []

    def test_a_broken_speed_channel_still_fails(self, frames) -> None:
        def implausible(f):
            f["speed"] = 450.0
            return f

        findings = [f for f in findings_for(corrupt(frames, "grid", implausible), "grid")
                    if f.rule.startswith("Range") and f.columns == ("speed",)]
        assert len(findings) == 1 and findings[0].is_error

    def test_the_tolerance_does_not_reach_structural_rules(self) -> None:
        """Uniqueness, keys, invariants and completeness stay absolute: those
        failing means the pipeline is wrong, not that the feed hiccupped."""
        from src.quality.contracts import CONTRACTS
        from src.quality.rules import ForeignKey, Invariant, Range, Unique

        for name, contract in CONTRACTS.items():
            for rule in contract.rules:
                if isinstance(rule, (Unique, ForeignKey, Invariant)):
                    assert getattr(rule, "max_fraction", 0.0) == 0.0, f"{name}: {rule.label}"

    def test_a_drs_value_that_is_not_a_status_byte(self, frames) -> None:
        """The check is that the channel is DRS at all. 7 is an unused code but
        a plausible byte; 100 is not the channel this contract describes."""
        def not_a_byte(f):
            f.loc[2, "drs"] = 100
            return f

        findings = [f for f in findings_for(corrupt(frames, "grid", not_a_byte), "grid")
                    if f.rule == "AllowedValues"]
        assert len(findings) == 1 and "100" in findings[0].detail

    def test_an_unused_but_plausible_drs_code_passes(self, frames) -> None:
        """Enumerating the codes one session happened to show was the bug: 2,
        3, 11, 13 and 15 each blocked a race until they were added (F015)."""
        def unused_code(f):
            f.loc[2, "drs"] = 7
            return f

        findings = [f for f in findings_for(corrupt(frames, "grid", unused_code), "grid")
                    if f.rule == "AllowedValues"]
        assert findings == []

    def test_one_impossible_gear_is_tolerated(self, frames) -> None:
        """The timing feed does emit the occasional nonsense sample -- 49 of
        ~300k in the 2024 Miami Grand Prix report a gear of 72 -- and global
        CLAUDE.md 3.1 says a corrupt record is skipped, not that the batch
        stops. Failing a whole race over 0.02 % of its rows does the opposite."""
        def ninth_gear(f):
            f.loc[1, "n_gear"] = 9
            return f

        findings = [f for f in findings_for(corrupt(frames, "grid", ninth_gear), "grid")
                    if f.rule.startswith("Range") and f.columns == ("n_gear",)]
        assert findings == []

    def test_a_broken_gear_channel_still_fails(self, frames) -> None:
        """The tolerance is a sliver, not an amnesty."""
        def every_gear(f):
            f["n_gear"] = 9
            return f

        findings = [f for f in findings_for(corrupt(frames, "grid", every_gear), "grid")
                    if f.rule.startswith("Range") and f.columns == ("n_gear",)]
        assert len(findings) == 1
        assert findings[0].is_error

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


class TestF018LeavesTheGateAlone:
    """F018 fixed the producer of `lift_m`, not the contract that caught it.

    The invariant that refused six sessions of the 2024 season is correct and
    stays a structural rule: absolute, and never softened by a glitch fraction.
    """

    def test_the_ordering_invariant_is_still_gated(self) -> None:
        invariants = [r for r in CONTRACTS["events"].rules if isinstance(r, Invariant)]
        assert [r.name for r in invariants] == ["event_boundaries_are_ordered"]
        assert all(getattr(r, "max_fraction", 0.0) == 0.0 for r in invariants)

    def test_the_microsector_label_rule_is_still_gated(self) -> None:
        rules = [r for r in CONTRACTS["microsectors"].rules if isinstance(r, NotNull)]
        assert any(set(r.check_columns) == {"event_id", "corners"} and r.unless is not None for r in rules)


class TestF021RedFlagLaps:
    """F021: the lap-time envelope is a unit check, judged on accurate laps.

    Lap 1 of the red-flagged 2024 Monaco Grand Prix reports 2,456-2,526 s for
    16 of 20 drivers. Those laps are real, in the right unit, marked
    `is_accurate=False`, and dropped before the grid -- but they refused the
    whole session, the last gate refusal of the 2024 season.
    """

    RED_FLAG_S = 2456.33

    def set_lap_time(self, seconds: float, *, accurate: bool):
        def mutate(f):
            row = f.index[f["is_accurate"].fillna(False).astype(bool)][0]
            f.loc[row, "lap_time"] = seconds
            f.loc[row, "is_accurate"] = accurate
            return f
        return mutate

    def range_findings(self, frames, seconds: float, *, accurate: bool):
        damaged = corrupt(frames, "laps", self.set_lap_time(seconds, accurate=accurate))
        return [f for f in findings_for(damaged, "laps")
                if f.rule.startswith("Range") and f.columns == ("lap_time",)]

    def test_a_red_flag_lap_time_on_an_inaccurate_lap_passes(self, frames) -> None:
        assert self.range_findings(frames, self.RED_FLAG_S, accurate=False) == []

    def test_the_same_time_on_an_accurate_lap_still_fails(self, frames) -> None:
        findings = self.range_findings(frames, self.RED_FLAG_S, accurate=True)
        assert len(findings) == 1 and findings[0].is_error

    @pytest.mark.parametrize("seconds", [1.47, 88197.0])
    def test_a_unit_error_on_an_accurate_lap_is_what_the_envelope_is_for(self, frames, seconds: float) -> None:
        """Minutes and milliseconds: the two cases the contract's comment names."""
        findings = self.range_findings(frames, seconds, accurate=True)
        assert len(findings) == 1 and findings[0].is_error

    @pytest.mark.parametrize("seconds", [1.47, 88197.0])
    def test_and_is_not_judged_on_an_inaccurate_one(self, frames, seconds: float) -> None:
        assert self.range_findings(frames, seconds, accurate=False) == []

    def test_the_exemption_reaches_the_four_timing_envelopes_and_no_further(self) -> None:
        from src.quality.rules import AllowedValues, ForeignKey, Invariant, Range, Unique

        exempt = {r.column for r in CONTRACTS["laps"].rules
                  if isinstance(r, Range) and r.unless is not None}
        assert exempt == {"lap_time", "sector1_time", "sector2_time", "sector3_time"}

        for name, contract in CONTRACTS.items():
            for rule in contract.rules:
                if isinstance(rule, (Unique, ForeignKey, Invariant, AllowedValues)):
                    assert getattr(rule, "unless", None) is None, f"{name}: {rule.label}"
                if isinstance(rule, Range) and rule.unless is not None:
                    assert name == "laps", f"{name}: {rule.label}"

    def test_identity_columns_stay_absolute(self) -> None:
        """An inaccurate lap must still have a plausible number, stint and age."""
        from src.quality.rules import Range

        absolute = {r.column for r in CONTRACTS["laps"].rules
                    if isinstance(r, Range) and r.unless is None}
        assert absolute == {"lap_number", "stint", "tyre_life"}


@pytest.mark.data
class TestEveryAccurateLapIsInsideItsEnvelopes:
    """F021 criterion 7, on every snapshot on disk.

    The exemption can only remove findings on inaccurate rows, so the way to
    know it changed no other session's verdict is to show there was nothing to
    remove: no accurate lap in the 2024 season breaches any laps-table envelope.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def snapshots() -> list[Path]:
        roots = sorted((DATA_ROOT / "raw" / "fastf1").glob("*/2024_*"))
        if not roots:
            pytest.skip("no FastF1 snapshots under data/raw/fastf1/")
        return roots

    @staticmethod
    def timing_rules():
        from src.quality.rules import Range
        return [r for r in CONTRACTS["laps"].rules if isinstance(r, Range) and r.unless is not None]

    def test_no_accurate_lap_breaches_any_envelope(self, snapshots) -> None:
        from src.quality.rules import Range

        rules = self.timing_rules()
        assert rules, "the timing envelopes should carry an exemption"
        breaches: dict[str, int] = {}
        for root in snapshots:
            frame = pd.read_parquet(root / "laps.parquet")
            accurate = frame[frame["is_accurate"].fillna(False).astype(bool)]
            for rule in rules:
                bare = Range(column=rule.column, low=rule.low, high=rule.high)
                count = int(bare.violations(accurate, {}).sum())
                if count:
                    breaches[f"{root.name} {rule.column}"] = count
        assert breaches == {}

    def test_the_exemption_leaves_nothing_flagged_anywhere(self, snapshots) -> None:
        from src.quality.rules import Range

        rules = [r for r in CONTRACTS["laps"].rules if isinstance(r, Range)]
        flagged: dict[str, int] = {}
        for root in snapshots:
            frame = pd.read_parquet(root / "laps.parquet")
            for rule in rules:
                count = int(rule.violations(frame, {}).sum())
                if count:
                    flagged[f"{root.name} {rule.column}"] = count
        assert flagged == {}

    def test_monaco_is_the_session_this_was_written_for(self, snapshots) -> None:
        """Without the exemption, exactly one session breaks, on 16 lap-1 rows."""
        from src.quality.rules import Range

        rule = next(r for r in self.timing_rules() if r.column == "lap_time")
        bare = Range(column="lap_time", low=rule.low, high=rule.high)
        broken = {}
        for root in snapshots:
            frame = pd.read_parquet(root / "laps.parquet")
            mask = bare.violations(frame, {})
            if mask.any():
                broken[root.name] = frame.loc[mask, "lap_number"].unique().tolist()
        assert list(broken) == ["2024_Monaco-Grand-Prix_R"]
        assert broken["2024_Monaco-Grand-Prix_R"] == [1]
