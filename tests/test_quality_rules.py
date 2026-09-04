"""F011 rules: each one on a small frame with a designed violation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.quality.rules import (
    ERROR,
    WARNING,
    AllowedValues,
    ContractError,
    Finding,
    ForeignKey,
    Invariant,
    NotNull,
    Range,
    Unique,
    samples_for,
)

KEY = ("driver", "lap_number")


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver": pd.array(["AAA", "AAA", "BBB", "BBB", "CCC"], dtype="string"),
            "lap_number": pd.array([1, 2, 1, 2, 1], dtype="Int16"),
            "speed": pd.array([100.0, 200.0, None, 300.0, 250.0], dtype="float32"),
            "gear": pd.array([3, 4, 5, 6, 7], dtype="Int8"),
            "ok": pd.array([True, True, False, True, True], dtype="boolean"),
        }
    )


def permitted(f: pd.DataFrame, parents) -> pd.Series:
    """Rows where a null is legitimate: this fixture's 'not ok' row."""
    return ~f["ok"].fillna(False).astype(bool)


class TestNotNull:
    def test_reports_the_null_row(self) -> None:
        rule = NotNull(check_columns=("speed",))
        mask = rule.violations(frame(), {})
        assert mask.tolist() == [False, False, True, False, False]
        assert rule.columns == ("speed",) and rule.severity == ERROR

    def test_unless_permits_it_and_names_the_reason(self) -> None:
        rule = NotNull(check_columns=("speed",), unless=permitted)
        assert not rule.violations(frame(), {}).any()
        assert rule.label == "NotNull unless permitted"

    def test_unless_does_not_permit_a_null_elsewhere(self) -> None:
        f = frame()
        f.loc[3, "speed"] = None  # a row where 'ok' is True
        assert NotNull(check_columns=("speed",), unless=permitted).violations(f, {}).tolist() == [
            False, False, False, True, False
        ]

    def test_max_fraction_tolerates_a_documented_sliver(self) -> None:
        rule = NotNull(check_columns=("speed",), max_fraction=0.25)
        assert not rule.violations(frame(), {}).any()  # 1 of 5 = 0.20
        assert NotNull(check_columns=("speed",), max_fraction=0.1).violations(frame(), {}).any()

    def test_several_columns_are_checked_together(self) -> None:
        f = frame()
        f.loc[0, "gear"] = None
        mask = NotNull(check_columns=("speed", "gear")).violations(f, {})
        assert mask.tolist() == [True, False, True, False, False]

    def test_missing_column_is_a_contract_error(self) -> None:
        with pytest.raises(ContractError, match="missing columns"):
            NotNull(check_columns=("nope",)).violations(frame(), {})


class TestUnique:
    def test_clean_key_passes(self) -> None:
        assert not Unique(key=KEY).violations(frame(), {}).any()

    def test_every_row_of_a_duplicate_group_is_reported(self) -> None:
        f = pd.concat([frame(), frame().iloc[[0]]], ignore_index=True)
        mask = Unique(key=KEY).violations(f, {})
        assert mask.sum() == 2
        assert mask.tolist() == [True, False, False, False, False, True]

    def test_single_column_key(self) -> None:
        assert Unique(key=("driver",)).violations(frame(), {}).sum() == 4


class TestForeignKey:
    def parent(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"driver": pd.array(["AAA", "BBB"], dtype="string"), "lap_number": pd.array([1, 2], dtype="Int16")}
        )

    def test_orphans_are_found(self) -> None:
        rule = ForeignKey(key=KEY, parent="laps")
        mask = rule.violations(frame(), {"laps": self.parent()})
        assert mask.tolist() == [False, True, True, False, True]
        assert rule.label == "ForeignKey -> laps"

    def test_dtype_mismatch_does_not_create_false_orphans(self) -> None:
        """Int16 child against int64 parent must still match."""
        parent = self.parent().astype({"lap_number": "int64", "driver": "object"})
        mask = ForeignKey(key=KEY, parent="laps").violations(frame(), {"laps": parent})
        assert mask.tolist() == [False, True, True, False, True]

    def test_parent_key_can_be_named_separately(self) -> None:
        parent = self.parent().rename(columns={"driver": "d", "lap_number": "n"})
        mask = ForeignKey(key=KEY, parent="laps", parent_key=("d", "n")).violations(frame(), {"laps": parent})
        assert mask.sum() == 3

    def test_nullable_exempts_null_keys(self) -> None:
        f = frame()
        f["event_id"] = pd.array([1, None, 1, None, 9], dtype="Int16")
        parent = pd.DataFrame({"event_id": pd.array([1], dtype="Int16")})
        strict = ForeignKey(key=("event_id",), parent="events").violations(f, {"events": parent})
        lenient = ForeignKey(key=("event_id",), parent="events", nullable=True).violations(f, {"events": parent})
        assert strict.tolist() == [False, True, False, True, True]
        assert lenient.tolist() == [False, False, False, False, True]

    def test_missing_parent_is_a_contract_error(self) -> None:
        with pytest.raises(ContractError, match="was not supplied"):
            ForeignKey(key=KEY, parent="laps").violations(frame(), {})


class TestRangeAndAllowedValues:
    def test_bounds_are_inclusive(self) -> None:
        assert not Range(column="speed", low=100.0, high=300.0).violations(frame(), {}).any()
        assert Range(column="speed", low=101.0, high=300.0).violations(frame(), {}).sum() == 1

    def test_nulls_are_not_this_rules_business(self) -> None:
        mask = Range(column="speed", low=0.0, high=1.0).violations(frame(), {})
        assert mask.tolist() == [True, True, False, True, True]  # the null row is left to NotNull

    def test_allowed_values_and_its_detail(self) -> None:
        rule = AllowedValues(column="gear", values=(3, 4, 5, 6))
        mask = rule.violations(frame(), {})
        assert mask.tolist() == [False, False, False, False, True]
        assert "7" in rule.detail(frame())

    def test_allowed_values_ignores_nulls(self) -> None:
        f = frame()
        f.loc[0, "gear"] = None
        assert not AllowedValues(column="gear", values=(4, 5, 6, 7)).violations(f, {}).iloc[0]


class TestInvariant:
    def test_named_check_and_label(self) -> None:
        def speed_exceeds_forty_times_gear(f, parents):
            return f["speed"].fillna(0).astype(float) > f["gear"].astype(float) * 40.0

        rule = Invariant(name="speed_vs_gear", check=speed_exceeds_forty_times_gear, about=("speed", "gear"))
        assert rule.label == "Invariant(speed_vs_gear)"
        assert rule.violations(frame(), {}).tolist() == [False, True, False, True, False]

    def test_a_predicate_returning_a_bare_array_is_accepted(self) -> None:
        rule = Invariant(name="none", check=lambda f, parents: np.zeros(len(f), dtype=bool))
        assert not rule.violations(frame(), {}).any()

    def test_wrong_shape_is_a_contract_error(self) -> None:
        rule = Invariant(name="bad", check=lambda f, parents: np.zeros(3, dtype=bool))
        with pytest.raises(ContractError, match="shape"):
            rule.violations(frame(), {})

    def test_missing_check_is_a_contract_error(self) -> None:
        with pytest.raises(ContractError, match="no check"):
            Invariant(name="empty").violations(frame(), {})


class TestSamplesAndFindings:
    def test_samples_carry_the_business_key(self) -> None:
        mask = NotNull(check_columns=("speed",)).violations(frame(), {})
        assert samples_for(frame(), mask, KEY) == (("BBB", 1),)

    def test_samples_are_capped(self) -> None:
        big = pd.DataFrame({"driver": ["A"] * 50, "lap_number": range(50), "speed": [None] * 50})
        mask = NotNull(check_columns=("speed",)).violations(big, {})
        assert len(samples_for(big, mask, KEY)) == 20

    def test_finding_serialises_and_reads_back(self) -> None:
        import json

        finding = Finding("grid", "NotNull", ("speed",), WARNING, 3, 0.5, samples=(("AAA", 1),), detail="x")
        payload = finding.to_dict()
        assert payload["columns"] == "speed" and payload["severity"] == WARNING
        assert json.loads(payload["samples"]) == [["AAA", 1]]
        assert not finding.is_error
        assert "table=grid" in str(finding) and "count=3" in str(finding)
