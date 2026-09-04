"""F011 engine: every rule runs, the gate decides once, findings are logged."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.quality.engine import (
    QualityGateError,
    TableContract,
    contract_version,
    findings_frame,
    gate,
    require,
    validate_table,
    validate_tables,
)
from src.quality.rules import ERROR, WARNING, ForeignKey, NotNull, Range, Unique

KEY = ("driver", "lap_number")


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver": pd.array(["AAA", "AAA", "BBB"], dtype="string"),
            "lap_number": pd.array([1, 2, 1], dtype="Int16"),
            "speed": pd.array([100.0, 200.0, 300.0], dtype="float32"),
        }
    )


CLEAN = TableContract(
    name="grid", key=KEY,
    rules=(NotNull(check_columns=("driver", "speed")), Unique(key=KEY), Range(column="speed", low=0.0, high=400.0)),
)


class TestValidateTable:
    def test_a_clean_table_has_no_findings(self) -> None:
        report = validate_table(frame(), CLEAN)
        assert report.ok and report.findings == () and report.rows == 3 and report.rules_run == 3

    def test_every_rule_runs_even_after_one_fails(self) -> None:
        """Criterion 4: five broken rules yield five findings, not one."""
        f = pd.concat([frame(), frame().iloc[[0]]], ignore_index=True)
        f.loc[1, "speed"] = None
        f.loc[2, "speed"] = 900.0
        # Not the duplicated row: nulling its key would dissolve the duplicate.
        f.loc[2, "driver"] = None
        contract = TableContract(
            name="grid", key=KEY,
            rules=(
                NotNull(check_columns=("speed",)), NotNull(check_columns=("driver",)), Unique(key=KEY),
                Range(column="speed", low=0.0, high=400.0), Range(column="lap_number", low=5, high=9),
            ),
        )
        report = validate_table(f, contract)
        assert len(report.findings) == 5
        assert {finding.rule for finding in report.findings} >= {"NotNull", "Unique"}

    def test_an_inapplicable_rule_becomes_a_finding_not_a_crash(self) -> None:
        contract = TableContract(name="grid", key=KEY, rules=(NotNull(check_columns=("absent",)), Unique(key=KEY)))
        report = validate_table(frame(), contract)
        assert len(report.findings) == 1
        assert "not applicable" in report.findings[0].rule
        assert "missing columns" in report.findings[0].detail

    def test_fraction_and_samples_are_populated(self) -> None:
        f = frame()
        f.loc[2, "speed"] = None
        finding = validate_table(f, CLEAN).findings[0]
        assert finding.count == 1 and finding.fraction == pytest.approx(1 / 3)
        assert finding.samples == (("BBB", 1),)

    def test_severity_travels_from_the_rule(self) -> None:
        contract = TableContract(name="grid", key=KEY,
                                 rules=(Range(column="speed", low=0.0, high=1.0, severity=WARNING),))
        report = validate_table(frame(), contract)
        assert report.ok and len(report.warnings) == 1 and not report.errors


class TestValidateTables:
    def test_parents_are_resolved_by_name(self) -> None:
        child = TableContract(
            name="grid", key=KEY, rules=(ForeignKey(key=KEY, parent="laps"),), parents=("laps",)
        )
        parent = TableContract(name="laps", key=KEY, rules=(Unique(key=KEY),))
        frames = {"grid": frame(), "laps": frame().iloc[:2]}
        report = validate_tables(frames, {"grid": child, "laps": parent})
        assert not report.ok
        orphan = [f for f in report.errors if f.table == "grid"][0]
        assert orphan.count == 1 and orphan.samples == (("BBB", 1),)

    def test_a_table_without_a_contract_is_an_error(self) -> None:
        report = validate_tables({"grid": frame(), "mystery": frame()}, {"grid": CLEAN})
        findings = [f for f in report.errors if f.table == "mystery"]
        assert len(findings) == 1 and findings[0].rule == "NoContract"
        assert not report.ok

    def test_a_contract_without_a_table_is_a_warning(self) -> None:
        report = validate_tables({"grid": frame()}, {"grid": CLEAN, "weather": CLEAN})
        missing = [f for f in report.warnings if f.rule == "MissingTable"]
        assert len(missing) == 1 and missing[0].table == "weather"
        assert report.ok

    def test_report_serialises(self) -> None:
        import json

        report = validate_tables({"grid": frame()}, {"grid": CLEAN})
        payload = report.to_dict()
        json.dumps(payload, default=str)
        assert payload["ok"] is True and payload["counts"]["findings"] == 0
        assert payload["tables"][0]["table"] == "grid"


class TestGate:
    def test_gate_passes_a_clean_report(self) -> None:
        report = validate_tables({"grid": frame()}, {"grid": CLEAN})
        assert gate(report) is True
        require(report)

    def test_gate_blocks_and_names_every_error(self) -> None:
        f = frame()
        f.loc[0, "speed"] = None
        report = validate_tables({"grid": f}, {"grid": CLEAN})
        assert gate(report) is False
        with pytest.raises(QualityGateError) as excinfo:
            require(report)
        assert "grid" in str(excinfo.value)
        assert len(excinfo.value.findings) == 1

    def test_warnings_alone_do_not_block(self) -> None:
        contract = TableContract(name="grid", key=KEY,
                                 rules=(Range(column="speed", low=0.0, high=1.0, severity=WARNING),))
        report = validate_tables({"grid": frame()}, {"grid": contract})
        assert gate(report) is True
        require(report)


class TestLoggingAndVersion:
    def test_one_log_line_per_finding_at_its_severity(self, caplog: pytest.LogCaptureFixture) -> None:
        f = frame()
        f.loc[0, "speed"] = None
        contract = TableContract(
            name="grid", key=KEY,
            rules=(NotNull(check_columns=("speed",)), Range(column="speed", low=0.0, high=1.0, severity=WARNING)),
        )
        with caplog.at_level(logging.WARNING, logger="src.quality.engine"):
            validate_table(f, contract)
        records = [r for r in caplog.records if "quality_finding" in r.message]
        assert len(records) == 2
        assert {r.levelno for r in records} == {logging.ERROR, logging.WARNING}
        assert "table=grid" in records[0].getMessage()

    def test_contract_version_is_stable_and_sensitive(self) -> None:
        first = contract_version({"grid": CLEAN})
        assert first == contract_version({"grid": CLEAN})
        changed = TableContract(name="grid", key=KEY, rules=CLEAN.rules[:2])
        assert contract_version({"grid": changed}) != first

    def test_findings_frame_schema(self) -> None:
        f = frame()
        f.loc[0, "speed"] = None
        report = validate_tables({"grid": f}, {"grid": CLEAN})
        out = findings_frame(report)
        assert list(out.columns) == ["table", "rule", "columns", "severity", "count", "fraction", "samples", "detail"]
        assert str(out["count"].dtype) == "Int64" and str(out["fraction"].dtype) == "float32"
        assert len(out) == 1

    def test_findings_frame_is_empty_but_typed_when_clean(self) -> None:
        out = findings_frame(validate_tables({"grid": frame()}, {"grid": CLEAN}))
        assert out.empty and str(out["table"].dtype) == "string"
