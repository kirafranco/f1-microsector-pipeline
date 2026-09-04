"""Applying contracts, and the gate F005 calls before a load."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from src.quality.rules import ERROR, WARNING, ContractError, Finding, Rule, samples_for

logger = logging.getLogger(__name__)


class QualityGateError(RuntimeError):
    """The gate refused: at least one error finding stands."""

    def __init__(self, findings: Sequence[Finding]) -> None:
        self.findings = tuple(findings)
        listed = "; ".join(str(f) for f in self.findings[:10])
        more = f" (+{len(self.findings) - 10} more)" if len(self.findings) > 10 else ""
        super().__init__(f"{len(self.findings)} error finding(s): {listed}{more}")


@dataclass(frozen=True)
class TableContract:
    """The rules one artefact must satisfy before it is fit to load."""

    name: str
    key: tuple[str, ...]
    rules: tuple[Rule, ...]
    parents: tuple[str, ...] = ()

    @property
    def signature(self) -> str:
        parts = [self.name, "|".join(self.key), "|".join(self.parents)]
        parts += [f"{r.label}:{','.join(r.columns)}:{r.severity}" for r in self.rules]
        return ";".join(parts)


@dataclass(frozen=True)
class TableReport:
    table: str
    rows: int
    rules_run: int
    findings: tuple[Finding, ...]

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.is_error)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if not f.is_error)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "table": self.table,
            "rows": self.rows,
            "rules_run": self.rules_run,
            "ok": self.ok,
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass(frozen=True)
class QualityReport:
    tables: tuple[TableReport, ...]
    contract_version: str
    elapsed_s: float = 0.0

    @property
    def findings(self) -> tuple[Finding, ...]:
        return tuple(f for t in self.tables for f in t.findings)

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.is_error)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if not f.is_error)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "contract_version": self.contract_version,
            "elapsed_s": self.elapsed_s,
            "tables": [t.to_dict() for t in self.tables],
            "counts": {"findings": len(self.findings), "errors": len(self.errors), "warnings": len(self.warnings)},
        }


def contract_version(contracts: Mapping[str, TableContract]) -> str:
    """Stable hash of the rule set, so a report can be tied to what produced it."""
    joined = "\n".join(contracts[name].signature for name in sorted(contracts))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def _log(finding: Finding) -> None:
    logger.log(logging.ERROR if finding.is_error else logging.WARNING, "quality_finding %s", finding)


def validate_table(
    frame: pd.DataFrame, contract: TableContract, parents: Mapping[str, pd.DataFrame] | None = None
) -> TableReport:
    """Apply every rule of one contract.

    Every rule runs even after one fails, and a rule that cannot be applied at
    all becomes a finding of its own rather than an exception -- a contract
    that has drifted from its table must be visible, not fatal.
    """
    parents = dict(parents or {})
    findings: list[Finding] = []
    rows = len(frame)

    for rule in contract.rules:
        try:
            mask = rule.violations(frame, parents)
        except ContractError as exc:
            findings.append(
                Finding(contract.name, f"{rule.label} [not applicable]", rule.columns, ERROR, rows, 1.0,
                        detail=str(exc))
            )
            continue
        count = int(mask.sum())
        if not count:
            continue
        findings.append(
            Finding(
                table=contract.name,
                rule=rule.label,
                columns=rule.columns,
                severity=rule.severity,
                count=count,
                fraction=count / rows if rows else 0.0,
                samples=samples_for(frame, mask, contract.key),
                detail=rule.detail(frame),
            )
        )

    for finding in findings:
        _log(finding)
    return TableReport(contract.name, rows, len(contract.rules), tuple(findings))


def validate_tables(
    frames: Mapping[str, pd.DataFrame], contracts: Mapping[str, TableContract]
) -> QualityReport:
    """Validate every supplied frame against the registry.

    A frame with no contract is an error: nothing loads on the strength of no
    rule having said otherwise. A contract with no frame is a warning -- the
    artefact is optional or was not produced for this session.
    """
    reports: list[TableReport] = []

    for name in sorted(frames):
        if name in contracts:
            continue
        finding = Finding(name, "NoContract", (), ERROR, len(frames[name]), 1.0,
                          detail="table presented for validation has no contract")
        _log(finding)
        reports.append(TableReport(name, len(frames[name]), 0, (finding,)))

    for name in sorted(contracts):
        contract = contracts[name]
        if name not in frames:
            finding = Finding(name, "MissingTable", (), WARNING, 0, 0.0, detail="no frame supplied for this contract")
            _log(finding)
            reports.append(TableReport(name, 0, 0, (finding,)))
            continue
        parents = {p: frames[p] for p in contract.parents if p in frames}
        reports.append(validate_table(frames[name], contract, parents))

    return QualityReport(tuple(reports), contract_version(contracts))


def gate(report: QualityReport) -> bool:
    """True when nothing blocks a load."""
    return report.ok


def require(report: QualityReport) -> None:
    """Raise unless the report is clean. What F005 calls before writing."""
    if not report.ok:
        raise QualityGateError(report.errors)


def findings_frame(report: QualityReport) -> pd.DataFrame:
    """Every finding as a row, for the parquet artefact."""
    rows = [f.to_dict() for f in report.findings]
    frame = pd.DataFrame(rows, columns=["table", "rule", "columns", "severity", "count", "fraction", "samples", "detail"])
    return frame.astype(
        {"table": "string", "rule": "string", "columns": "string", "severity": "string",
         "count": "Int64", "fraction": "float32", "samples": "string", "detail": "string"}
    )
