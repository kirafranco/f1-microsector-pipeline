"""Validation rules and the findings they produce.

Every rule answers one question about one frame and returns the mask of rows
that violate it. Nothing here reads or writes files, and no rule knows which
table it is attached to -- the engine supplies that when it builds a finding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

ERROR = "error"
WARNING = "warning"
SEVERITIES = (ERROR, WARNING)

#: Keys carried on a finding so a loader can quarantine exactly those rows.
MAX_SAMPLES = 20

#: A predicate over (frame, parents) marking rows where something is permitted.
Predicate = Callable[[pd.DataFrame, Mapping[str, pd.DataFrame]], "pd.Series | np.ndarray"]


class ContractError(ValueError):
    """A rule cannot be applied to the frame it was given."""


@dataclass(frozen=True)
class Finding:
    """One rule's verdict on one table."""

    table: str
    rule: str
    columns: tuple[str, ...]
    severity: str
    count: int
    fraction: float
    samples: tuple = ()
    detail: str = ""

    @property
    def is_error(self) -> bool:
        return self.severity == ERROR

    def to_dict(self) -> dict:
        return {
            "table": self.table,
            "rule": self.rule,
            "columns": ",".join(self.columns),
            "severity": self.severity,
            "count": int(self.count),
            "fraction": float(self.fraction),
            "samples": json.dumps([list(s) if isinstance(s, tuple) else s for s in self.samples], default=str),
            "detail": self.detail,
        }

    def __str__(self) -> str:
        return (
            f"table={self.table} rule={self.rule} columns={','.join(self.columns)} "
            f"severity={self.severity} count={self.count} fraction={self.fraction:.6f}"
            + (f" detail={self.detail}" if self.detail else "")
        )


def _mask(values, index: pd.Index) -> pd.Series:
    """Normalise a predicate's return value to a boolean Series on ``index``."""
    if isinstance(values, pd.Series):
        return values.reindex(index).fillna(False).astype(bool)
    array = np.asarray(values)
    if array.shape != (len(index),):
        raise ContractError(f"predicate returned shape {array.shape}, expected ({len(index)},)")
    return pd.Series(array.astype(bool), index=index)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ContractError(f"missing columns {missing}")


@dataclass(frozen=True)
class Rule:
    """Base: a named check returning the mask of violating rows."""

    severity: str = ERROR

    @property
    def label(self) -> str:
        return type(self).__name__

    @property
    def columns(self) -> tuple[str, ...]:
        return ()

    def violations(self, frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
        raise NotImplementedError

    def detail(self, frame: pd.DataFrame) -> str:
        return ""


@dataclass(frozen=True)
class NotNull(Rule):
    """Critical columns are complete.

    ``unless`` names the rows where a null is legitimate -- an inaccurate lap
    has no lap time, a corner without braking has no braking point -- so the
    rule stays strict everywhere else. ``max_fraction`` tolerates a documented
    sliver whose rows cannot be identified from the frame alone.
    """

    check_columns: tuple[str, ...] = ()
    unless: Predicate | None = None
    max_fraction: float = 0.0

    @property
    def label(self) -> str:
        suffix = f" unless {self.unless.__name__}" if self.unless is not None else ""
        return f"NotNull{suffix}"

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.check_columns)

    def violations(self, frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
        _require_columns(frame, self.check_columns)
        missing = frame[list(self.check_columns)].isna().any(axis=1)
        if self.unless is not None:
            missing &= ~_mask(self.unless(frame, parents), frame.index)
        if self.max_fraction > 0 and len(frame):
            if float(missing.sum()) / len(frame) <= self.max_fraction:
                return pd.Series(False, index=frame.index)
        return missing

    def detail(self, frame: pd.DataFrame) -> str:
        return f"max_fraction={self.max_fraction}" if self.max_fraction else ""


@dataclass(frozen=True)
class Unique(Rule):
    """One row per business key. Every row of a duplicated group is reported."""

    key: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return "Unique"

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.key)

    def violations(self, frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
        _require_columns(frame, self.key)
        return frame.duplicated(list(self.key), keep=False)


@dataclass(frozen=True)
class ForeignKey(Rule):
    """Every key exists in the parent table.

    ``nullable`` exempts rows whose key is null -- a micro-sector that belongs
    to no event, a corner that labels none.
    """

    key: tuple[str, ...] = ()
    parent: str = ""
    parent_key: tuple[str, ...] | None = None
    nullable: bool = False

    @property
    def label(self) -> str:
        return f"ForeignKey -> {self.parent}"

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.key)

    def violations(self, frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
        _require_columns(frame, self.key)
        if self.parent not in parents:
            raise ContractError(f"parent table {self.parent!r} was not supplied")
        parent_frame = parents[self.parent]
        parent_key = list(self.parent_key or self.key)
        _require_columns(parent_frame, parent_key)

        child = _key_frame(frame, self.key)
        known = set(_key_tuples(parent_frame, parent_key))
        present = pd.Series([tuple(row) in known for row in child], index=frame.index)
        null_key = frame[list(self.key)].isna().any(axis=1)
        if self.nullable:
            return ~present & ~null_key
        return ~present | null_key


@dataclass(frozen=True)
class Range(Rule):
    """Physical bounds, inclusive. Nulls are NotNull's business, not this rule's.

    ``max_fraction`` tolerates a documented sliver of corrupt samples without
    losing the check. The source is a live timing feed and it does produce
    nonsense occasionally -- 49 samples of one 2024 race report a gear of 72 --
    and global CLAUDE.md 3.1 is explicit that a corrupt record is logged and
    skipped rather than stopping the batch. A rule that fails the whole session
    over 0.02 % of its rows enforces the opposite. Set the fraction low enough
    that a channel which has genuinely broken still fails.
    """

    column: str = ""
    low: float = -np.inf
    high: float = np.inf
    max_fraction: float = 0.0

    @property
    def label(self) -> str:
        return f"Range[{self.low}, {self.high}]"

    @property
    def columns(self) -> tuple[str, ...]:
        return (self.column,)

    def violations(self, frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
        _require_columns(frame, [self.column])
        values = pd.to_numeric(frame[self.column], errors="coerce")
        outside = ((values < self.low) | (values > self.high)).fillna(False).astype(bool)
        if self.max_fraction > 0 and len(frame):
            if float(outside.sum()) / len(frame) <= self.max_fraction:
                return pd.Series(False, index=frame.index)
        return outside

    def detail(self, frame: pd.DataFrame) -> str:
        return f"max_fraction={self.max_fraction}" if self.max_fraction else ""


@dataclass(frozen=True)
class AllowedValues(Rule):
    """A discrete channel takes only the values its source defines."""

    column: str = ""
    values: tuple = ()

    @property
    def label(self) -> str:
        return "AllowedValues"

    @property
    def columns(self) -> tuple[str, ...]:
        return (self.column,)

    def violations(self, frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
        _require_columns(frame, [self.column])
        column = frame[self.column]
        return (~column.isin(list(self.values)) & column.notna()).astype(bool)

    def detail(self, frame: pd.DataFrame) -> str:
        if self.column not in frame.columns:
            return ""
        unexpected = sorted({v for v in frame[self.column].dropna().unique() if v not in self.values}, key=str)
        return f"unexpected={unexpected[:10]}" if unexpected else ""


@dataclass(frozen=True)
class Invariant(Rule):
    """A structural property established by an earlier feature.

    ``check`` returns the mask of violating rows; when a property is about a
    whole lap or a whole grain, it marks every row of the offending group so
    the samples point somewhere useful.
    """

    name: str = ""
    check: Predicate | None = None
    about: tuple[str, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        return f"Invariant({self.name})"

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.about)

    def violations(self, frame: pd.DataFrame, parents: Mapping[str, pd.DataFrame]) -> pd.Series:
        if self.check is None:
            raise ContractError(f"invariant {self.name!r} has no check")
        _require_columns(frame, self.about)
        return _mask(self.check(frame, parents), frame.index)


def _key_frame(frame: pd.DataFrame, key: Sequence[str]) -> list[tuple]:
    return list(_key_tuples(frame, key))


def _key_tuples(frame: pd.DataFrame, key: Sequence[str]):
    """Key tuples with pandas scalars normalised, so joins across dtypes work."""
    columns = [frame[c] for c in key]
    normalised = []
    for column in columns:
        if pd.api.types.is_float_dtype(column):
            normalised.append(column.astype(float))
        elif pd.api.types.is_integer_dtype(column) or pd.api.types.is_bool_dtype(column):
            normalised.append(column.astype("Int64"))
        else:
            normalised.append(column.astype("string"))
    values = zip(*(c.tolist() for c in normalised))
    return (tuple(None if pd.isna(v) else v for v in row) for row in values)


def samples_for(frame: pd.DataFrame, mask: pd.Series, key: Sequence[str]) -> tuple:
    """Up to MAX_SAMPLES key tuples of violating rows, for quarantine or triage."""
    offending = frame.loc[mask]
    if offending.empty:
        return ()
    usable = [c for c in key if c in offending.columns]
    if not usable:
        return tuple(offending.index[:MAX_SAMPLES].tolist())
    return tuple(list(_key_tuples(offending.head(MAX_SAMPLES), usable)))
