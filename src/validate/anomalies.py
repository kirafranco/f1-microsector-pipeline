"""Per-lap anomalies in the reconstructed timing, flagged rather than averaged in.

F022. Two kinds of lap distort a session's statistics without saying anything
about the alignment, and both are properties of individual records:

**The sector split falls elsewhere.** For 131 laps of the 2024 season (0.50 %),
two adjacent sector residuals are large and cancel while the lap residual holds
-- Sainz at Austin reads S1 +1.098 s, S2 -1.311 s, lap -0.401. For every one of
them the official S1 + S2 + S3 equals the official lap time to 0.000 s, so the
official sectors are self-consistent and the pipeline's lap is right: what
differs is *where* the source put that lap's split, against the session-median
boundary the pipeline uses. Averaged into a session's spread they triple it --
Austin's S1 spread is 0.144 s with them and 0.055 s without, which is Bahrain's.

**The lap itself is far out.** A residual beyond 0.30 s is three times the worst
per-crossing registration floor, twice over. 70 laps of the season.

Global CLAUDE.md 3.1: a corrupt or anomalous record is logged and skipped, the
batch does not stop. F010 already does this with coverage-poor laps and F019
with distance excursions; these join them. What stays gated is their *fraction*,
because a pipeline broken for every lap must still fail.
"""

from __future__ import annotations

import pandas as pd

#: A residual this far out is not registration noise. Three times the worst
#: per-crossing floor F010 measured (~0.09 s), twice over.
ANOMALY_S = 0.30

#: A cancelling pair has to cancel: the two residuals sum to less than half the
#: larger one. A genuinely slow sector followed by a genuinely slow next sector
#: adds up instead, and is not flagged.
CANCEL_FRACTION = 0.5

SECTOR_RESIDUALS = ("s1_residual_s", "s2_residual_s", "s3_residual_s")


def _cancels(first: pd.Series, second: pd.Series) -> pd.Series:
    """``first`` is large and ``second`` gives most of it back."""
    return (first.abs() > ANOMALY_S) & ((first + second).abs() < CANCEL_FRACTION * first.abs())


def sector_split_outliers(frame: pd.DataFrame) -> pd.Series:
    """Laps whose adjacent sector residuals cancel: the source split them elsewhere.

    Checked in both directions on both adjacent pairs, so it does not matter
    which of the two sectors carries the surplus. S1 and S3 are not a pair --
    they are adjacent only across the timing line, where a cancelling pair would
    be a lap-length error, which the lap residual already catches.
    """
    r1, r2, r3 = (frame[name].astype(float) for name in SECTOR_RESIDUALS)
    flagged = _cancels(r1, r2) | _cancels(r2, r1) | _cancels(r2, r3) | _cancels(r3, r2)
    return flagged.fillna(False).astype(bool)


def lap_outliers(frame: pd.DataFrame) -> pd.Series:
    """Laps whose own residual is beyond what registration noise can produce."""
    return (frame["lap_residual_s"].astype(float).abs() > ANOMALY_S).fillna(False).astype(bool)
