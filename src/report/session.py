"""Building the findings for one session and one pair of laps.

Reads the processed artefacts, runs every comparison the write-up quotes,
renders the figures, and writes one summary.json that the document is checked
against. Nothing in docs/findings/ may state a number that is not in here.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.grid.resample import GRID_SPACING_M
from src.report import crosscheck, figures, pairs, traces
from src.report.pairs import LapKey, ReportError

logger = logging.getLogger(__name__)

FIGURES_DIR = PROJECT_ROOT / "docs" / "findings" / "figures"

#: How much of the sector disagreement with official timing F010's per-lap
#: registration residuals have to explain before the reconciliation counts.
UNEXPLAINED_MAX_S = 0.005

#: Laps per driver used for the trace corroboration: each driver's quickest
#: two, which at Suzuka is the pair of Q3 runs.
QUICK_LAPS = 2

#: Padding either side of a corner event when cutting the trace window.
TRACE_PAD_M = 60.0


@dataclass(frozen=True)
class FindingsReport:
    """Every figure the document is allowed to quote."""

    session: str
    snapshot: str
    pair: str
    official_gap_s: float
    decomposed_total_s: float
    complete_sectors: int
    of_sectors: int
    max_phase_ratio: float
    max_event_ratio: float
    phases: dict
    events: dict
    focus_event: str
    focus_median_s: float
    focus_positive: int
    focus_pairings: int
    focus_sigma_s: float
    focus_window_m: list[float]
    focus_deficit: dict
    control: dict
    consistency: dict
    reconciliation: dict
    max_unexplained_s: float
    braking: dict
    dab_event: str
    dab_laps: int
    dab_of: int
    dab_absent: list[str]
    elapsed_s: float = 0.0
    figures: list[str] = field(default_factory=list)

    @property
    def nothing_decomposes(self) -> bool:
        """The headline negative: no phase and no event clears 2 sigma."""
        return self.max_phase_ratio < 2.0 and self.max_event_ratio < 2.0

    @property
    def claim_ok(self) -> bool:
        """The focus corner costs the same driver time on every pairing, and the
        raw speed agrees on every quick pairing.

        Two independent signals, each a sign test: the sector times say B is
        slower there every time, and the speed channel -- which no micro-sector
        boundary touches -- says the same.
        """
        return (self.focus_positive == self.focus_pairings
                and bool(self.focus_deficit.get("all_slower_on_average")))

    @property
    def reconciliation_ok(self) -> bool:
        return self.max_unexplained_s <= UNEXPLAINED_MAX_S

    @property
    def braking_ok(self) -> bool:
        """Every braking-point difference the stored metric reports as wider
        than the D6 window is either confirmed by the traces or explained by an
        extra brake application that the metric happened to pick up.

        This is deliberately not "the two laps braked in the same places". They
        did not, and the write-up says which ones differ.
        """
        return bool(self.braking.get("metric_outliers_explained", False))

    @property
    def ok(self) -> bool:
        return self.claim_ok and self.reconciliation_ok and self.braking_ok

    def to_dict(self) -> dict:
        out = asdict(self)
        out["checks"] = {
            "nothing_decomposes": self.nothing_decomposes,
            "claim_repeats": self.claim_ok,
            "sector_reconciliation": self.reconciliation_ok,
            "braking_within_window": self.braking_ok,
            "all": self.ok,
        }
        return out


@dataclass(frozen=True)
class FindingsResult:
    root: Path
    report: FindingsReport
    decomposition: pairs.PairDecomposition
    pairings: pd.DataFrame
    consistency: pd.DataFrame
    reconciliation: pd.DataFrame
    braking: pd.DataFrame
    trace: pd.DataFrame


def load_findings_inputs(processed_root: Path, grid_root: Path, microsector_root: Path) -> dict[str, pd.DataFrame]:
    """The artefacts the write-up needs, with a clear error when one is absent."""
    wanted = {
        "lap_summary": processed_root / "lap_summary.parquet",
        "ground_truth": processed_root / "ground_truth.parquet",
        "microsector_times": processed_root / "microsector_times.parquet",
        "microsector_summary": processed_root / "microsector_summary.parquet",
        "corner_metrics": processed_root / "corner_metrics.parquet",
        "delta_t": processed_root / "delta_t.parquet",
        "grid": grid_root / "grid.parquet",
        "microsectors": microsector_root / "microsectors.parquet",
        "events": microsector_root / "events.parquet",
    }
    missing = [name for name, path in wanted.items() if not path.exists()]
    if missing:
        raise ReportError(f"missing artefacts: {', '.join(missing)}; run the pipeline for this session first")
    return {name: pd.read_parquet(path) for name, path in wanted.items()}


def quickest(lap_summary: pd.DataFrame, driver: str, count: int = QUICK_LAPS) -> list[LapKey]:
    return [(driver, lap) for lap in pairs.timed_laps(lap_summary, driver)[:count]]


def choose_focus(consistency: pd.DataFrame) -> str:
    """The corner worth writing about: the one that points the same way every time.

    Ties, and the case where nothing is consistent, fall back to the largest
    median difference -- which the report then fails to claim, correctly.
    """
    consistent = consistency[consistency["always_same_sign"]]
    pool = consistent if not consistent.empty else consistency
    return str(pool["median_s"].abs().idxmax())


def _official_gap(lap_summary: pd.DataFrame, a: LapKey, b: LapKey) -> float:
    def lap_time(key: LapKey) -> float:
        rows = lap_summary[(lap_summary["driver"] == key[0]) & (lap_summary["lap_number"] == key[1])]
        if rows.empty:
            raise ReportError(f"no lap summary row for {key[0]} lap {key[1]}")
        return float(rows["lap_time_s"].iloc[0])
    return lap_time(b) - lap_time(a)


def _sector_boundaries(processed_root: Path) -> tuple[float, float]:
    meta_path = processed_root / "metrics_meta.json"
    if not meta_path.exists():
        raise ReportError(f"no metrics_meta.json in {processed_root}")
    boundaries = json.loads(meta_path.read_text(encoding="utf-8"))["sector_boundaries_m"]
    return float(boundaries[0]), float(boundaries[1])


def _session_identity(processed_root: Path) -> tuple[str, str]:
    meta_path = processed_root / "metrics_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    snapshot = str(meta.get("snapshot", ""))
    return processed_root.name, snapshot


def build_findings(processed_root: Path, grid_root: Path, microsector_root: Path,
                   a: LapKey, b: LapKey, *, out_root: Path | None = None,
                   figures_dir: Path | None = None, spacing_m: float = GRID_SPACING_M,
                   render: bool = True) -> FindingsResult:
    """Everything the write-up quotes, computed once and written down."""
    started = time.perf_counter()
    frames = load_findings_inputs(processed_root, grid_root, microsector_root)
    out_root = Path(out_root) if out_root else processed_root / "findings"
    figures_dir = Path(figures_dir) if figures_dir else FIGURES_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    decomposition = pairs.decompose(frames["microsector_times"], frames["microsectors"],
                                    frames["microsector_summary"], a, b)
    every_pairing = pairs.pairings(frames["microsector_times"], frames["microsectors"],
                                   frames["microsector_summary"], frames["lap_summary"], a[0], b[0])
    noise = pairs.event_noise(frames["microsectors"], frames["microsector_summary"])
    agreement = pairs.consistency(every_pairing, noise)
    focus = choose_focus(agreement)

    quick_a, quick_b = quickest(frames["lap_summary"], a[0]), quickest(frames["lap_summary"], b[0])
    control_pair = None
    if len(quick_a) >= 2:
        control_pair = pairs.control(frames["microsector_times"], frames["microsectors"],
                                     frames["microsector_summary"], a[0], quick_a[1][1], quick_a[0][1])

    event_row = frames["events"][frames["events"]["corners"].astype(str) == focus]
    if event_row.empty:
        raise ReportError(f"focus corner {focus!r} is not an event in this session")
    event = event_row.iloc[0]

    # The speed claim is measured over exactly the stretch the sector delta
    # covers -- the focus event's own micro-sectors -- so the window cannot be
    # widened or narrowed after the fact to make the difference look better than
    # it is. The figure gets a padded version of the same window, for context.
    focus_sectors = decomposition.sectors[decomposition.sectors["corners"].astype(str) == focus]
    deficit_start = float(focus_sectors["start_m"].min())
    deficit_end = float(focus_sectors["end_m"].max())
    window_start = float(event["left_max_m"]) - TRACE_PAD_M
    window_end = float(event["exit_end_m"]) + TRACE_PAD_M

    trace = traces.window(frames["grid"], a, b, window_start, window_end)
    deficits = traces.deficit_across_pairings(frames["grid"], quick_a, quick_b, deficit_start, deficit_end)
    deficit_summary = {
        "window_m": [deficit_start, deficit_end],
        "pairings": int(len(deficits)),
        "mean_kmh_smallest": float(deficits["mean_kmh"].max()),
        "mean_kmh_largest": float(deficits["mean_kmh"].min()),
        "all_slower_on_average": bool((deficits["mean_kmh"] < 0).all()),
        "share_slower_min": float(deficits["share_slower"].min()),
        "per_pairing": {str(index): {"mean_kmh": float(row.mean_kmh),
                                     "share_slower": float(row.share_slower)}
                        for index, row in deficits.iterrows()},
    }

    reconciliation = crosscheck.sector_reconciliation(
        frames["lap_summary"], frames["ground_truth"], frames["delta_t"],
        _sector_boundaries(processed_root), a, b, spacing_m)

    braking = crosscheck.braking_comparison(frames["grid"], frames["events"],
                                            frames["corner_metrics"], a, b)
    braking_report = crosscheck.braking_summary(braking)

    # Where the metric was misled by an extra application, ask the rest of the
    # session which behaviour is the norm: an artefact matters only if the thing
    # the metric picked up is what most laps do.
    artefacts = braking[braking["verdict"] == "metric artefact"]
    dab_laps, dab_of, dab_absent, dab_event = 0, 0, [], ""
    if not artefacts.empty:
        row = artefacts.iloc[0]
        event_id = int(artefacts.index[0])
        source = frames["events"].set_index("event_id").loc[event_id]
        lead_low = float(min(row["leading_on_a_m"], row["leading_on_b_m"]))
        prevalence = crosscheck.dab_prevalence(
            frames["grid"], frames["events"], event_id,
            (float(source["left_max_m"]), lead_low - spacing_m))
        dab_laps, dab_of = int(prevalence["dabbed"].sum()), int(len(prevalence))
        dab_absent = [f"{lap.driver} L{lap.lap_number}" for lap in
                      prevalence[~prevalence["dabbed"]].itertuples()]
        dab_event = str(row["corners"])

    session, snapshot = _session_identity(processed_root)
    rendered: list[str] = []
    if render:
        slug = f"{session}-{a[0]}{a[1]}-{b[0]}{b[1]}".lower()
        label_a, label_b = f"{a[0]} L{a[1]}", f"{b[0]} L{b[1]}"
        rendered = [
            figures.fig_phase_totals(decomposition.phases, figures_dir / f"{slug}-phase-totals.png",
                                     label_a=label_a, label_b=label_b,
                                     official_gap_s=_official_gap(frames["lap_summary"], a, b),
                                     total_s=decomposition.total_s).name,
            figures.fig_event_pairings(every_pairing, noise, figures_dir / f"{slug}-event-pairings.png",
                                       driver_a=a[0], driver_b=b[0], highlight=focus).name,
            figures.fig_trace_window(trace, figures_dir / f"{slug}-{focus.lower()}-traces.png",
                                     label_a=label_a, label_b=label_b,
                                     highlight_m=(deficit_start, deficit_end),
                                     title=f"{focus}: speed, throttle and the running delta",
                                     subtitle=f"shaded is the stretch the claim covers, "
                                              f"{int(deficit_start)}-{int(deficit_end)} m; "
                                              f"either side is context").name,
            figures.fig_sector_reconciliation(reconciliation,
                                              figures_dir / f"{slug}-sector-reconciliation.png",
                                              label_a=label_a, label_b=label_b).name,
        ]

    focus_values = every_pairing[focus].dropna()
    report = FindingsReport(
        session=session,
        snapshot=snapshot,
        pair=decomposition.label,
        official_gap_s=_official_gap(frames["lap_summary"], a, b),
        decomposed_total_s=decomposition.total_s,
        complete_sectors=decomposition.complete,
        of_sectors=decomposition.of,
        max_phase_ratio=float(decomposition.phases["ratio"].max()),
        max_event_ratio=float(decomposition.events["ratio"].max()),
        phases=decomposition.to_dict()["phases"],
        events=decomposition.to_dict()["events"],
        focus_event=focus,
        focus_median_s=float(focus_values.median()),
        focus_positive=int((focus_values > 0).sum()),
        focus_pairings=int(len(focus_values)),
        focus_sigma_s=float(noise[focus]),
        focus_window_m=[window_start, window_end],
        focus_deficit=deficit_summary,
        control=({"pair": control_pair.label,
                  "events": {str(row.corners): float(row.delta_s) for _, row in control_pair.events.iterrows()},
                  "focus_s": float(control_pair.events.loc[
                      control_pair.events["corners"].astype(str) == focus, "delta_s"].iloc[0])}
                 if control_pair is not None else {}),
        consistency={str(index): {"positive": int(row.positive), "pairings": int(row.pairings),
                                  "median_s": float(row.median_s), "sigma_s": float(row.sigma_s),
                                  "above_2_sigma": int(row.above_2_sigma)}
                     for index, row in agreement.iterrows()},
        reconciliation={str(index): {key: float(value) for key, value in row.items()}
                        for index, row in reconciliation.iterrows()},
        max_unexplained_s=float(reconciliation["unexplained_s"].abs().max()),
        braking=braking_report,
        dab_event=dab_event,
        dab_laps=dab_laps,
        dab_of=dab_of,
        dab_absent=dab_absent,
        elapsed_s=round(time.perf_counter() - started, 3),
        figures=rendered,
    )

    decomposition.sectors.to_parquet(out_root / "pair_sectors.parquet")
    every_pairing.to_parquet(out_root / "pairings.parquet")
    agreement.to_parquet(out_root / "consistency.parquet")
    reconciliation.to_parquet(out_root / "sector_reconciliation.parquet")
    braking.to_parquet(out_root / "braking.parquet")
    trace.to_parquet(out_root / "trace_window.parquet")
    (out_root / "summary.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    logger.info("findings_built session=%s pair=%s total_s=%.4f focus=%s elapsed_s=%.2f",
                session, report.pair, report.decomposed_total_s, focus, report.elapsed_s)
    return FindingsResult(root=out_root, report=report, decomposition=decomposition,
                          pairings=every_pairing, consistency=agreement,
                          reconciliation=reconciliation, braking=braking, trace=trace)
