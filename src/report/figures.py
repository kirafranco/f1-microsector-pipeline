"""The document's figures, rendered from frames.

Code, not screenshots: the document in docs/findings/ has to be rebuildable
from data/ after a re-ingest, and a pasted image cannot be. The Agg backend is
selected explicitly so this imports the same way inside a container, a test and
a notebook, with no display attached.

Every figure that shows a delta also shows the spread that bounds it. A bar
chart of differences with no error bars is the single most misleading thing
this project could publish, given that its headline result is that the
differences do not clear the spread.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

DPI = 130
COLOUR_A = "#1f77b4"
COLOUR_B = "#ff7f0e"
COLOUR_NOISE = "#bbbbbb"
COLOUR_POSITIVE = "#c0392b"
COLOUR_NEGATIVE = "#27ae60"

#: Strip matplotlib's version stamp so a re-render of unchanged data produces
#: an unchanged file and the repository does not churn.
_METADATA = {"Software": None}


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight", metadata=_METADATA)
    plt.close(figure)
    logger.info("figure_written path=%s bytes=%d", path.name, path.stat().st_size)
    return path


def _delta_colours(values) -> list[str]:
    return [COLOUR_POSITIVE if value > 0 else COLOUR_NEGATIVE for value in values]


def fig_phase_totals(phases: pd.DataFrame, path: Path, *, label_a: str, label_b: str,
                     official_gap_s: float, total_s: float) -> Path:
    """Where the lap time went by corner phase, against the noise on each figure.

    The point of this figure is that every bar is inside its own error bar.
    """
    order = [phase for phase in ("braking", "entry", "apex", "exit", "straight") if phase in phases.index]
    frame = phases.loc[order]
    positions = np.arange(len(frame))

    figure, axes = plt.subplots(figsize=(8.2, 4.0))
    axes.barh(positions, frame["delta_s"], height=0.55,
              color=_delta_colours(frame["delta_s"]), zorder=3)
    axes.errorbar(frame["delta_s"], positions, xerr=frame["sigma_s"], fmt="none",
                  ecolor="#444444", elinewidth=1.4, capsize=5, zorder=4)
    axes.axvline(0, color="#333333", linewidth=1.0, zorder=2)

    axes.set_yticks(positions)
    axes.set_yticklabels([f"{phase}\n({int(row.sectors)} sectors, {row.length_m:,.0f} m)"
                          for phase, row in frame.iterrows()], fontsize=9)
    axes.invert_yaxis()
    axes.set_xlabel(f"time lost by {label_b} against {label_a}  (s)")
    axes.set_title(f"No phase separates from zero\n"
                   f"{label_a} vs {label_b}: official gap {official_gap_s:+.3f} s, "
                   f"micro-sectors sum to {total_s:+.3f} s", fontsize=11)
    axes.grid(axis="x", color="#eeeeee", zorder=0)
    axes.set_axisbelow(True)
    span = float((frame["delta_s"].abs() + frame["sigma_s"]).max() * 1.75)
    axes.set_xlim(-span, span)

    # Beyond the error-bar cap, never on top of it: a label drawn over the bar
    # loses its minus sign against the ink underneath.
    for position, (_, row) in zip(positions, frame.iterrows()):
        side = 1.0 if row.delta_s >= 0 else -1.0
        edge = side * (abs(row.delta_s) + row.sigma_s) + side * span * 0.03
        axes.annotate(f"{row.delta_s:+.3f} s   {row.ratio:.1f}σ",
                      xy=(edge, position), va="center",
                      ha="left" if side > 0 else "right", fontsize=8.5, color="#222222", zorder=5)
    figure.text(0.01, -0.02, "Error bars are the within-driver spread of the same micro-sectors "
                             "across the session, combined in quadrature.", fontsize=8, color="#555555")
    return _save(figure, path)


def fig_event_pairings(frame: pd.DataFrame, noise: pd.Series, path: Path, *,
                       driver_a: str, driver_b: str, highlight: str | None = None) -> Path:
    """Every lap pairing at every corner: the sign test the result rests on."""
    events = [column for column in frame.columns if column in noise.index]
    ordered = sorted(events, key=lambda event: float(frame[event].median()))
    positions = np.arange(len(ordered))

    figure, axes = plt.subplots(figsize=(8.6, 4.4))
    for position, event in zip(positions, ordered):
        sigma = float(noise[event])
        axes.add_patch(plt.Rectangle((position - 0.36, -sigma), 0.72, 2 * sigma,
                                     facecolor=COLOUR_NOISE, alpha=0.45, edgecolor="none", zorder=1))
        values = frame[event].dropna().to_numpy(dtype=float)
        jitter = np.linspace(-0.2, 0.2, len(values))
        colour = COLOUR_POSITIVE if event == highlight else "#33475b"
        axes.scatter(position + jitter, values, s=22, color=colour, alpha=0.85, zorder=3)
        axes.plot([position - 0.36, position + 0.36], [np.median(values)] * 2,
                  color=colour, linewidth=2.2, zorder=4)

    axes.axhline(0, color="#333333", linewidth=1.0, zorder=2)
    axes.set_xticks(positions)
    axes.set_xticklabels([f"{event}\n{int((frame[event] > 0).sum())}/{int(frame[event].notna().sum())}"
                          for event in ordered], fontsize=9)
    axes.set_ylabel(f"time lost by {driver_b} against {driver_a}  (s)")
    axes.set_title(f"One corner points the same way every time\n"
                   f"every {driver_a} lap against every {driver_b} lap; "
                   f"grey band is ±1σ for that corner", fontsize=11)
    axes.grid(axis="y", color="#eeeeee", zorder=0)
    axes.set_axisbelow(True)
    figure.text(0.01, -0.03, "Counts under each label are the pairings on which the corner cost "
                             f"{driver_b} time.", fontsize=8, color="#555555")
    return _save(figure, path)


def fig_trace_window(frame: pd.DataFrame, path: Path, *, label_a: str, label_b: str,
                     title: str, subtitle: str = "", highlight_m: tuple[float, float] | None = None) -> Path:
    """Speed, throttle and the running delta over one stretch of track.

    `highlight_m` shades the stretch the claim is actually about -- the corner's
    own micro-sectors -- so that a reader can see how much of the plotted window
    the claim covers, and how much is context either side of it.
    """
    figure, (speed, throttle, delta) = plt.subplots(
        3, 1, figsize=(8.4, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]})

    distance = frame["distance_m"]
    if highlight_m is not None:
        for axis in (speed, throttle, delta):
            axis.axvspan(highlight_m[0], highlight_m[1], color="#f2c94c", alpha=0.18,
                         zorder=0, linewidth=0)
    speed.plot(distance, frame["speed_a"], color=COLOUR_A, linewidth=1.8, label=label_a)
    speed.plot(distance, frame["speed_b"], color=COLOUR_B, linewidth=1.8, label=label_b)
    speed.set_ylabel("speed (km/h)")
    speed.legend(loc="lower left", frameon=False, fontsize=9)
    speed.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=11)

    throttle.plot(distance, frame["throttle_a"], color=COLOUR_A, linewidth=1.5)
    throttle.plot(distance, frame["throttle_b"], color=COLOUR_B, linewidth=1.5)
    throttle.set_ylabel("throttle (%)")
    throttle.set_ylim(-5, 105)

    delta.fill_between(distance, frame["cumulative_delta_s"], 0,
                       where=frame["cumulative_delta_s"] >= 0, color=COLOUR_POSITIVE, alpha=0.35)
    delta.fill_between(distance, frame["cumulative_delta_s"], 0,
                       where=frame["cumulative_delta_s"] < 0, color=COLOUR_NEGATIVE, alpha=0.35)
    delta.plot(distance, frame["cumulative_delta_s"], color="#333333", linewidth=1.4)
    delta.axhline(0, color="#333333", linewidth=0.9)
    delta.set_ylabel(f"Δt in window (s)")
    delta.set_xlabel("distance on the aligned axis (m)")

    for axis in (speed, throttle, delta):
        axis.grid(color="#eeeeee")
        axis.set_axisbelow(True)
    figure.align_ylabels()
    return _save(figure, path)


def fig_sector_reconciliation(frame: pd.DataFrame, path: Path, *, label_a: str, label_b: str) -> Path:
    """Grid sector gaps against official ones, and what accounts for the difference."""
    positions = np.arange(len(frame))
    width = 0.26

    figure, axes = plt.subplots(figsize=(8.2, 4.0))
    axes.bar(positions - width, frame["official_gap_s"], width, label="official timing",
             color="#33475b", zorder=3)
    axes.bar(positions, frame["grid_gap_s"], width, label="reconstructed from the grid",
             color=COLOUR_A, zorder=3)
    axes.bar(positions + width, frame["f010_residual_difference_s"], width,
             label="F010 registration residual, B − A", color=COLOUR_NOISE, zorder=3)
    axes.axhline(0, color="#333333", linewidth=1.0, zorder=2)

    axes.set_xticks(positions)
    axes.set_xticklabels([name.upper() for name in frame.index])
    axes.set_ylabel(f"time lost by {label_b} against {label_a}  (s)")
    axes.set_title("The pipeline accounts for its disagreement with official timing\n"
                   f"{label_a} vs {label_b}, per official sector", fontsize=11)
    axes.legend(frameon=False, fontsize=9)

    # Headroom first, then the labels: annotating against the current limits
    # puts the tallest group's caption through the top of the frame.
    columns = ["official_gap_s", "grid_gap_s", "f010_residual_difference_s"]
    tallest = float(frame[columns].to_numpy().max())
    lowest = float(frame[columns].to_numpy().min())
    axes.set_ylim(lowest - abs(lowest) * 0.15 - 0.01, tallest + abs(tallest) * 0.55 + 0.01)
    for position, (_, row) in zip(positions, frame.iterrows()):
        axes.annotate(f"unexplained {row.unexplained_s:+.4f} s",
                      xy=(position, max(row[column] for column in columns)),
                      xytext=(0, 9), textcoords="offset points", ha="center",
                      fontsize=8.5, color="#222222")
    axes.grid(axis="y", color="#eeeeee", zorder=0)
    axes.set_axisbelow(True)
    return _save(figure, path)
