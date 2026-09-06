"""The overlay figure: six panels, one distance axis, one crosshair (F017).

This is the reason the service exists. D1 chose Grafana for a dashboard-wide
crosshair -- hovering at 1,240 m reading every channel at that distance -- and
its 2026-09-04 amendment found the Trend panel has none: a distance x-axis and
a shared crosshair are mutually exclusive there, so F007 shipped one crowded
multi-axis panel instead. The amendment named the way out, and this is it:
Plotly subplots sharing an x-axis.

Pure: rows in, figure out, no I/O and no database. Everything about the
interaction is therefore checkable offline, on a designed frame.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.pitwall.queries import Overlay

#: One colour per car, held across all six panels so the eye tracks a driver
#: rather than a row. Δt is neither driver's, so it gets its own.
COLOUR_A = "#4c78a8"
COLOUR_B = "#f58518"
COLOUR_DELTA = "#e45756"

#: (title, column A, column B, step?) per panel, stacked in this order. Gear,
#: brake and DRS are discrete channels: drawn as steps, never as slopes, for
#: the same reason the resampler never interpolates them.
PANELS: tuple[tuple[str, str, str, bool], ...] = (
    ("Δt  (B − A, s)", "delta_t_s", "", False),
    ("Speed (km/h)", "speed_a", "speed_b", False),
    ("Throttle (%)", "throttle_a", "throttle_b", False),
    ("Brake", "brake_a", "brake_b", True),
    ("Gear", "n_gear_a", "n_gear_b", True),
    ("DRS", "drs_a", "drs_b", True),
)
ROW_HEIGHTS = (0.26, 0.22, 0.16, 0.12, 0.12, 0.12)


def build(overlay: Overlay) -> go.Figure:
    """Six stacked panels sharing one distance axis and one crosshair."""
    distance = overlay.column("distance_m")
    a, b = overlay.lap_a, overlay.lap_b

    fig = make_subplots(
        rows=len(PANELS), cols=1, shared_xaxes=True, vertical_spacing=0.028,
        row_heights=list(ROW_HEIGHTS),
        subplot_titles=[title for title, *_ in PANELS],
    )

    for index, (title, column_a, column_b, step) in enumerate(PANELS, start=1):
        shape = "hv" if step else "linear"
        if not column_b:
            # Δt belongs to neither car: filled to zero, so which side of the
            # line the area sits on is the answer at a glance.
            fig.add_trace(
                go.Scatter(x=distance, y=overlay.column(column_a), name="Δt",
                           line=dict(color=COLOUR_DELTA, width=2, shape=shape),
                           fill="tozeroy", fillcolor="rgba(228,87,86,0.18)",
                           hovertemplate="%{y:+.3f} s<extra>Δt</extra>"),
                row=index, col=1)
            continue
        for column, code, colour in ((column_a, a.code, COLOUR_A), (column_b, b.code, COLOUR_B)):
            fig.add_trace(
                go.Scatter(x=distance, y=overlay.column(column), name=f"{code} {title.split(' (')[0].lower()}",
                           legendgroup=code, showlegend=index == 2,
                           line=dict(color=colour, width=1.6, shape=shape),
                           hovertemplate="%{y}<extra>" + code + "</extra>"),
                row=index, col=1)

    fig.update_layout(
        title=dict(text=f"{a.label}    vs    {b.label}", x=0.01, xanchor="left"),
        # The interaction, in three settings: one tooltip listing every channel
        # at the hovered distance, a spike line drawn across all six panels,
        # and (from shared_xaxes) axes that pan and zoom together.
        hovermode="x unified",
        template="plotly_dark",
        height=1080,
        margin=dict(l=64, r=24, t=88, b=48),
        legend=dict(orientation="h", y=1.04, x=0.01, xanchor="left"),
        hoverlabel=dict(namelength=-1),
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikedash="dot", spikecolor="#888888")
    fig.update_xaxes(title_text="distance (m)", row=len(PANELS), col=1)
    fig.update_yaxes(zeroline=True, zerolinecolor="#666666", row=1, col=1)
    for index, (_, _, _, step) in enumerate(PANELS, start=1):
        if step:
            fig.update_yaxes(dtick=1, row=index, col=1)
    fig.update_yaxes(range=[-0.1, 1.1], dtick=1, row=4, col=1)
    return fig


def to_html(overlay: Overlay, script_url: str) -> str:
    """A whole page. ``script_url`` is served by this service, never a CDN."""
    return build(overlay).to_html(include_plotlyjs=script_url, full_html=True)
