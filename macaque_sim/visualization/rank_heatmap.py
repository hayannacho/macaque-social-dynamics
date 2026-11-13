"""
Rank heatmap: rows = agents (sorted by initial rank), columns = time,
color = current rank. Scales readably to N=30 where an overlapping
step-line bump chart becomes illegible.

Extracted from final_demo.ipynb §8 / §12 (renamed from checkpoint2_demo.ipynb
at final submission) so the notebook can just import and call instead of
defining the helper inline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from macaque_sim.visualization.style import (
    AXIS_LABEL_SIZE,
    EXPORT_SCALE,
    FONT_FAMILY,
    TICK_SIZE,
    TITLE_SIZE,
)


def build_rank_heatmap(
    logger,
    out_png: str | Path,
    title: str = "Rank heatmap",
    orphan_id: int = -1,
    clique_ids: set[int] | None = None,
) -> go.Figure:
    """
    Render and save a rank-over-time heatmap.

    Parameters
    ----------
    logger
        Populated SimLogger (has get_time_axis, get_rank_history, get_events,
        and an N attribute).
    out_png
        PNG output path. The figure is written at scale=EXPORT_SCALE (300 DPI).
    title
        Figure title; the overthrow count is appended automatically.
    orphan_id
        If >= 0, the agent with this index gets an "  ORPHAN" suffix on the
        y-axis label. Pass -1 (default) for runs with no orphan.
    clique_ids
        Optional set of agent indices to prefix with "[C1]..[Ck]" on the
        y-axis so coalition members are identifiable at a glance.

    Returns
    -------
    The plotly Figure (already written to disk).
    """
    t_ax = logger.get_time_axis()
    R_hist = logger.get_rank_history()
    R0 = R_hist[0]
    N_agents = logger.N

    row_order = np.argsort(R0)
    R_sorted = R_hist[:, row_order].T

    clique_rank: dict[int, int] = {}
    if clique_ids:
        for i, a in enumerate(sorted(clique_ids), start=1):
            clique_rank[a] = i

    y_labels = []
    for a in row_order:
        tag_c = f"[C{clique_rank[a]}] " if a in clique_rank else ""
        tag_o = "  ORPHAN" if a == orphan_id else ""
        y_labels.append(f"{tag_c}agent {a} (rank {int(R0[a])}){tag_o}")

    fig = go.Figure(
        go.Heatmap(
            z=R_sorted,
            x=t_ax,
            y=y_labels,
            colorscale="Viridis_r",
            zmin=1,
            zmax=N_agents,
            colorbar=dict(
                title=dict(
                    text="Rank<br>(1 = Alpha)",
                    font=dict(size=AXIS_LABEL_SIZE, family=FONT_FAMILY),
                ),
                tickfont=dict(size=TICK_SIZE, family=FONT_FAMILY),
                len=0.85,
            ),
            hovertemplate="t=%{x:.2f}<br>%{y}<br>rank=%{z}<extra></extra>",
        )
    )

    # Overlay white dots at overthrow events (winner + loser each get a marker).
    ev_t: list[float] = []
    ev_y: list[str] = []
    for e in logger.get_events():
        if e.get("type") != "overthrow":
            continue
        t_e = e["t"]
        for agent in (e["details"]["winner_id"], e["details"]["loser_id"]):
            row_pos = int(np.where(row_order == agent)[0][0])
            ev_t.append(t_e)
            ev_y.append(y_labels[row_pos])
    if ev_t:
        fig.add_trace(
            go.Scatter(
                x=ev_t,
                y=ev_y,
                mode="markers",
                marker=dict(
                    size=5,
                    color="rgba(255,255,255,0.9)",
                    line=dict(color="crimson", width=0.5),
                ),
                name="swap events",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    n_over = sum(1 for e in logger.get_events() if e.get("type") == "overthrow")
    fig.update_layout(
        title=dict(
            text=title + f"  (overthrows={n_over})",
            font=dict(size=TITLE_SIZE, family=FONT_FAMILY),
        ),
        xaxis=dict(
            title=dict(text="Simulation Time",
                       font=dict(size=AXIS_LABEL_SIZE, family=FONT_FAMILY)),
            tickfont=dict(size=TICK_SIZE, family=FONT_FAMILY),
            showline=True, linecolor="black", linewidth=1, mirror=True,
        ),
        yaxis=dict(
            title=dict(text="Agent (rows sorted by initial rank)",
                       font=dict(size=AXIS_LABEL_SIZE, family=FONT_FAMILY)),
        ),
        height=600,
        width=1000,
        font=dict(family=FONT_FAMILY, size=TICK_SIZE),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=200, r=60, t=70, b=60),
        legend=dict(font=dict(size=TICK_SIZE, family=FONT_FAMILY)),
    )
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=9, family=FONT_FAMILY))

    fig.write_image(str(out_png), scale=EXPORT_SCALE)
    return fig
