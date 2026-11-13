# centrality_ts.py -- Power-score time series + combined dual-panel plot.
# Assignee: Dev F (S2-T9).
# See: project_management/tasks/sprint_2/S2-T9_centrality_ts.md
#
# plot_centrality draws P_i(t) for every agent -- useful on its own, but the
# real value-add is plot_combined, which stacks a bump chart on top so you can
# see rank swaps and the power crossings that drove them side-by-side.

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from macaque_sim.visualization.style import (
    MATRILINE_NAMES,
    FONT_FAMILY,
    TITLE_SIZE,
    AXIS_LABEL_SIZE,
    TICK_SIZE,
    LEGEND_SIZE,
    FIGURE_WIDTH,
    FIGURE_HEIGHT_SINGLE,
    FIGURE_HEIGHT_COMBINED,
    add_event_lines,
    build_color_map,
    get_matriline_ids,
    base_layout,
    styled_axis,
)
from macaque_sim.visualization.bump_chart import plot_bump_chart


# ═══════════════════════════════════════════════════════════════════════════
# Centrality Time Series (standalone)
# ═══════════════════════════════════════════════════════════════════════════

def plot_centrality(
    logger,
    num_matrilines: int = 4,
    color_map: dict = None,
    matriline_ids: np.ndarray = None,
    title: str = "Centrality Time Series — Power Scores P_i(t)",
    show_events: bool = True,
    top_k: int = None,
    fig: go.Figure = None,
    row: int = None,
    save_path: str = None,
    _suppress_legend: bool = False,
) -> go.Figure:
    """
    Plot per-agent power score P_i(t) as stacked line traces.

    Mirrors plot_bump_chart's interface on purpose: same `fig` / `row` override
    hook so plot_combined can stack both panels without duplicating this logic.

    _suppress_legend is a private kwarg the combined view uses to hide per-row
    legends (the legend already lives in the top panel of the combined figure).
    """
    # ── Extract data ──
    time_ax = logger.get_time_axis()
    power   = logger.get_power_history()
    ranks   = logger.get_rank_history()
    events  = logger.get_events()
    N       = logger.N

    # ── Resolve colors (see bump_chart for why we build a single map) ──
    if matriline_ids is None:
        matriline_ids = get_matriline_ids(N, num_matrilines)
    if color_map is None:
        color_map = build_color_map(N, num_matrilines, matriline_ids)

    standalone = fig is None
    if standalone:
        fig = go.Figure()

    # ── Select agents + group by matriline (same rules as bump chart) ──
    agents_to_plot = _select_agents(ranks, time_ax, N, top_k)
    mat_groups = _group_by_matriline(agents_to_plot, matriline_ids)

    legend_shown: set[int] = set()

    for mid in sorted(mat_groups):
        for agent in mat_groups[mid]:
            mat_name = MATRILINE_NAMES.get(mid, f"Matriline {mid}")
            # Suppress legend entirely when we're the bottom panel of a
            # combined figure; otherwise show one entry per matriline.
            show_legend = (not _suppress_legend) and (mid not in legend_shown)
            legend_shown.add(mid)

            trace = go.Scatter(
                x=time_ax,
                y=power[:, agent],
                mode="lines",
                line=dict(color=color_map[agent], width=1.5),
                name=mat_name if show_legend else "",
                legendgroup=mat_name,
                showlegend=show_legend,
                hovertemplate=(
                    f"<b>Agent {agent}</b> ({mat_name})<br>"
                    "Time: %{x:.1f}<br>"
                    "Power: %{y:.4f}<extra></extra>"
                ),
                opacity=0.85,
            )
            if row is not None:
                fig.add_trace(trace, row=row, col=1)
            else:
                fig.add_trace(trace)

    # ── Event markers (dotted verticals at each overthrow time) ──
    if show_events and events:
        # Pad the y-range 5% so the top line doesn't clip the highest trace.
        y_max = float(power[:, agents_to_plot].max()) * 1.05
        add_event_lines(fig, events, y_min=0, y_max=y_max, row=row)

    # ── Layout (standalone) ──
    if standalone:
        layout = base_layout()
        layout.update(
            title=dict(text=title,
                       font=dict(size=TITLE_SIZE, family=FONT_FAMILY)),
            xaxis=dict(
                title=dict(text="Simulation Time",
                           font=dict(size=AXIS_LABEL_SIZE)),
                **styled_axis(),
            ),
            yaxis=dict(
                title=dict(text="Power Score P_i(t)",
                           font=dict(size=AXIS_LABEL_SIZE)),
                **styled_axis(),
            ),
            width=FIGURE_WIDTH,
            height=FIGURE_HEIGHT_SINGLE,
            legend=dict(font=dict(size=LEGEND_SIZE, family=FONT_FAMILY),
                        tracegroupgap=2,
                        x=1.02, y=1, xanchor="left"),
        )
        fig.update_layout(**layout)

    if save_path is not None:
        fig.write_image(save_path, scale=3)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Combined Dual-Panel (bump chart + centrality, shared x-axis)
# ═══════════════════════════════════════════════════════════════════════════

def plot_combined(
    logger,
    num_matrilines: int = 4,
    color_map: dict = None,
    matriline_ids: np.ndarray = None,
    title: str = "Macaque Hierarchy Dynamics — Rank & Power",
    top_k: int = None,
    save_path: str = None,
) -> go.Figure:
    """
    Dual-panel: bump chart on top, power scores below, shared x-axis.

    This is the CP2 flagship figure. Lining up the swap markers across both
    panels lets a reader see "this rank flipped BECAUSE these two power lines
    crossed" without mentally stitching two separate plots together.
    """
    N = logger.N

    # ── Build a single color map at this scope so both panels match ──
    if matriline_ids is None:
        matriline_ids = get_matriline_ids(N, num_matrilines)
    if color_map is None:
        color_map = build_color_map(N, num_matrilines, matriline_ids)

    # Shared x-axis -- zooming into one panel auto-zooms the other.
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Rank Bump Chart", "Power Score Time Series"),
        row_heights=[0.5, 0.5],
    )

    # ── Top panel: bump chart draws traces + event lines, no layout touch ──
    plot_bump_chart(
        logger,
        num_matrilines=num_matrilines,
        color_map=color_map,
        matriline_ids=matriline_ids,
        show_events=True,
        top_k=top_k,
        fig=fig,
        row=1,
    )

    # ── Bottom panel: reuse plot_centrality with legend suppressed ──
    # gotcha: used to duplicate the trace-building loop here, which drifted
    # out of sync with plot_centrality when one side changed. Now both panels
    # share one implementation.
    plot_centrality(
        logger,
        num_matrilines=num_matrilines,
        color_map=color_map,
        matriline_ids=matriline_ids,
        show_events=True,
        top_k=top_k,
        fig=fig,
        row=2,
        _suppress_legend=True,   # legend already lives in the top panel
    )

    # ── Layout ──
    fig.update_layout(
        title=dict(text=title,
                   font=dict(size=TITLE_SIZE, family=FONT_FAMILY)),
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT_COMBINED,
        font=dict(family=FONT_FAMILY, size=TICK_SIZE),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(font=dict(size=LEGEND_SIZE, family=FONT_FAMILY),
                    tracegroupgap=2,
                    x=1.02, y=1, xanchor="left"),
    )

    # Top panel y-axis
    fig.update_yaxes(
        title_text="Rank (1 = alpha)",
        autorange="reversed",
        dtick=max(1, N // 10),
        showgrid=True, gridcolor="rgba(200,200,200,0.4)",
        showline=True, linecolor="black", linewidth=1, mirror=True,
        row=1, col=1,
    )
    # Bottom panel y-axis
    fig.update_yaxes(
        title_text="Power Score P_i(t)",
        showgrid=True, gridcolor="rgba(200,200,200,0.4)",
        showline=True, linecolor="black", linewidth=1, mirror=True,
        row=2, col=1,
    )
    # Shared x-axis label on bottom panel only
    fig.update_xaxes(
        title_text="Simulation Time",
        title_font=dict(size=AXIS_LABEL_SIZE),
        showgrid=True, gridcolor="rgba(200,200,200,0.4)",
        showline=True, linecolor="black", linewidth=1, mirror=True,
        row=2, col=1,
    )

    if save_path is not None:
        fig.write_image(save_path, scale=3)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _select_agents(ranks: np.ndarray, time_ax: np.ndarray,
                   N: int, top_k: int = None) -> list[int]:
    """
    Pick which agents to draw. top_k=None -> everyone; otherwise "ever in top-k".

    "Ever in top-k" is deliberately generous -- it catches agents who rose and
    fell over the run, which is exactly what we want to see on a bump chart.
    Ranks are 1-indexed (rank 1 = alpha), so top_k=5 means ranks 1..5.
    """
    if top_k is None:
        return list(range(N))
    ever_top: set[int] = set()
    for t_idx in range(len(time_ax)):
        ever_top.update(int(a) for a in np.where(ranks[t_idx] <= top_k)[0])
    return sorted(ever_top)


def _group_by_matriline(agents: list[int],
                        matriline_ids: np.ndarray) -> dict[int, list[int]]:
    """Group agent indices by matriline so the legend can list families, not agents."""
    groups: dict[int, list[int]] = {}
    for agent in agents:
        mid = int(matriline_ids[agent])
        groups.setdefault(mid, []).append(agent)
    return groups