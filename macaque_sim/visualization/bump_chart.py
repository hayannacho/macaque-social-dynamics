# bump_chart.py -- Rank step-plot over continuous time.
# Assignee: Dev F (S1-T8 prototype, S2-T8 real data).
# See: project_management/tasks/sprint_1/S1-T8_viz_prototypes.md
#      project_management/tasks/sprint_2/S2-T8_bump_chart.md
#
# The bump chart is the headline figure for CP2 -- it's what lets a reader
# eyeball "did the hierarchy actually break?" in one glance. Rank 1 sits at
# the top (y-axis reversed) and every overthrow shows up as a crossing
# between two agents' lines.

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from macaque_sim.visualization.style import (
    MATRILINE_NAMES,
    FONT_FAMILY,
    TITLE_SIZE,
    AXIS_LABEL_SIZE,
    LEGEND_SIZE,
    FIGURE_WIDTH,
    FIGURE_HEIGHT_SINGLE,
    add_event_lines,
    build_color_map,
    get_matriline_ids,
    base_layout,
    styled_axis,
)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def plot_bump_chart(
    logger,
    num_matrilines: int = 4,
    color_map: dict = None,
    matriline_ids: np.ndarray = None,
    title: str = "Rank Bump Chart — Dominance Hierarchy Over Time",
    show_events: bool = True,
    top_k: int = None,
    fig: go.Figure = None,
    row: int = None,
    save_path: str = None,
) -> go.Figure:
    """
    Build a rank-over-time step plot from a populated SimLogger.

    Each agent is one line, colored by matriline. Rank 1 sits at the top
    because y is reversed; that way a line climbing up = agent rising in
    status, which is the intuitive reading.

    Parameters worth calling out
    ----------------------------
    fig, row
        If `fig` is passed in, we add traces to that existing figure instead
        of building our own (plot_combined uses this to stack us on top of a
        centrality panel). `row` is the subplot row number in that case.
    top_k
        Only plot agents who were ever in the top-k. Useful on N=50 when the
        bottom half would just be visual noise.
    save_path
        If given, also export a static PNG at 3x scale via kaleido.
    """
    # ── Extract data from logger ──
    time_ax = logger.get_time_axis()
    ranks   = logger.get_rank_history()
    events  = logger.get_events()
    N       = logger.N

    # ── Resolve color map ──
    # Caller can pre-build one at notebook scope so every figure matches; if
    # they didn't, we build it here and the colors will still be deterministic
    # for the same (N, num_matrilines) pair.
    if matriline_ids is None:
        matriline_ids = get_matriline_ids(N, num_matrilines)
    if color_map is None:
        color_map = build_color_map(N, num_matrilines, matriline_ids)

    # ── Standalone vs. subplot mode ──
    # If the caller handed us a figure, we're a panel in something bigger and
    # must not touch the layout (they own it). If not, we manage our own.
    standalone = fig is None
    if standalone:
        fig = go.Figure()

    # ── Select which agents to draw ──
    if top_k is not None:
        # "Ever in the top-k" -- any agent that held a rank <= top_k at any
        # point, not just at the end. Catches agents who fell after rising.
        # Ranks are 1-indexed (rank 1 = alpha), so top_k=5 means ranks 1..5.
        ever_top = set()
        for t_idx in range(len(time_ax)):
            ever_top.update(np.where(ranks[t_idx] <= top_k)[0])
        agents_to_plot = sorted(ever_top)
    else:
        agents_to_plot = list(range(N))

    # ── Group by matriline so the legend lists families, not agents ──
    # 30 agents would blow out the legend; 4 matrilines fits fine.
    mat_groups: dict[int, list[int]] = {}
    for agent in agents_to_plot:
        mid = int(matriline_ids[agent])
        mat_groups.setdefault(mid, []).append(agent)

    legend_shown: set[int] = set()

    # ── Plot each agent as a step-line ──
    for mid in sorted(mat_groups):
        for agent in mat_groups[mid]:
            # Ranks are already 1-indexed in the logger (rank 1 = alpha);
            # see initializer/mri_init.py for the convention.
            display_ranks = ranks[:, agent]
            mat_name = MATRILINE_NAMES.get(mid, f"Matriline {mid}")
            # One legend entry per matriline, not per agent.
            show_legend = mid not in legend_shown
            legend_shown.add(mid)

            trace = go.Scatter(
                x=time_ax,
                y=display_ranks,
                mode="lines",
                # shape="hv" = step line (horizontal-then-vertical). Rank is
                # piecewise-constant between swaps, so a step is honest;
                # a smooth line would imply ranks moving continuously.
                line=dict(color=color_map[agent], width=1.5, shape="hv"),
                name=mat_name if show_legend else "",
                legendgroup=mat_name,
                showlegend=show_legend,
                hovertemplate=(
                    f"<b>Agent {agent}</b> ({mat_name})<br>"
                    "Time: %{x:.1f}<br>"
                    "Rank: %{y}<extra></extra>"
                ),
                opacity=0.85,
            )
            if row is not None:
                fig.add_trace(trace, row=row, col=1)
            else:
                fig.add_trace(trace)

    # ── Overlay swap event markers ──
    if show_events and events:
        # y range covers the full rank column with a small pad so the dotted
        # line extends past the top/bottom agents.
        add_event_lines(fig, events, y_min=0.5, y_max=N + 0.5, row=row)

    # ── Layout (standalone only; subplots own their own layout) ──
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
                title=dict(text="Rank (1 = alpha)",
                           font=dict(size=AXIS_LABEL_SIZE)),
                # Flip so rank 1 is at the top -- higher = better status, which
                # is what everyone expects visually.
                autorange="reversed",
                # Cap the tick density at ~10 visible ticks for N=30..50 troops.
                dtick=max(1, N // 10),
                **styled_axis(),
            ),
            width=FIGURE_WIDTH,
            height=FIGURE_HEIGHT_SINGLE,
            # Legend pinned to the right side so it doesn't eat plot area.
            legend=dict(font=dict(size=LEGEND_SIZE, family=FONT_FAMILY),
                        tracegroupgap=2,
                        x=1.02, y=1, xanchor="left"),
        )
        fig.update_layout(**layout)

    # ── Optional static export ──
    # scale=3 gives ~300 DPI at default width; good enough for report embeds.
    if save_path is not None:
        fig.write_image(save_path, scale=3)

    return fig