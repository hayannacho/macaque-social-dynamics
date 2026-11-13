# phase_diagram.py -- 2D stability heatmap (lambda vs grooming -> survival rate).
# Assignee: Dev F (S4-T2).
# See: project_management/tasks/sprint_4/S4-T2_phase_diagram.md
#
# This is the centerpiece figure for the final report. Each (x, y) cell is the
# Monte-Carlo-estimated probability that the alpha's matriline survives a full
# run at those parameters. The black dashed contour at 50% marks the transition
# between stable-aristocracy and frequent-overthrow regimes.

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from macaque_sim.visualization.style import (
    FONT_FAMILY,
    TITLE_SIZE,
    AXIS_LABEL_SIZE,
    TICK_SIZE,
    FIGURE_WIDTH,
    base_layout,
    styled_axis,
)


# ═══════════════════════════════════════════════════════════════════════════
# 2D Phase Diagram (heatmap)
# ═══════════════════════════════════════════════════════════════════════════

def plot_phase_diagram(
    survival_grid: np.ndarray,
    lambda_values: np.ndarray,
    grooming_values: np.ndarray,
    title: str = "Phase Diagram — Aristocracy Survival Rate",
    x_label: str = "Memory Decay Rate \u03bb",
    y_label: str = "Grooming Rate Multiplier",
    contour_level: float = 0.5,
    save_path: str = None,
) -> go.Figure:
    """
    Heatmap of aristocracy survival rate over (lambda, grooming) + a dashed
    contour at `contour_level` marking the stability transition.

    survival_grid is expected as [len(grooming_values), len(lambda_values)]
    with values in [0, 1] (multiplied by 100 for display). That layout matches
    what survival_grid_from_mc_results spits out, so you can usually pipe the
    two together directly.
    """
    fig = go.Figure()

    # ── Main heatmap ──
    # zmin/zmax pinned to [0, 100] so the colorscale's midpoint (white) always
    # lands on exactly the 50% transition -- regardless of what the actual
    # min/max of the survival_grid happen to be for this sweep.
    fig.add_trace(go.Heatmap(
        z=survival_grid * 100,
        x=lambda_values,
        y=grooming_values,
        colorscale=_DIVERGING_COLORSCALE,
        colorbar=dict(
            title=dict(
                text="Aristocracy<br>Survival Rate (%)",
                font=dict(size=AXIS_LABEL_SIZE, family=FONT_FAMILY),
            ),
            tickfont=dict(size=TICK_SIZE, family=FONT_FAMILY),
            ticksuffix="%",
            len=0.85,
        ),
        zmin=0, zmax=100,
        hovertemplate=(
            "\u03bb (decay): %{x:.3f}<br>"
            "Grooming rate: %{y:.2f}\u00d7<br>"
            "Survival: %{z:.1f}%<extra></extra>"
        ),
    ))

    # ── Transition boundary contour ──
    # Single isocontour at `contour_level` (default 50% survival).
    # gotcha: Plotly's go.Contour fills regions by default with its own colorscale
    # (Viridis). That fill paints over the underlying Heatmap and replaces the
    # red/white/blue colors with yellow/teal/blue, breaking the colorbar mapping.
    # Setting contours.coloring="lines" suppresses the fill so only the dashed
    # boundary line is drawn on top of the heatmap.
    fig.add_trace(go.Contour(
        z=survival_grid * 100,
        x=lambda_values,
        y=grooming_values,
        contours=dict(
            coloring="lines",   # CRITICAL: suppress fill, draw only the line
            start=contour_level * 100,
            end=contour_level * 100,
            size=0,
            showlabels=True,
            labelfont=dict(size=12, color="black", family=FONT_FAMILY),
        ),
        # gotcha: with contours.coloring="lines", the line color is taken from
        # the colorscale, NOT from line.color. Pin the entire colorscale to
        # black so the contour renders as a solid black dashed line.
        colorscale=[[0.0, "black"], [1.0, "black"]],
        line=dict(width=2.5, dash="dash"),
        showscale=False,
        name=f"Transition ({int(contour_level * 100)}% survival)",
        hoverinfo="skip",
    ))

    # ── Region annotations ──
    _annotate_regions(fig, lambda_values, grooming_values)

    # ── Layout ──
    fig.update_layout(
        title=dict(text=title,
                   font=dict(size=TITLE_SIZE, family=FONT_FAMILY)),
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=AXIS_LABEL_SIZE)),
            showgrid=False, showline=True, linecolor="black", linewidth=1,
            mirror=True,
            tickfont=dict(size=TICK_SIZE, family=FONT_FAMILY),
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=AXIS_LABEL_SIZE)),
            showgrid=False, showline=True, linecolor="black", linewidth=1,
            mirror=True,
            tickfont=dict(size=TICK_SIZE, family=FONT_FAMILY),
        ),
        width=FIGURE_WIDTH,
        height=650,
        font=dict(family=FONT_FAMILY, size=TICK_SIZE),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=30, t=70, b=70),
    )

    if save_path is not None:
        fig.write_image(save_path, scale=3)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 1D Sweep Plot (from MCResult)
# ═══════════════════════════════════════════════════════════════════════════

def plot_sweep(
    mc_result,
    metric: str = "mean_overthrow_count",
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    show_error: bool = True,
    color: str = "rgb(31,119,180)",
    save_path: str = None,
) -> go.Figure:
    """
    1-D plot of a Monte Carlo sweep result: metric vs the swept parameter.

    Takes an MCResult (Dev E's container), pulls `metric` off each grid point,
    and draws a line with an optional +-1 standard-deviation band. Used for
    the noise / decay sweeps where we vary one parameter and care about how a
    single scalar outcome responds.
    """
    if mc_result.sweep_key is None:
        raise ValueError(
            "MCResult has no sweep — use plot_phase_diagram for 2D data "
            "or pass a sweep-based MCResult."
        )

    sweep_vals = np.array(mc_result.sweep_values)
    y_vals = np.array([getattr(gp, metric) for gp in mc_result.grid])

    if title is None:
        title = f"{mc_result.scenario_name} — {metric}"
    if x_label is None:
        x_label = mc_result.sweep_key
    if y_label is None:
        y_label = metric.replace("_", " ").title()

    fig = go.Figure()

    # Error band (only wired up for overthrow count so far -- other metrics
    # don't expose a std field on the grid-point struct. Could generalize if
    # Dev E's MCResult grows more stddev fields down the road).
    if show_error and metric == "mean_overthrow_count":
        stds = np.array([gp.std_overthrow_count for gp in mc_result.grid])
        # Two invisible scatter traces with fill="tonexty" between them =
        # standard Plotly trick for a shaded band without a dedicated widget.
        fig.add_trace(go.Scatter(
            x=sweep_vals, y=y_vals + stds,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=sweep_vals, y=np.maximum(0, y_vals - stds),
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor=color.replace("rgb", "rgba").replace(")", ",0.2)"),
            showlegend=False, hoverinfo="skip",
        ))

    # Main line
    fig.add_trace(go.Scatter(
        x=sweep_vals,
        y=y_vals,
        mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=7),
        name=y_label,
        hovertemplate=(
            f"{x_label}: " + "%{x:.4g}<br>"
            f"{y_label}: " + "%{y:.3g}<extra></extra>"
        ),
    ))

    layout = base_layout()
    layout.update(
        title=dict(text=title,
                   font=dict(size=TITLE_SIZE, family=FONT_FAMILY)),
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=AXIS_LABEL_SIZE)),
            **styled_axis(),
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=AXIS_LABEL_SIZE)),
            **styled_axis(),
        ),
        width=FIGURE_WIDTH, height=500,
    )
    fig.update_layout(**layout)

    if save_path is not None:
        fig.write_image(save_path, scale=3)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Helper: assemble 2D grid from multiple MCResults
# ═══════════════════════════════════════════════════════════════════════════

def survival_grid_from_mc_results(
    mc_results: list,
    outer_values: np.ndarray,
    metric: str = "rank_stability_index",
) -> np.ndarray:
    """
    Stitch a list of 1D MCResults into the 2D grid plot_phase_diagram wants.

    Typical usage: run one sweep per value of some outer parameter (e.g. one
    grooming-rate sweep per value of sigma), collect the MCResults in a list,
    then call this to get the (outer x inner) grid for the heatmap.

    Raises ValueError if the grids aren't rectangular (each MCResult must have
    the same number of inner grid points as the first one).
    """
    if len(mc_results) != len(outer_values):
        raise ValueError(
            f"mc_results length ({len(mc_results)}) must match "
            f"outer_values length ({len(outer_values)})."
        )

    n_inner = len(mc_results[0].grid)
    grid = np.zeros((len(outer_values), n_inner))

    for i, mcr in enumerate(mc_results):
        if len(mcr.grid) != n_inner:
            raise ValueError(
                f"All MCResults must have the same number of grid points. "
                f"Expected {n_inner}, got {len(mcr.grid)} at index {i}."
            )
        for j, gp in enumerate(mcr.grid):
            grid[i, j] = getattr(gp, metric)

    return grid


# ═══════════════════════════════════════════════════════════════════════════
# Internal constants and helpers
# ═══════════════════════════════════════════════════════════════════════════

# Diverging blue-white-red colorscale: red = low survival (aristocracy falls),
# white = 50% transition, blue = high survival (aristocracy persists). Picked
# a perceptually-symmetric ramp so "half the cells red, half blue" reads as
# "you're right at the transition" at a glance.
_DIVERGING_COLORSCALE = [
    [0.0, "rgb(103, 0, 31)"],       # dark red   -- aristocracy topples ~0%
    [0.2, "rgb(214, 96, 77)"],
    [0.4, "rgb(244, 165, 130)"],
    [0.5, "rgb(247, 247, 247)"],    # near-white -- 50% transition
    [0.6, "rgb(146, 197, 222)"],
    [0.8, "rgb(67, 147, 195)"],
    [1.0, "rgb(5, 48, 97)"],        # dark blue  -- aristocracy always holds
]


def _annotate_regions(fig: go.Figure, lambda_values: np.ndarray,
                      grooming_values: np.ndarray) -> None:
    """
    Drop STABLE / UNSTABLE labels into the two extreme quadrants.

    Placement is heuristic (1/4 and 3/4 along each axis) rather than inferred
    from the actual grid data. Works fine for the default sweep ranges; if a
    future sweep puts the transition somewhere unusual, the labels may land on
    the wrong side and want manual adjustment.
    """
    # Stable region: low-lambda, low-grooming corner
    fig.add_annotation(
        x=lambda_values[len(lambda_values) // 4],
        y=grooming_values[len(grooming_values) // 4],
        text="<b>STABLE</b><br>Aristocracy<br>Persists",
        showarrow=False,
        font=dict(size=14, color="white", family=FONT_FAMILY),
        bgcolor="rgba(5,48,97,0.5)",
        bordercolor="white", borderwidth=1,
    )
    # Unstable region: high-lambda, high-grooming corner
    fig.add_annotation(
        x=lambda_values[3 * len(lambda_values) // 4],
        y=grooming_values[3 * len(grooming_values) // 4],
        text="<b>UNSTABLE</b><br>Frequent<br>Overthrows",
        showarrow=False,
        font=dict(size=14, color="white", family=FONT_FAMILY),
        bgcolor="rgba(103,0,31,0.5)",
        bordercolor="white", borderwidth=1,
    )