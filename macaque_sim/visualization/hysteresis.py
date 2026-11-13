# hysteresis.py -- σ ramp-up/ramp-down loop: alpha survival vs σ on both legs.
"""
Hysteresis loop visualization. Two curves on a single axis -- forward
(blue, triangles up) and backward (red, triangles down). If the curves
separate, the system is path-dependent: the same σ yields different
alpha-survival probabilities depending on how you arrived.

Expected JSON schema (see experiments/hysteresis_experiment.py):
  {
    "sigma_forward":      [...],  # σ values on up-ramp
    "sigma_backward":     [...],  # σ values on down-ramp (== reverse)
    "forward_survival":   [...],  # P(alpha holds | σ, up-ramp)
    "backward_survival":  [...],  # P(alpha holds | σ, down-ramp)
    "num_mc_runs": int, "segment_T": float, "n_steps": int, ...
  }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from macaque_sim.visualization.style import (
    AXIS_LABEL_SIZE,
    EXPORT_SCALE,
    FIGURE_HEIGHT_SINGLE,
    FIGURE_WIDTH,
    FONT_FAMILY,
    LEGEND_SIZE,
    TITLE_SIZE,
    base_layout,
    styled_axis,
)


def plot_hysteresis_loop(
    results: dict[str, Any],
    title: str = "Hysteresis Loop — Alpha Survival vs σ (Ramp Up / Ramp Down)",
    save_path: str | Path | None = None,
) -> go.Figure:
    """
    Render the forward/backward σ sweep as a single hysteresis-loop plot.

    A persistent gap between the two curves is the smoking gun for bistability:
    within the σ range where both curves differ, the system has two (or more)
    stable states and the one it settles into depends on its history.
    """
    sigma_fwd = np.array(results["sigma_forward"], dtype=float)
    sigma_bwd = np.array(results["sigma_backward"], dtype=float)
    surv_fwd = np.array(results["forward_survival"], dtype=float) * 100.0
    surv_bwd = np.array(results["backward_survival"], dtype=float) * 100.0

    fig = go.Figure()

    # --- Forward leg (ramp up) ---
    fig.add_trace(go.Scatter(
        x=sigma_fwd, y=surv_fwd,
        mode="lines+markers",
        line=dict(color="rgb(31,119,180)", width=2.5),
        marker=dict(size=9, symbol="triangle-up"),
        name="Forward (ramp ↑)",
        hovertemplate=(
            "σ: %{x:.3f}<br>"
            "P(alpha holds): %{y:.1f}%<br>"
            "leg: forward<extra></extra>"
        ),
    ))

    # --- Backward leg (ramp down) ---
    fig.add_trace(go.Scatter(
        x=sigma_bwd, y=surv_bwd,
        mode="lines+markers",
        line=dict(color="rgb(214,39,40)", width=2.5),
        marker=dict(size=9, symbol="triangle-down"),
        name="Backward (ramp ↓)",
        hovertemplate=(
            "σ: %{x:.3f}<br>"
            "P(alpha holds): %{y:.1f}%<br>"
            "leg: backward<extra></extra>"
        ),
    ))

    # --- Shade the hysteresis gap so the path-dependence jumps out visually ---
    # sigma_bwd is reversed; sort both onto the same grid to compute the gap.
    order = np.argsort(sigma_fwd)
    xs = sigma_fwd[order]
    y_fwd_sorted = surv_fwd[order]
    # Pair backward by σ value (not index) so we compare at matching σ.
    bwd_map = dict(zip(np.round(sigma_bwd, 8), surv_bwd))
    y_bwd_sorted = np.array([bwd_map[round(x, 8)] for x in xs])

    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([y_fwd_sorted, y_bwd_sorted[::-1]]),
        fill="toself",
        fillcolor="rgba(148,103,189,0.15)",
        line=dict(width=0),
        showlegend=True,
        name="hysteresis gap",
        hoverinfo="skip",
    ))

    # --- Annotation with run metadata ---
    n_mc = results.get("num_mc_runs", "?")
    seg_T = results.get("segment_T", "?")
    meta = f"n_MC = {n_mc}, segment_T = {seg_T}"
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98, xanchor="left", yanchor="top",
        text=meta,
        showarrow=False,
        font=dict(size=11, family=FONT_FAMILY, color="rgba(0,0,0,0.6)"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.15)",
        borderwidth=1,
    )

    layout = base_layout()
    layout.update(
        title=dict(text=title, font=dict(size=TITLE_SIZE, family=FONT_FAMILY)),
        xaxis=dict(
            title=dict(text="Social Noise σ", font=dict(size=AXIS_LABEL_SIZE)),
            **styled_axis(),
        ),
        yaxis=dict(
            title=dict(text="Alpha Survival Rate (%)",
                       font=dict(size=AXIS_LABEL_SIZE)),
            range=[0, 105],
            **styled_axis(),
        ),
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT_SINGLE,
        legend=dict(
            font=dict(size=LEGEND_SIZE, family=FONT_FAMILY),
            x=1.02, y=1, xanchor="left",
        ),
    )
    fig.update_layout(**layout)

    if save_path is not None:
        fig.write_image(str(save_path), scale=EXPORT_SCALE)

    return fig
