"""
Animated rank bump chart with a sweeping now-cursor and gold-star swap flashes.

Extracted from final_demo.ipynb §12 (renamed from checkpoint2_demo.ipynb at
final submission) so the notebook can just import it.
Accepts an explicit color_map (per-agent color dict) rather than relying on
notebook-scoped globals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from macaque_sim.visualization.style import (
    AXIS_LABEL_SIZE,
    EXPORT_SCALE,
    FONT_FAMILY,
    LEGEND_SIZE,
    TICK_SIZE,
    TITLE_SIZE,
)


def build_rank_animation(
    logger,
    color_map: dict,
    out_html: str | Path,
    out_png: str | Path,
    title: str = "Rank animation",
    orphan_id: int = -1,
) -> tuple[go.Figure, go.Figure]:
    """
    Render an animated bump chart with swap-event flashes, write both the
    interactive HTML and a final-frame PNG snapshot.

    Parameters
    ----------
    logger
        SimLogger populated by run_simulation.
    color_map
        Dict mapping agent index -> plotly-compatible color string. Typically
        built by the notebook via `build_color_map(matriline_ids)`.
    out_html, out_png
        Output paths. HTML is interactive; PNG is the last-frame snapshot.
    title
        Figure title.
    orphan_id
        If >= 0, that agent's line is drawn in thick black with explicit
        legend entry. Pass -1 for pure-MRI runs.

    Returns
    -------
    (animated_figure, snapshot_figure). Both already written to disk.
    """
    t_ax = logger.get_time_axis()
    R_hist = logger.get_rank_history()
    events = [e for e in logger.get_events() if e.get("type") == "overthrow"]
    N_agents = logger.N

    base_lines = []
    for agent in range(N_agents):
        is_orphan = agent == orphan_id
        base_lines.append(
            go.Scatter(
                x=t_ax,
                y=R_hist[:, agent],
                mode="lines",
                line=dict(
                    color="black" if is_orphan else color_map[agent],
                    width=3 if is_orphan else 1,
                    shape="hv",
                ),
                name=f"Orphan (agent {orphan_id})" if is_orphan else "",
                showlegend=is_orphan,
                hoverinfo="skip",
                opacity=1.0 if is_orphan else 0.6,
            )
        )

    # Collect (t, y) of every swap event for both winner and loser so flashes
    # land on the right rows.
    ev_t: list[float] = []
    ev_y: list[float] = []
    for e in events:
        w = e["details"]["winner_id"]
        l = e["details"]["loser_id"]
        t_e = e["t"]
        t_idx = int(np.searchsorted(t_ax, t_e))
        t_idx = min(t_idx, len(t_ax) - 1)
        ev_t.extend([t_e, t_e])
        ev_y.extend([R_hist[t_idx, w], R_hist[t_idx, l]])

    stride = max(1, len(t_ax) // 60)
    frame_idx = list(range(0, len(t_ax), stride))
    frames = []
    for i in frame_idx:
        t_curr = t_ax[i]
        flash_window = 1.5
        sizes: list[float] = []
        colors: list[str] = []
        for t_e in ev_t:
            if t_e > t_curr:
                sizes.append(0)
                colors.append("rgba(0,0,0,0)")
            else:
                age = t_curr - t_e
                if age <= flash_window:
                    frac = age / flash_window
                    sizes.append(28 - 18 * frac)
                    colors.append(f"rgba(255, 215, 0, {1.0 - 0.5 * frac})")
                else:
                    sizes.append(7)
                    colors.append("rgba(200, 170, 30, 0.55)")
        frames.append(
            go.Frame(
                name=f"{t_curr:.2f}",
                data=[
                    *base_lines,
                    go.Scatter(
                        x=[t_curr, t_curr],
                        y=[0, N_agents + 1],
                        mode="lines",
                        line=dict(color="crimson", width=2, dash="dash"),
                        showlegend=False,
                        hoverinfo="skip",
                        name="now",
                    ),
                    go.Scatter(
                        x=ev_t,
                        y=ev_y,
                        mode="markers",
                        marker=dict(
                            size=sizes,
                            color=colors,
                            line=dict(color="red", width=1),
                            symbol="star",
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                        name="swaps",
                    ),
                ],
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Simulation time",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.4)",
        ),
        yaxis=dict(
            title="Rank (1 = alpha)",
            autorange="reversed",
            dtick=max(1, N_agents // 10),
            showgrid=True,
            gridcolor="rgba(200,200,200,0.4)",
        ),
        plot_bgcolor="white",
        height=500,
        width=1000,
        margin=dict(l=60, r=60, t=70, b=90),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.12,
                x=1.0,
                xanchor="right",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=140, redraw=True),
                                fromcurrent=True,
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                y=-0.12,
                currentvalue=dict(prefix="t = ", font=dict(size=12)),
                steps=[
                    dict(
                        method="animate",
                        label=f"{t_ax[i]:.1f}",
                        args=[
                            [f"{t_ax[i]:.2f}"],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                            ),
                        ],
                    )
                    for i in frame_idx
                ],
            )
        ],
    )
    fig.write_html(str(out_html))

    fig_snap = go.Figure(data=frames[-1].data)
    fig_snap.update_layout(
        title=dict(
            text=title + " (final frame)",
            font=dict(size=TITLE_SIZE, family=FONT_FAMILY),
        ),
        xaxis=dict(
            title=dict(text="Simulation Time",
                       font=dict(size=AXIS_LABEL_SIZE, family=FONT_FAMILY)),
            tickfont=dict(size=TICK_SIZE, family=FONT_FAMILY),
            showgrid=True,
            gridcolor="rgba(200,200,200,0.4)",
            showline=True, linecolor="black", linewidth=1, mirror=True,
        ),
        yaxis=dict(
            title=dict(text="Rank (1 = Alpha)",
                       font=dict(size=AXIS_LABEL_SIZE, family=FONT_FAMILY)),
            tickfont=dict(size=TICK_SIZE, family=FONT_FAMILY),
            autorange="reversed",
            dtick=max(1, N_agents // 10),
            showgrid=True,
            gridcolor="rgba(200,200,200,0.4)",
            showline=True, linecolor="black", linewidth=1, mirror=True,
        ),
        font=dict(family=FONT_FAMILY, size=TICK_SIZE),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
        width=1000,
        showlegend=True,
        legend=dict(
            font=dict(size=LEGEND_SIZE, family=FONT_FAMILY),
            x=1.02, y=1.0, xanchor="left",
        ),
        margin=dict(l=70, r=30, t=60, b=60),
    )
    fig_snap.write_image(str(out_png), scale=EXPORT_SCALE)
    return fig, fig_snap
