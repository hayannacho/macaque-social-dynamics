"""
Network-flow animation: directed edges F_ij = A_ij · P_j show where power is
flowing. Same spring layout as the arena, but edges carry flux magnitude
rather than symmetric alliance strength.

Extracted from final_demo.ipynb §15 (renamed from checkpoint2_demo.ipynb at
final submission). Reuses spring_positions from `.arena` so the layout code
doesn't duplicate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from macaque_sim.visualization.arena import spring_positions
from macaque_sim.visualization.style import (
    EXPORT_SCALE,
    FONT_FAMILY,
    TICK_SIZE,
    TITLE_SIZE,
)


def build_flow_animation(
    cfg,
    A_hist: np.ndarray,
    P_hist: np.ndarray,
    R_hist: np.ndarray,
    t_axis: np.ndarray,
    out_html: str | Path,
    out_png: str | Path,
    title: str = "Network flow",
    orphan_id: int = -1,
    matriline_labels: list[str] | None = None,
    top_k_edges: int = 40,
) -> go.Figure:
    """
    Per-frame flux-edge animation. Edge color + width = F_ij = A_ij · P_j
    (flow rate from source j to destination i); right colorbar = node rank,
    left colorbar = edge flux.

    Parameters
    ----------
    cfg
        SimConfig (needs N).
    A_hist, P_hist, R_hist, t_axis
        Per-step histories from a completed SimLogger.
    out_html, out_png
        Output paths.
    title
        Figure title.
    orphan_id
        Agent index to outline in red (-1 to disable).
    matriline_labels
        Optional per-agent label; used in hover text.
    top_k_edges
        Only the top-k strongest fluxes per frame are drawn.

    Returns
    -------
    The animated figure (HTML and PNG both already written).
    """
    stride = max(1, len(t_axis) // 60)
    frame_idx = list(range(0, len(t_axis), stride))

    _rng = np.random.default_rng(0)
    prev_pos = {a: (_rng.uniform(-1, 1), _rng.uniform(-1, 1)) for a in range(cfg.N)}
    pos_per_frame = []
    for i in frame_idx:
        prev_pos = spring_positions(
            A_hist[i], prev_pos, k=0.55, n_iters=15, min_weight=0.08
        )
        pos_per_frame.append(prev_pos)

    all_x = np.array([p[a][0] for p in pos_per_frame for a in range(cfg.N)])
    all_y = np.array([p[a][1] for p in pos_per_frame for a in range(cfg.N)])
    pad = 0.15
    x_range = [all_x.min() - pad, all_x.max() + pad]
    y_range = [all_y.min() - pad, all_y.max() + pad]

    F_max = 0.0
    for i in frame_idx:
        A, P = A_hist[i], P_hist[i]
        F_frame = A * P[np.newaxis, :]
        np.fill_diagonal(F_frame, 0.0)
        F_max = max(F_max, float(F_frame.max()))
    F_max = max(F_max, 1e-9)

    N_BUCKETS = 5
    bucket_fracs = [(b + 0.5) / N_BUCKETS for b in range(N_BUCKETS)]
    bucket_colors = sample_colorscale("Plasma", bucket_fracs)
    bucket_widths = [0.7, 1.3, 2.0, 2.7, 3.5]

    labels = matriline_labels if matriline_labels is not None else [f"agent {a}" for a in range(cfg.N)]

    frames = []
    for k_idx, i in enumerate(frame_idx):
        A, P, R = A_hist[i], P_hist[i], R_hist[i]
        pos = pos_per_frame[k_idx]
        px = np.array([pos[a][0] for a in range(cfg.N)])
        py = np.array([pos[a][1] for a in range(cfg.N)])

        F = A * P[np.newaxis, :]
        np.fill_diagonal(F, 0.0)

        edges: list[tuple[int, int, float]] = []
        for ii in range(cfg.N):
            for jj in range(cfg.N):
                if ii == jj:
                    continue
                f = float(F[ii, jj])
                if f > 0:
                    edges.append((ii, jj, f))
        edges.sort(key=lambda e: -e[2])
        edges = edges[:top_k_edges]

        bucket_xs: list[list[float]] = [[] for _ in range(N_BUCKETS)]
        bucket_ys: list[list[float]] = [[] for _ in range(N_BUCKETS)]
        for dst, src, f in edges:
            frac = f / F_max
            b = min(N_BUCKETS - 1, int(frac * N_BUCKETS))
            bucket_xs[b].extend([px[src], px[dst], None])
            bucket_ys[b].extend([py[src], py[dst], None])

        arrow_x: list[float] = []
        arrow_y: list[float] = []
        for dst, src, f in edges:
            arrow_x.append(0.2 * px[src] + 0.8 * px[dst])
            arrow_y.append(0.2 * py[src] + 0.8 * py[dst])

        sizes = [12 + 500 * float(P[a]) for a in range(cfg.N)]
        outline_color = ["red" if a == orphan_id else "white" for a in range(cfg.N)]
        outline_width = [3 if a == orphan_id else 1 for a in range(cfg.N)]
        hover = [
            (
                f"agent {a}<br>rank={int(R[a])}<br>P={P[a]:.3f}"
                f"<br>in-flux: {float(F[a].sum()):.3f}"
                f"<br>{'ORPHAN' if a == orphan_id else labels[a]}"
            )
            for a in range(cfg.N)
        ]

        frame_data = []
        for b in range(N_BUCKETS):
            frame_data.append(
                go.Scatter(
                    x=bucket_xs[b],
                    y=bucket_ys[b],
                    mode="lines",
                    line=dict(color=bucket_colors[b], width=bucket_widths[b]),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"flux_b{b}",
                )
            )
        frame_data.append(
            go.Scatter(
                x=arrow_x,
                y=arrow_y,
                mode="markers",
                marker=dict(
                    size=5,
                    color="rgba(255,255,255,0.85)",
                    line=dict(color="black", width=0.5),
                ),
                hoverinfo="skip",
                showlegend=False,
                name="arrows",
            )
        )
        frame_data.append(
            go.Scatter(
                x=px,
                y=py,
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=R.astype(float),
                    colorscale="Viridis_r",
                    cmin=1,
                    cmax=cfg.N,
                    colorbar=dict(title="rank<br>(1=alpha)", x=1.02, len=0.85, y=0.5),
                    line=dict(color=outline_color, width=outline_width),
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
                name="monkeys",
            )
        )
        frame_data.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    color=[0.0, F_max],
                    colorscale="Plasma",
                    cmin=0.0,
                    cmax=F_max,
                    showscale=True,
                    colorbar=dict(
                        title="edge flux<br>F_ij = A_ij·P_j",
                        x=-0.18,
                        len=0.85,
                        y=0.5,
                        thickness=14,
                    ),
                    size=0.001,
                ),
                hoverinfo="skip",
                showlegend=False,
                name="flux_cbar",
            )
        )
        frame_data.append(
            go.Scatter(
                x=[x_range[0] + 0.05],
                y=[y_range[1] - 0.07],
                mode="text",
                text=[f"t = {t_axis[i]:.2f}  |  top {len(edges)} fluxes shown"],
                textfont=dict(size=13, color="black"),
                textposition="top right",
                hoverinfo="skip",
                showlegend=False,
                name="time",
            )
        )

        frames.append(go.Frame(name=f"{t_axis[i]:.2f}", data=frame_data))

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False, range=x_range),
        yaxis=dict(visible=False, range=y_range, scaleanchor="x"),
        height=640,
        width=880,
        plot_bgcolor="rgba(248, 248, 248, 1)",
        paper_bgcolor="white",
        margin=dict(l=110, r=120, t=60, b=80),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.08,
                x=1.0,
                xanchor="right",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(frame=dict(duration=160, redraw=True), fromcurrent=True),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                y=-0.06,
                currentvalue=dict(prefix="t = ", font=dict(size=12)),
                steps=[
                    dict(
                        method="animate",
                        label=f"{t_axis[i]:.1f}",
                        args=[
                            [f"{t_axis[i]:.2f}"],
                            dict(mode="immediate", frame=dict(duration=0, redraw=True)),
                        ],
                    )
                    for i in frame_idx
                ],
            )
        ],
    )
    fig.write_html(str(out_html))

    snap_idx = len(frames) // 2
    fig_snap = go.Figure(data=frames[snap_idx].data)
    fig_snap.update_layout(
        title=dict(
            text=title + f"  (t = {t_axis[frame_idx[snap_idx]]:.1f})",
            font=dict(size=TITLE_SIZE, family=FONT_FAMILY),
        ),
        xaxis=dict(visible=False, range=x_range),
        yaxis=dict(visible=False, range=y_range, scaleanchor="x"),
        height=620,
        width=820,
        font=dict(family=FONT_FAMILY, size=TICK_SIZE),
        plot_bgcolor="rgba(248, 248, 248, 1)",
        paper_bgcolor="white",
        margin=dict(l=110, r=120, t=60, b=40),
    )
    fig_snap.write_image(str(out_png), scale=EXPORT_SCALE)
    return fig
