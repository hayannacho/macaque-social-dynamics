"""
Particle-arena animation of the alliance graph. Spring layout is driven by
A_ij; strong dyads pull pairs close. Node color = rank, size = P.

Extracted from final_demo.ipynb §13 (the notebook was named
checkpoint2_demo.ipynb in earlier sprints). Takes explicit matriline context
rather than relying on notebook-scoped globals.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from macaque_sim.visualization.style import (
    EXPORT_SCALE,
    FONT_FAMILY,
    TICK_SIZE,
    TITLE_SIZE,
)


def greedy_dyads(A: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """
    Strict-dyadic greedy match: each monkey in at most one pair per frame.

    Used by build_arena_animation to compute which pairs are "actively
    grooming" at each frame, capped at floor(N/2) concurrent dyads.
    """
    N = A.shape[0]
    A_sym = (A + A.T) / 2
    candidates = [
        (i, j, float(A_sym[i, j]))
        for i in range(N)
        for j in range(i + 1, N)
        if A_sym[i, j] >= threshold
    ]
    candidates.sort(key=lambda p: -p[2])
    busy: set[int] = set()
    dyads: list[tuple[int, int, float]] = []
    for i, j, s in candidates:
        if i in busy or j in busy:
            continue
        dyads.append((i, j, s))
        busy.update({i, j})
    return dyads


def spring_positions(
    A: np.ndarray,
    prev_pos: dict | None,
    k: float = 0.6,
    n_iters: int = 20,
    min_weight: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Per-frame NetworkX spring layout driven by alliance strengths, warm-started
    from the previous frame so particles glide rather than teleport.
    """
    N = A.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    A_sym = (A + A.T) / 2
    for i in range(N):
        for j in range(i + 1, N):
            w = float(A_sym[i, j])
            if w > min_weight:
                G.add_edge(i, j, weight=w)
    return nx.spring_layout(
        G, pos=prev_pos, weight="weight", k=k, iterations=n_iters, seed=seed
    )


def _hover_text(
    a: int,
    rank: int,
    P_val: float,
    busy_agents: set[int],
    orphan_id: int,
    matriline_label: str,
) -> str:
    orphan_tag = "ORPHAN" if a == orphan_id else matriline_label
    status = "grooming" if a in busy_agents else "idle"
    return f"agent {a}<br>rank={rank}<br>P={P_val:.3f}<br>{orphan_tag}<br>{status}"


def build_arena_animation(
    cfg,
    A_hist: np.ndarray,
    P_hist: np.ndarray,
    R_hist: np.ndarray,
    t_axis: np.ndarray,
    out_html: str | Path,
    out_png: str | Path,
    title: str = "Particle arena",
    orphan_id: int = -1,
    matriline_labels: list[str] | None = None,
) -> tuple[go.Figure, float, list[int]]:
    """
    Agar.io-style dyadic-grooming animation.

    Edge color + width encodes pair alliance strength (A_ij + A_ji)/2 via the
    Inferno colorscale; a dedicated left-side colorbar keeps the edge channel
    legible alongside the right-side rank colorbar.

    Parameters
    ----------
    cfg
        SimConfig with N attribute.
    A_hist, P_hist, R_hist, t_axis
        Per-step histories from a completed SimLogger.
    out_html, out_png
        Output paths.
    title
        Figure title.
    orphan_id
        Agent to outline in red (-1 to disable).
    matriline_labels
        Optional per-agent matriline label string (list length N). If None,
        hovers fall back to "agent a" with no matriline tag.

    Returns
    -------
    (figure, pair_threshold, frame_idx_used) -- caller can reuse the threshold
    and frame indices for downstream analysis (e.g., printing dyad stats).
    """
    A_max = float(A_hist.max())
    pair_threshold = A_max * 0.40

    stride = max(1, len(t_axis) // 60)
    frame_idx = list(range(0, len(t_axis), stride))

    _rng = np.random.default_rng(0)
    initial_pos = {a: (_rng.uniform(-1, 1), _rng.uniform(-1, 1)) for a in range(cfg.N)}
    pos_per_frame = []
    prev_pos = initial_pos
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

    edge_vmin = pair_threshold
    edge_vmax = A_max

    N_EDGE_BUCKETS = 5
    bucket_fracs = [(b + 0.5) / N_EDGE_BUCKETS for b in range(N_EDGE_BUCKETS)]
    edge_bucket_colors = sample_colorscale("Inferno", bucket_fracs)
    edge_bucket_widths = [1.2, 1.8, 2.4, 3.0, 3.6]

    def _bucket(strength: float) -> int:
        frac = (strength - edge_vmin) / max(edge_vmax - edge_vmin, 1e-9)
        return min(N_EDGE_BUCKETS - 1, max(0, int(frac * N_EDGE_BUCKETS)))

    labels = matriline_labels if matriline_labels is not None else [f"agent {a}" for a in range(cfg.N)]

    frames = []
    for k_idx, i in enumerate(frame_idx):
        A, P, R = A_hist[i], P_hist[i], R_hist[i]
        dyads = greedy_dyads(A, pair_threshold)
        busy_agents = {a for pair in dyads for a in pair[:2]}

        pos = pos_per_frame[k_idx]
        px = np.array([pos[a][0] for a in range(cfg.N)])
        py = np.array([pos[a][1] for a in range(cfg.N)])

        bucket_xs: list[list[float]] = [[] for _ in range(N_EDGE_BUCKETS)]
        bucket_ys: list[list[float]] = [[] for _ in range(N_EDGE_BUCKETS)]
        for ai, aj, s in dyads:
            b = _bucket(s)
            bucket_xs[b].extend([px[ai], px[aj], None])
            bucket_ys[b].extend([py[ai], py[aj], None])

        sizes = [12 + 500 * float(P[a]) for a in range(cfg.N)]
        outline_color = ["red" if a == orphan_id else "white" for a in range(cfg.N)]
        outline_width = [3 if a == orphan_id else 1 for a in range(cfg.N)]
        hover = [
            _hover_text(a, int(R[a]), float(P[a]), busy_agents, orphan_id, labels[a])
            for a in range(cfg.N)
        ]

        frame_data = []
        for b in range(N_EDGE_BUCKETS):
            frame_data.append(
                go.Scatter(
                    x=bucket_xs[b],
                    y=bucket_ys[b],
                    mode="lines",
                    line=dict(color=edge_bucket_colors[b], width=edge_bucket_widths[b]),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"dyad_b{b}",
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
                    color=[edge_vmin, edge_vmax],
                    colorscale="Inferno",
                    cmin=edge_vmin,
                    cmax=edge_vmax,
                    showscale=True,
                    colorbar=dict(
                        title="dyad<br>strength<br>(A_ij+A_ji)/2",
                        x=-0.18,
                        len=0.85,
                        y=0.5,
                        thickness=14,
                    ),
                    size=0.001,
                ),
                hoverinfo="skip",
                showlegend=False,
                name="edge_cbar",
            )
        )
        frame_data.append(
            go.Scatter(
                x=[x_range[0] + 0.05],
                y=[y_range[1] - 0.07],
                mode="text",
                text=[f"t = {t_axis[i]:.2f}  |  dyads = {len(dyads)}"],
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
    return fig, pair_threshold, frame_idx
