# style.py -- Shared Plotly theme, color palettes, font settings.
# Assignee: Dev F (S1-T8 prototype, S4-T3 polish).
# See: project_management/tasks/sprint_1/S1-T8_viz_prototypes.md
#      project_management/tasks/sprint_4/S4-T3_chart_polish.md
#
# This is the single source of truth for every Plotly figure in the project.
# bump_chart / centrality_ts / phase_diagram all import from here so changing
# the font or palette in one place propagates to every figure.

from __future__ import annotations

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Font & dimension constants
# ═══════════════════════════════════════════════════════════════════════════

FONT_FAMILY = "Computer Modern, Times New Roman, serif"
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 14
TICK_SIZE = 11
LEGEND_SIZE = 10
FIGURE_WIDTH = 1100
FIGURE_HEIGHT_SINGLE = 550
FIGURE_HEIGHT_COMBINED = 900
EXPORT_SCALE = 3  # Plotly base is 100 DPI; scale=3 → 300 DPI

# ═══════════════════════════════════════════════════════════════════════════
# Matriline color palettes
# ═══════════════════════════════════════════════════════════════════════════

# Each matriline gets its own perceptually-distinct hue; agents within the same
# matriline share that hue but vary in lightness so you can still tell them
# apart on a 30-agent bump chart without the legend turning into a rainbow.
# Hand-tuned up to 8 matrilines (biological default is num_matrilines=4, so 8
# gives us plenty of headroom). Beyond 8 we just wrap -- not great but also
# not a regime we run in practice.

MATRILINE_BASE_COLORS = {
    0: (31,  119, 180),   # Blue
    1: (214,  39,  40),   # Red
    2: (44,  160,  44),   # Green
    3: (148, 103, 189),   # Purple
    4: (255, 127,  14),   # Orange
    5: (140,  86,  75),   # Brown
    6: (227, 119, 194),   # Pink
    7: (127, 127, 127),   # Gray
}

MATRILINE_NAMES = {
    0: "Matriline A",  1: "Matriline B",
    2: "Matriline C",  3: "Matriline D",
    4: "Matriline E",  5: "Matriline F",
    6: "Matriline G",  7: "Matriline H",
}

# ═══════════════════════════════════════════════════════════════════════════
# Matriline assignment
# ═══════════════════════════════════════════════════════════════════════════

def get_matriline_ids(N: int, num_matrilines: int) -> np.ndarray:
    """
    Bucket N agents into num_matrilines families using cyclic (one-by-one)
    assignment, so this matches Dev C's MRI initializer exactly.

    The MRI initializer in `mri_init.build_matrilines` assigns agent i to
    matriline `i % num_matrilines` -- agents 0, 4, 8, ... share the rank-1
    matriline, agents 1, 5, 9, ... share rank-2, etc. (See its docstring:
    "Monkeys are assigned to matrilines in a cyclic (one-by-one) manner".)
    If this function disagrees with that, every visualization that colors
    by matriline silently lies about who's related to whom.

    Returns
    -------
    np.ndarray of shape (N,), dtype=int; entry i is agent i's matriline ID.
    """
    return np.array([i % num_matrilines for i in range(N)], dtype=int)

# ═══════════════════════════════════════════════════════════════════════════
# Agent color generation
# ═══════════════════════════════════════════════════════════════════════════

def _agent_color(matriline_id: int, total_in_matriline: int,
                 position_in_matriline: int) -> str:
    """
    Turn (matriline, position) into a single "rgb(r,g,b)" string.

    Agents within a matriline share a base hue but run from ~55% brightness
    (eldest) to 100% (youngest). The idea is "same family, same color family";
    you can scan a bump chart and instantly see if matriline C is clustered
    near the top vs scattered.
    """
    # Fall back to gray if someone asks for a matriline beyond the palette --
    # shouldn't happen at num_matrilines <= 8 but don't want to crash on it.
    base = MATRILINE_BASE_COLORS.get(
        matriline_id % len(MATRILINE_BASE_COLORS), (100, 100, 100)
    )
    if total_in_matriline > 1:
        frac = position_in_matriline / (total_in_matriline - 1)
    else:
        frac = 0.5
    # 0.55 lower bound keeps the darkest member visible against white bg.
    factor = 0.55 + 0.45 * frac
    r = min(255, int(base[0] * factor))
    g = min(255, int(base[1] * factor))
    b = min(255, int(base[2] * factor))
    return f"rgb({r},{g},{b})"


def build_color_map(N: int, num_matrilines: int,
                    matriline_ids: np.ndarray = None) -> dict:
    """
    Precompute {agent_index: "rgb(...)"} once so every trace on every figure
    uses identical colors.

    Building this once at notebook scope and passing it into every plotting call
    is cheaper than re-deriving inside each plot_* function, and -- more
    importantly -- keeps agent 7 the same shade of red in the bump chart and
    the centrality time series. Visual identity across panels is the whole point.
    """
    if matriline_ids is None:
        matriline_ids = get_matriline_ids(N, num_matrilines)

    colors = {}
    unique_mats = np.unique(matriline_ids)
    for mat_id in unique_mats:
        agents_in_mat = np.where(matriline_ids == mat_id)[0]
        for pos, agent in enumerate(agents_in_mat):
            colors[int(agent)] = _agent_color(
                int(mat_id), len(agents_in_mat), pos
            )
    return colors

# ═══════════════════════════════════════════════════════════════════════════
# Layout helpers (Plotly)
# ═══════════════════════════════════════════════════════════════════════════

def base_layout() -> dict:
    """
    Plotly layout defaults: font, white background, comfortable margins.

    Deliberately leaves xaxis/yaxis out so callers can merge this with their own
    axis dicts via fig.update_layout(**base_layout(), xaxis=..., yaxis=...)
    without Plotly complaining about duplicate keys.
    """
    return dict(
        font=dict(family=FONT_FAMILY, size=TICK_SIZE),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=30, t=60, b=60),
    )


def styled_axis() -> dict:
    """
    Standard axis styling (light grid, black border, mirrored top/right).

    Drop this straight into xaxis=dict(**styled_axis(), title=...). Every
    figure in the project looks the same because every figure uses this.
    """
    return dict(
        showgrid=True,
        gridcolor="rgba(200,200,200,0.4)",
        zeroline=False,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
    )


def add_event_lines(fig, events: list, y_min: float, y_max: float, row: int = None) -> None:
    """
    Drop a vertical dotted line at each overthrow timestamp.

    Used on both bump chart and centrality time series so swap events line up
    visually across panels. Non-overthrow events (e.g. perturbations) are
    ignored on purpose -- perturbations are usually pre-scheduled and the
    caller already knows when they fired.

    Parameters
    ----------
    row : int, optional
        Subplot row number for multi-panel figures. Needed so the shape gets
        anchored to the right axis pair -- without this, shapes drawn on a
        subplot all stack onto the top panel.
    """
    import plotly.graph_objects as go

    overthrow_events = [e for e in events if e.get("type") == "overthrow"]
    if not overthrow_events:
        return

    # Dedupe by time: if 3 agents overthrow in the same macro-step we only
    # want one line, not three on top of each other.
    unique_times = sorted({round(e["t"], 4) for e in overthrow_events})

    for et in unique_times:
        kwargs = dict(
            type="line",
            x0=et, x1=et,
            y0=y_min, y1=y_max,
            line=dict(color="rgba(0,0,0,0.25)", width=1, dash="dot"),
        )
        if row is not None:
            # Plotly names the first xaxis "x" (no suffix) and the rest "x2", "x3"...
            kwargs["xref"] = f"x{row}" if row > 1 else "x"
            kwargs["yref"] = f"y{row}" if row > 1 else "y"
        fig.add_shape(**kwargs)