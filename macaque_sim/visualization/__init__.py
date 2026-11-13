# Public viz API. This is what the notebook + tests import from.
#
# Keep the imports + __all__ in sync -- if you add a new plot_* function,
# remember to re-export it here, otherwise `from macaque_sim.visualization
# import ...` won't find it even though the module is on disk.

from macaque_sim.visualization.style import (
    MATRILINE_BASE_COLORS,
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
)
from macaque_sim.visualization.arena import (
    build_arena_animation,
    greedy_dyads,
    spring_positions,
)
from macaque_sim.visualization.bump_chart import plot_bump_chart
from macaque_sim.visualization.centrality_ts import plot_centrality, plot_combined
from macaque_sim.visualization.flow import build_flow_animation
from macaque_sim.visualization.phase_diagram import (
    plot_phase_diagram,
    plot_sweep,
    survival_grid_from_mc_results,
)
from macaque_sim.visualization.rank_animation import build_rank_animation
from macaque_sim.visualization.rank_heatmap import build_rank_heatmap

__all__ = [
    # Plots
    "plot_bump_chart",
    "plot_centrality",
    "plot_combined",
    "plot_phase_diagram",
    "plot_sweep",
    "survival_grid_from_mc_results",
    "build_rank_animation",
    "build_rank_heatmap",
    "build_arena_animation",
    "build_flow_animation",
    "greedy_dyads",
    "spring_positions",
    # Style helpers
    "build_color_map",
    "get_matriline_ids",
    "add_event_lines",
    # Style constants
    "MATRILINE_BASE_COLORS",
    "MATRILINE_NAMES",
    "FONT_FAMILY",
    "TITLE_SIZE",
    "AXIS_LABEL_SIZE",
    "TICK_SIZE",
    "LEGEND_SIZE",
    "FIGURE_WIDTH",
    "FIGURE_HEIGHT_SINGLE",
    "FIGURE_HEIGHT_COMBINED",
]
