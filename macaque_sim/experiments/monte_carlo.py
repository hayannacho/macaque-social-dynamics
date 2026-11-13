# monte_carlo.py — Batch runner with parameter grid expansion
# Assignee: Dev E (S2-T7 skeleton, S3-T3 wiring)
# See: project_management/tasks/sprint_2/S2-T7_mc_runner_skeleton.md
# See: project_management/tasks/sprint_3/S3-T3_scenario_controller.md


"""
Implements:
  - parameter sweep grid expansion (supports multiple sweep dimensions),
  - per-cell Monte Carlo loops with per-run seed overrides,
  - metric extraction from a logger-like object returned by `sim_fn`,
  - aggregated experiment result container that is indexable by grid coordinates.

AI Usage: Asked AI to generate docstrings and comments for the code. After completing logic, also asked for it to refactor.
Also asked for it to check for cohesion with the logger to make sure that the parameters are correct. 
Asked AI to check for edge cases and potential bugs.

"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from macaque_sim.config.scenarios import (
    scenario_alpha_removal,
    scenario_decay,
    scenario_grooming_bias,
    scenario_noise,
)
from macaque_sim.config.sim_config import default_config


@dataclass
class TrialResult:
    """
    Per-run Monte Carlo metrics for one simulation trial.

    This keeps a typed interface while the Sprint 2 runner still accepts
    logger-like payloads from `sim_fn` and converts them here.
    """

    alpha_survived: bool
    num_swaps: int
    final_ranks: list[int] = field(default_factory=list)
    raw_logger: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_survived": self.alpha_survived,
            "num_swaps": self.num_swaps,
            "final_ranks": list(self.final_ranks),
        }


@dataclass
class ExperimentCell:
    """Results for one parameter-grid coordinate."""

    coords: tuple[int, ...]
    param_values: dict[str, Any]
    runs: list[TrialResult] = field(default_factory=list)

    @property
    def alpha_survival_rate(self) -> float:
        if not self.runs:
            return float("nan")
        return float(np.mean([r.alpha_survived for r in self.runs]))

    @property
    def mean_num_swaps(self) -> float:
        if not self.runs:
            return float("nan")
        return float(np.mean([r.num_swaps for r in self.runs]))

    @property
    def run_dicts(self) -> list[dict[str, Any]]:
        """Dict view for downstream callers expecting primitive payloads."""
        return [r.to_dict() for r in self.runs]


@dataclass
class ExperimentResults:
    """Top-level Monte Carlo output for one scenario."""

    scenario_name: str
    sweep_keys: list[str]
    sweep_values: dict[str, list[Any]]
    shape: tuple[int, ...]
    cells: dict[tuple[int, ...], ExperimentCell]
    elapsed_seconds: float = 0.0

    def __getitem__(self, coords: tuple[int, ...]) -> ExperimentCell:
        return self.cells[coords]

    def __len__(self) -> int:
        return len(self.cells)


def _is_sweep_value(value: Any) -> bool:
    if not isinstance(value, (tuple, list)):
        return False
    if len(value) == 3:
        start, stop, num_steps = value
        return isinstance(start, (int, float)) and isinstance(stop, (int, float)) and isinstance(num_steps, int)
    return len(value) >= 2 and all(isinstance(v, str) for v in value)


def expand_param_grid(
    param_overrides: dict[str, Any],
) -> tuple[list[str], dict[str, list[Any]], list[tuple[tuple[int, ...], dict[str, Any]]]]:
    """
    Expand scenario overrides into a Cartesian parameter grid.

    Returns:
      - sweep_keys: ordered sweep parameter names,
      - sweep_values: concrete values for each sweep key,
      - grid_cells: list[(coords, params)] where coords is an index tuple.
    """
    fixed_values = {k: v for k, v in param_overrides.items() if not _is_sweep_value(v)}
    sweep_keys = [k for k, v in param_overrides.items() if _is_sweep_value(v)]

    sweep_values: dict[str, list[Any]] = {}
    for key in sweep_keys:
        raw = param_overrides[key]
        if len(raw) == 3 and isinstance(raw[2], int) and isinstance(raw[0], (int, float)) and isinstance(raw[1], (int, float)):
            start, stop, num_steps = raw
            sweep_values[key] = list(np.linspace(start, stop, int(num_steps)))
        else:
            sweep_values[key] = list(raw)

    if not sweep_keys:
        return [], {}, [((0,), dict(fixed_values))]

    index_ranges = [range(len(sweep_values[k])) for k in sweep_keys]
    grid_cells: list[tuple[tuple[int, ...], dict[str, Any]]] = []
    for coords in itertools.product(*index_ranges):
        params = dict(fixed_values)
        for axis, key in enumerate(sweep_keys):
            params[key] = sweep_values[key][coords[axis]]
        grid_cells.append((coords, params))

    return sweep_keys, sweep_values, grid_cells


def _read_metric(logger: Any, key: str, default: Any = None) -> Any:
    if isinstance(logger, dict):
        return logger.get(key, default)
    return getattr(logger, key, default)


def _extract_final_ranks(logger: Any) -> list[int]:
    """Get final rank vector from logger-like objects."""
    direct = _read_metric(logger, "final_ranks", None)
    if direct is not None:
        return list(direct)

    rank_hist = _read_metric(logger, "R", None)
    step_idx = int(_read_metric(logger, "step_idx", 0) or 0)
    if rank_hist is not None and step_idx > 0:
        return list(np.asarray(rank_hist[step_idx - 1]).astype(int))
    return []


def _extract_num_swaps(logger: Any) -> int:
    """Get swap count from logger-like objects."""
    direct = _read_metric(logger, "num_swaps", None)
    if direct is not None:
        return int(direct)

    swap_events = _read_metric(logger, "swap_events", None)
    if swap_events is None and hasattr(logger, "get_events"):
        swap_events = logger.get_events()
    if swap_events is None:
        return 0
    return int(len(swap_events))


def _extract_alpha_survived(logger: Any, final_ranks: list[int]) -> bool:
    """
    Infer alpha survival.

    Priority:
      1) explicit field `alpha_survived` if provided by sim_fn output
         (run_simulation sets this via `R[initial_alpha] == 1` where alpha
         is the agent with rank 1 -- see macaque_sim/main.py and the
         1-indexed convention in initializer/mri_init.py).
      2) fallback: infer from final_ranks under 1-indexed convention
         (min rank == 1 means someone still holds alpha; this does NOT
         identify *which* agent held it originally, so this fallback is
         a lower bound on the true survival question and is only used
         when no logger-level field was populated).
    """
    direct = _read_metric(logger, "alpha_survived", None)
    if direct is not None:
        return bool(direct)

    if not final_ranks:
        return False
    return int(min(final_ranks)) == 1


def run_social_noise_experiment(sim_fn: Callable[[dict[str, Any]], Any], num_mc_runs: int = 100) -> ExperimentResults:
    """Run experiment 1: social-noise sigma sweep."""
    return run_experiment(scenario_noise(num_mc_runs=num_mc_runs), sim_fn)


def run_grooming_bias_experiment(sim_fn: Callable[[dict[str, Any]], Any], num_mc_runs: int = 100) -> ExperimentResults:
    """Run experiment 2: grooming-strategy sweep."""
    return run_experiment(scenario_grooming_bias(num_mc_runs=num_mc_runs), sim_fn)


def run_aristocratic_decay_experiment(sim_fn: Callable[[dict[str, Any]], Any], num_mc_runs: int = 100) -> ExperimentResults:
    """Run experiment 3: lambda-decay sweep."""
    return run_experiment(scenario_decay(num_mc_runs=num_mc_runs), sim_fn)


def run_alpha_removal_experiment(sim_fn: Callable[[dict[str, Any]], Any], num_mc_runs: int = 100) -> ExperimentResults:
    """Run experiment 4: alpha-removal perturbation."""
    return run_experiment(scenario_alpha_removal(num_mc_runs=num_mc_runs), sim_fn)


def _extract_run_metrics(logger: Any) -> TrialResult:
    """
    Convert a logger-like object into Sprint 2 metrics.

    Expected fields (dict key or attribute):
      - alpha_survived: bool
      - num_swaps: int
      - final_ranks: sequence[int]
    """
    final_ranks = _extract_final_ranks(logger)
    num_swaps = _extract_num_swaps(logger)
    alpha_survived = _extract_alpha_survived(logger, final_ranks)
    return TrialResult(
        alpha_survived=alpha_survived,
        num_swaps=num_swaps,
        final_ranks=list(final_ranks),
        raw_logger=logger,
    )


def run_experiment(
    scenario,
    sim_fn: Callable[[dict[str, Any]], Any],
    progress: bool = False,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
) -> ExperimentResults:
    """
    Execute Monte Carlo sweep over the scenario grid.

    If ``checkpoint_path`` is set, writes JSON after each completed grid cell
    (atomic replace) so crashes lose at most one in-progress cell. With
    ``resume=True``, completed cells are skipped on restart.
    """
    from macaque_sim.experiments.results_io import load_experiment_results, save_experiment_results

    base_config = default_config().to_dict()
    t_start = time.perf_counter()
    sweep_keys, sweep_values, grid_cells = expand_param_grid(scenario.param_overrides)

    if sweep_keys:
        shape = tuple(len(sweep_values[key]) for key in sweep_keys)
    else:
        shape = (1,)

    expected_coord_set = {coords for coords, _ in grid_cells}
    cells: dict[tuple[int, ...], ExperimentCell] = {}

    if checkpoint_path is not None and resume:
        cp = Path(checkpoint_path)
        if cp.is_file():
            try:
                prev = load_experiment_results(cp)
            except (json.JSONDecodeError, OSError, ValueError):
                prev = None
            else:
                if prev.scenario_name == scenario.name and not (set(prev.cells) - expected_coord_set):
                    cells = {k: v for k, v in prev.cells.items() if k in expected_coord_set}

    base_seed = int(base_config.get("random_seed", 42))

    for cell_idx, (coords, param_values) in enumerate(grid_cells):
        existing = cells.get(coords)
        if existing is not None and len(existing.runs) >= scenario.num_mc_runs:
            if progress:
                print(f"[{scenario.name}] cell {cell_idx + 1}/{len(grid_cells)} (resume skip) {param_values}")
            continue

        merged_config = {**base_config, **param_values}
        merged_config["perturbations"] = scenario.perturbations

        if progress:
            print(f"[{scenario.name}] cell {cell_idx + 1}/{len(grid_cells)} {param_values}")

        run_results: list[TrialResult] = []
        if existing is not None:
            run_results = list(existing.runs)

        # Offset by cell index so different grid points do not reuse the same RNG stream.
        cell_seed_base = base_seed + cell_idx * 100_000
        for run_i in range(len(run_results), scenario.num_mc_runs):
            run_config = dict(merged_config)
            run_config["random_seed"] = cell_seed_base + run_i
            logger = sim_fn(run_config)
            run_results.append(_extract_run_metrics(logger))

        cells[coords] = ExperimentCell(
            coords=coords,
            param_values=param_values,
            runs=run_results,
        )

        if checkpoint_path is not None:
            snapshot = ExperimentResults(
                scenario_name=scenario.name,
                sweep_keys=sweep_keys,
                sweep_values=sweep_values,
                shape=shape,
                cells=dict(cells),
                elapsed_seconds=time.perf_counter() - t_start,
            )
            save_experiment_results(checkpoint_path, snapshot)

    return ExperimentResults(
        scenario_name=scenario.name,
        sweep_keys=sweep_keys,
        sweep_values=sweep_values,
        shape=shape,
        cells=cells,
        elapsed_seconds=time.perf_counter() - t_start,
    )