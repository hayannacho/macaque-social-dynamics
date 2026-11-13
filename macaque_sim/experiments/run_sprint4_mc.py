# run_sprint4_mc.py — S4-T1 production Monte Carlo sweeps (checkpointed JSON)
"""Production Monte Carlo driver for Sprint 4 task S4-T1 (Execute Monte Carlo Sweeps).

This module is the **team-facing entry point** for running the four predefined
experiments at scale: it is not a unit test (see ``tests/test_run_sprint4_mc.py``).

What it does
------------
1. For each experiment, builds an :class:`~macaque_sim.config.scenarios.ExperimentScenario`
   via the factory functions in ``macaque_sim.config.scenarios`` (noise σ sweep,
   grooming strategy comparison, λ decay sweep, alpha removal with mid-sim
   perturbation).
2. Calls :func:`~macaque_sim.experiments.monte_carlo.run_experiment` with
   :func:`~macaque_sim.main.run_single_trial` as ``sim_fn`` so every grid cell runs
   the **full integrated pipeline** (same code path as ``run_simulation``).
3. Persists **checkpointed** JSON after each completed grid cell (atomic write via
   ``results_io``) so a crash or laptop sleep loses at most one in-progress cell;
   with ``resume=True`` (the default), restarting the same command skips finished
   cells and continues.

Artifacts written
-------------------
All paths are relative to ``--output-dir`` (default: ``output/results/``), which
should stay git-ignored (see repo ``.gitignore``).

* ``noise_sweep.json``, ``grooming_bias.json``, ``decay_sweep.json``,
  ``alpha_removal.json`` — serialized :class:`~macaque_sim.experiments.monte_carlo.ExperimentResults`
  (format version 1, no raw loggers embedded).
* ``sprint4_runtime.json`` — wall-clock seconds per experiment, total hours,
  ``within_budget`` vs ``--budget-hours`` (default 8), and grid cell counts.

After a full run (or after ``--only`` subset), the driver invokes verification:
each file must load, match the expected scenario name, contain the full parameter
grid (expected cell count), and have exactly ``num_mc_runs`` trials per cell.

CLI (run from repository root)
------------------------------
.. code-block:: text

   python -m macaque_sim.experiments.run_sprint4_mc --output-dir output/results --num-mc-runs 50

Smoke a single experiment with few MC repeats:

.. code-block:: text

   python -m macaque_sim.experiments.run_sprint4_mc --only noise_sweep --num-mc-runs 2

Re-validate existing JSON on disk without running the simulator:

.. code-block:: text

   python -m macaque_sim.experiments.run_sprint4_mc --verify-only --output-dir output/results --num-mc-runs 50

See also
--------
* ``project_management/tasks/sprint_4/S4-T1_mc_sweeps.md`` — acceptance criteria.
* ``macaque_sim.experiments.monte_carlo`` — grid expansion and MC loop.
* ``macaque_sim.experiments.results_io`` — JSON schema, load/save, ``verify_experiment_file``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from macaque_sim.config.scenarios import (
    ExperimentScenario,
    scenario_alpha_removal,
    scenario_decay,
    scenario_grooming_bias,
    scenario_noise,
    scenario_phase,
    scenario_volatility_spike,
)
from macaque_sim.experiments.monte_carlo import expand_param_grid, run_experiment
from macaque_sim.experiments.results_io import verify_experiment_file
from macaque_sim.main import run_single_trial

_JOBS: list[tuple[str, Callable[[int], ExperimentScenario], str]] = [
    ("noise_sweep", scenario_noise, "noise_sweep.json"),
    ("grooming_bias", scenario_grooming_bias, "grooming_bias.json"),
    ("decay_sweep", scenario_decay, "decay_sweep.json"),
    ("alpha_removal", scenario_alpha_removal, "alpha_removal.json"),
    ("phase_sweep", scenario_phase, "phase_sweep.json"),
    ("volatility_spike", scenario_volatility_spike, "volatility_spike.json"),
]


def run_all(
    *,
    output_dir: Path,
    num_mc_runs: int,
    progress: bool,
    resume: bool,
    jobs: list[tuple[str, Callable[[int], ExperimentScenario], str]] | None = None,
    budget_wall_clock_hours: float = 8.0,
) -> dict:
    """Run experiment jobs; return a runtime summary dict (S4-T1)."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "sprint4_runtime.json"
    job_list = jobs if jobs is not None else _JOBS

    wall_start = time.perf_counter()
    entries: list[dict] = []

    for scenario_key, factory, filename in job_list:
        scenario: ExperimentScenario = factory(num_mc_runs=num_mc_runs)
        out_path = output_dir / filename
        t0 = time.perf_counter()
        result = run_experiment(
            scenario,
            run_single_trial,
            progress=progress,
            checkpoint_path=out_path,
            resume=resume,
        )
        elapsed = time.perf_counter() - t0
        num_grid_cells = len(result.cells)
        entries.append(
            {
                "scenario_key": scenario_key,
                "result_file": str(out_path),
                "wall_clock_seconds": round(elapsed, 3),
                "num_mc_runs_per_cell": num_mc_runs,
                "num_grid_cells": num_grid_cells,
            }
        )
        if progress:
            print(f"Done {scenario_key}: {elapsed:.1f}s ({num_grid_cells} cells × {num_mc_runs} MC) -> {out_path}")

    total = time.perf_counter() - wall_start
    budget_seconds = max(0.0, float(budget_wall_clock_hours)) * 3600.0
    summary = {
        "sprint_task": "S4-T1",
        "total_wall_clock_seconds": round(total, 3),
        "total_wall_clock_hours": round(total / 3600.0, 4),
        "budget_wall_clock_hours": float(budget_wall_clock_hours),
        "within_budget": total <= budget_seconds if budget_seconds > 0 else None,
        "budget_note": "S4-T1: <= 8h wall-clock total across all four experiments on team laptops.",
        "num_mc_runs_per_cell": num_mc_runs,
        "experiments": entries,
        "num_grid_cells_total": sum(e["num_grid_cells"] for e in entries),
    }
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if progress:
        print(f"Total wall time: {total / 3600:.3f} h (logged to {log_path})")
        if summary["within_budget"] is False:
            print(
                f"WARNING: total {total / 3600:.2f} h exceeds budget {budget_wall_clock_hours} h.",
                file=sys.stderr,
            )
    return summary


def verify_all(
    output_dir: Path,
    num_mc_runs: int,
    jobs: list[tuple[str, Callable[[int], ExperimentScenario], str]] | None = None,
) -> None:
    output_dir = output_dir.resolve()
    job_list = jobs if jobs is not None else _JOBS
    for scenario_key, factory, fname in job_list:
        scenario = factory(num_mc_runs=num_mc_runs)
        _, _, grid_cells = expand_param_grid(scenario.param_overrides)
        verify_experiment_file(
            output_dir / fname,
            expected_scenario_name=scenario_key,
            expected_num_mc_runs=num_mc_runs,
            expected_num_cells=len(grid_cells),
        )


def _select_jobs(only: list[str] | None) -> list[tuple[str, Callable[[int], ExperimentScenario], str]] | None:
    if not only:
        return None
    allowed = {k for k, _, _ in _JOBS}
    bad = sorted(set(only) - allowed)
    if bad:
        raise ValueError(f"Unknown --only keys {bad}; allowed: {sorted(allowed)}")
    want = set(only)
    return [row for row in _JOBS if row[0] in want]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S4-T1: production Monte Carlo sweeps (checkpointed JSON under output/results/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/results"),
        help="Directory for JSON results and sprint4_runtime.json",
    )
    parser.add_argument(
        "--num-mc-runs",
        type=int,
        default=50,
        help="Monte Carlo repetitions per grid cell (S4-T1 default 50).",
    )
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=8.0,
        help="Laptop budget (hours) for within_budget flag in sprint4_runtime.json.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="KEY",
        help=(
            "Run a subset of experiments only (repeatable). "
            "Keys: noise_sweep, grooming_bias, decay_sweep, alpha_removal, phase_sweep."
        ),
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoint files.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only load and verify result JSON files (no simulation).",
    )
    args = parser.parse_args(argv)

    if args.num_mc_runs < 1:
        print("num_mc_runs must be >= 1", file=sys.stderr)
        return 2

    try:
        jobs = _select_jobs(args.only if args.only else None)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    if args.verify_only:
        try:
            verify_all(args.output_dir, args.num_mc_runs, jobs=jobs)
        except (FileNotFoundError, ValueError) as e:
            print(f"Verification failed: {e}", file=sys.stderr)
            return 1
        print("All result files load, grid is complete, and each cell has the expected MC depth.")
        return 0

    run_all(
        output_dir=args.output_dir,
        num_mc_runs=args.num_mc_runs,
        progress=not args.quiet,
        resume=not args.no_resume,
        jobs=jobs,
        budget_wall_clock_hours=args.budget_hours,
    )

    try:
        verify_all(args.output_dir, args.num_mc_runs, jobs=jobs)
    except (FileNotFoundError, ValueError) as e:
        print(f"Post-run verification failed: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
