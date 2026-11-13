# hysteresis_experiment.py — Ramp-up/ramp-down σ sweep to expose path dependence.
"""
Hysteresis experiment: ramp σ up from σ_min to σ_max, then back down, inside
a SINGLE continuous trajectory (state threaded across segments). Monte Carlo
over independent trajectories. If forward and backward survival curves don't
overlap at the same σ, the system is path-dependent -- hysteresis.

The core claim of the project is bistability driven by alliance memory + the
hysteresis margin (ε) in the overthrow engine. None of the four baseline
scenarios tests that claim directly; each of them re-initializes at every
trial so you only ever see the forward transition. This module closes that
gap.

Design rationale
----------------
- Single trajectory per MC trial: σ is held constant within a segment of
  length ``segment_T``, then stepped to the next value; (A, P, R) from the
  end of segment k become the initial state of segment k+1. This lets the
  state carry memory across σ changes, which is precisely what hysteresis
  requires.
- Forward and backward legs use the SAME σ values (reversed); we compare
  P(alpha holds | σ, forward leg) vs P(alpha holds | σ, backward leg).
  A persistent gap means the system didn't relax to the σ-conditional
  steady state -- it remembered where it came from.
- segment_T is the knob: too short and you're measuring "the system is
  slow to track σ"; too long and every segment equilibrates and the two
  legs collapse onto the same curve. Default 8.0 (≈ 1/12th of the default
  T=100) sits in the moderate-ramp-rate regime where hysteresis is
  physically meaningful.

Output
------
JSON at ``output/results/hysteresis_sweep.json`` (format defined below).
Consumed by visualization/hysteresis.py and generate_figures.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from macaque_sim.config.sim_config import default_config
from macaque_sim.main import run_simulation


def run_hysteresis_trial(
    base_config: dict[str, Any],
    sigma_forward: np.ndarray,
    sigma_backward: np.ndarray,
    segment_T: float,
    trial_seed: int,
) -> dict[str, Any]:
    """
    Run one hysteresis trajectory (forward ramp then backward ramp).

    The first segment uses MRI initialization; every subsequent segment
    reuses the previous segment's final (A, P, R) as initial_state so
    alliance memory persists across σ changes.

    Parameters
    ----------
    base_config
        Dict of SimConfig overrides. ``T``, ``sigma``, ``random_seed`` are
        set per-segment and will be overwritten.
    sigma_forward
        1-D array of σ values on the up-ramp (low → high).
    sigma_backward
        1-D array of σ values on the down-ramp (high → low). Typically
        sigma_forward[::-1] but kept separate for flexibility.
    segment_T
        Simulation time per segment (same units as cfg.T).
    trial_seed
        Base RNG seed; each segment gets trial_seed + segment_index to
        avoid RNG stream collisions.

    Returns
    -------
    dict with:
      "alpha_held_forward"  : list[bool], len == len(sigma_forward)
      "alpha_held_backward" : list[bool], len == len(sigma_backward)

    The booleans record whether the MRI-assigned alpha still holds rank 1
    at the END of that segment (not the average across the segment).
    """
    cfg_dict = dict(base_config)
    cfg_dict["T"] = float(segment_T)

    # --- Segment 1: MRI init, sigma = sigma_forward[0] ---
    cfg_dict["sigma"] = float(sigma_forward[0])
    cfg_dict["random_seed"] = int(trial_seed)
    logger = run_simulation(cfg_dict)

    # Identify the MRI-alpha (agent with rank 1 at t=0 of this trial).
    # Rank convention is 1-indexed (see initializer/mri_init.py); np.argmin
    # returns the index of the rank-1 agent.
    R0 = logger.R[0]
    initial_alpha = int(np.argmin(R0))

    def _tail_state(lg):
        last = lg.step_idx - 1
        return (lg.A[last].copy(), lg.P[last].copy(), lg.R[last].copy())

    A_end, P_end, R_end = _tail_state(logger)
    alpha_held_forward = [bool(R_end[initial_alpha] == 1)]

    # --- Remaining forward segments ---
    for seg_idx, sigma_val in enumerate(sigma_forward[1:], start=1):
        cfg_dict["sigma"] = float(sigma_val)
        cfg_dict["random_seed"] = int(trial_seed + seg_idx)
        logger = run_simulation(cfg_dict, initial_state=(A_end, P_end, R_end))
        A_end, P_end, R_end = _tail_state(logger)
        alpha_held_forward.append(bool(R_end[initial_alpha] == 1))

    # --- Backward segments ---
    alpha_held_backward: list[bool] = []
    offset = len(sigma_forward)
    for seg_idx, sigma_val in enumerate(sigma_backward):
        cfg_dict["sigma"] = float(sigma_val)
        cfg_dict["random_seed"] = int(trial_seed + offset + seg_idx)
        logger = run_simulation(cfg_dict, initial_state=(A_end, P_end, R_end))
        A_end, P_end, R_end = _tail_state(logger)
        alpha_held_backward.append(bool(R_end[initial_alpha] == 1))

    return {
        "alpha_held_forward": alpha_held_forward,
        "alpha_held_backward": alpha_held_backward,
    }


def run_hysteresis_mc(
    base_config: dict[str, Any] | None = None,
    num_mc_runs: int = 50,
    sigma_min: float = 0.10,
    sigma_max: float = 0.60,
    n_steps: int = 12,
    segment_T: float = 8.0,
    base_seed: int = 42,
    progress: bool = True,
) -> dict[str, Any]:
    """
    Monte Carlo aggregate over hysteresis trajectories.

    Each trial is a full forward+backward ramp; the MC average at each σ
    level gives P(alpha holds rank 1 at this σ | ramp direction).

    Returns a dict matching the JSON schema expected by the hysteresis viz
    and the verification pass in generate_figures.py.
    """
    if base_config is None:
        base_config = default_config().to_dict()

    sigma_forward = np.linspace(sigma_min, sigma_max, n_steps)
    sigma_backward = sigma_forward[::-1].copy()

    forward_accum = np.zeros(n_steps, dtype=int)
    backward_accum = np.zeros(n_steps, dtype=int)

    t0 = time.perf_counter()
    # Seed stride large enough that per-segment seeds within a trial never
    # collide with per-segment seeds in adjacent trials.
    seed_stride = 2 * n_steps + 10

    for trial_i in range(num_mc_runs):
        trial_seed = int(base_seed + trial_i * seed_stride)
        result = run_hysteresis_trial(
            base_config=base_config,
            sigma_forward=sigma_forward,
            sigma_backward=sigma_backward,
            segment_T=segment_T,
            trial_seed=trial_seed,
        )
        forward_accum += np.array(result["alpha_held_forward"], dtype=int)
        backward_accum += np.array(result["alpha_held_backward"], dtype=int)

        if progress and ((trial_i + 1) % max(1, num_mc_runs // 10) == 0):
            elapsed = time.perf_counter() - t0
            rate = (trial_i + 1) / elapsed
            print(f"  hysteresis MC {trial_i + 1}/{num_mc_runs}  ({rate:.2f} trials/s)")

    elapsed = time.perf_counter() - t0
    return {
        "format_version": 1,
        "scenario_name": "hysteresis_sweep",
        "description": (
            f"σ ramp-up/ramp-down hysteresis: {n_steps} steps from "
            f"{sigma_min:.3f} to {sigma_max:.3f} then back, segment_T={segment_T}, "
            f"{num_mc_runs} MC trajectories."
        ),
        "sigma_forward": sigma_forward.tolist(),
        "sigma_backward": sigma_backward.tolist(),
        "forward_survival": (forward_accum / num_mc_runs).tolist(),
        "backward_survival": (backward_accum / num_mc_runs).tolist(),
        "num_mc_runs": int(num_mc_runs),
        "segment_T": float(segment_T),
        "n_steps": int(n_steps),
        "elapsed_seconds": round(elapsed, 3),
    }


def save_hysteresis_results(path: str | Path, results: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def load_hysteresis_results(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run σ ramp-up/ramp-down hysteresis experiment.",
    )
    parser.add_argument("--output", type=Path, default=Path("output/results/hysteresis_sweep.json"))
    parser.add_argument("--num-mc-runs", type=int, default=50)
    parser.add_argument("--sigma-min", type=float, default=0.10)
    parser.add_argument("--sigma-max", type=float, default=0.60)
    parser.add_argument("--n-steps", type=int, default=12)
    parser.add_argument("--segment-T", type=float, default=8.0)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    results = run_hysteresis_mc(
        num_mc_runs=args.num_mc_runs,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        n_steps=args.n_steps,
        segment_T=args.segment_T,
        base_seed=args.base_seed,
        progress=not args.quiet,
    )
    save_hysteresis_results(args.output, results)
    if not args.quiet:
        print(f"Saved {args.output} ({results['elapsed_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main_cli())
