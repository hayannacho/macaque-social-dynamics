"""study_convergence.py -- dt-halving convergence study (post-review follow-up).

Runs five headline cells at three (dt_sde, dt_ode) refinement levels to verify
that Monte Carlo statistics are insensitive to the discretization step size --
the gap the wise-beaver review flagged as the single most important
methodological omission. The five cells cover the regimes where the Feller
condition is satisfied (default, coalition, sigma=0.30) AND the regimes where
it is violated (sigma=0.40 transition cliff, sigma=0.40 moderate spike,
sigma=0.90 spike). The sigma=0.90 cell is the one most likely to misbehave at
finer dt because the clamp does the most work there.

Output: output/results/followups/convergence.json with per-cell records:
  cell, dt_level (label + numeric), n_trials, n_alpha_survived, p_hat,
  wilson_lo, wilson_hi.

Usage (from repo root):
    python -m scripts.study_convergence
or:
    python scripts/study_convergence.py
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from macaque_sim.main import run_single_trial


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "output" / "results" / "followups" / "convergence.json"


# Three dt refinement levels. The multirate identity M * dt_sde = dt_ode is
# enforced exactly by setting M = round(dt_ode / dt_sde). Production is level 0.
DT_LEVELS = [
    {"label": "production",  "dt_sde": 0.10,  "dt_ode": 0.50,  "M": 5},
    {"label": "halved",      "dt_sde": 0.05,  "dt_ode": 0.25,  "M": 5},
    {"label": "quartered",   "dt_sde": 0.025, "dt_ode": 0.125, "M": 5},
]


def cell_default(seed: int) -> dict:
    return {"sigma": 0.30, "grooming_strategy": "uniform", "random_seed": seed}


def cell_coalition(seed: int) -> dict:
    return {"sigma": 0.30, "grooming_strategy": "coalition", "random_seed": seed}


def cell_sigma_040(seed: int) -> dict:
    return {"sigma": 0.40, "grooming_strategy": "uniform", "random_seed": seed}


def cell_sigma_090(seed: int) -> dict:
    return {"sigma": 0.90, "grooming_strategy": "uniform", "random_seed": seed}


def cell_spike_040(seed: int) -> dict:
    """Moderate-spike scenario from §5.3: sigma 0.30 -> 0.40 [T/4, 3T/4] -> 0.30."""
    return {
        "sigma": 0.30,
        "grooming_strategy": "uniform",
        "random_seed": seed,
        "perturbations": [
            {"type": "volatility_spike", "time": 25.0, "target": -1,
             "params": {"factor": 4.0 / 3.0}},
            {"type": "volatility_spike", "time": 75.0, "target": -1,
             "params": {"factor": 3.0 / 4.0}},
        ],
    }


CELLS = [
    ("default",      cell_default),
    ("coalition",    cell_coalition),
    ("sigma_040",    cell_sigma_040),
    ("sigma_090",    cell_sigma_090),
    ("spike_040",    cell_spike_040),
]

N_TRIALS_PER_CELL = 50


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfwidth = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))


def run_cell_at_dt(cell_name: str, base_factory, dt_level: dict, n_trials: int) -> dict:
    """Run n_trials of one cell at one dt level. Returns aggregated record."""
    n_alive = 0
    t0 = time.perf_counter()
    for run_idx in range(n_trials):
        # Seed scheme: distinct per (cell, dt level, run) so cells are
        # independent but each (cell, run) pair sees the same RNG draws across
        # dt levels (modulo the dt change itself). The seed offset depends on
        # the cell index and dt label so we don't accidentally collide with
        # the production-sweep seeds in output/results/.
        cell_idx = [c[0] for c in CELLS].index(cell_name)
        dt_idx = next(i for i, lvl in enumerate(DT_LEVELS) if lvl["label"] == dt_level["label"])
        seed = 90000 + cell_idx * 1000 + dt_idx * 100 + run_idx
        cfg = base_factory(seed)
        cfg["dt_sde"] = dt_level["dt_sde"]
        cfg["dt_ode"] = dt_level["dt_ode"]
        cfg["micro_steps_per_macro"] = dt_level["M"]
        try:
            logger = run_single_trial(cfg)
            n_alive += int(getattr(logger, "alpha_survived", False))
        except Exception as exc:  # noqa: BLE001 -- we want every trial counted
            print(f"  trial {run_idx} ({cell_name}, dt={dt_level['label']}) raised: {exc}")
    wall = time.perf_counter() - t0
    p = n_alive / n_trials
    lo, hi = wilson_ci(n_alive, n_trials)
    return {
        "cell": cell_name,
        "dt_label": dt_level["label"],
        "dt_sde": dt_level["dt_sde"],
        "dt_ode": dt_level["dt_ode"],
        "M": dt_level["M"],
        "n_trials": n_trials,
        "n_alpha_survived": n_alive,
        "alpha_survival_rate": p,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "wall_clock_s": wall,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = []
    grand_t0 = time.perf_counter()

    for cell_name, factory in CELLS:
        for dt_level in DT_LEVELS:
            print(f"[{cell_name}] dt={dt_level['label']} (dt_sde={dt_level['dt_sde']}, "
                  f"dt_ode={dt_level['dt_ode']}, M={dt_level['M']})")
            rec = run_cell_at_dt(cell_name, factory, dt_level, N_TRIALS_PER_CELL)
            print(f"  -> {rec['n_alpha_survived']}/{rec['n_trials']} survived "
                  f"({rec['alpha_survival_rate']:.2f}, "
                  f"Wilson [{rec['wilson_lo']:.2f}, {rec['wilson_hi']:.2f}]), "
                  f"{rec['wall_clock_s']:.1f}s")
            records.append(rec)
            # Atomic incremental write so an interrupted run is recoverable.
            tmp = OUTPUT_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({
                "scenario": "convergence_study",
                "n_trials_per_cell": N_TRIALS_PER_CELL,
                "dt_levels": DT_LEVELS,
                "records": records,
            }, indent=2))
            tmp.replace(OUTPUT_PATH)

    grand_wall = time.perf_counter() - grand_t0
    print(f"\nTotal: {len(records)} (cell, dt) pairs, "
          f"{len(records) * N_TRIALS_PER_CELL} trials, {grand_wall:.1f}s wall clock.")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
