"""study_adaptive_epsilon.py -- adaptive-epsilon grooming-bias experiment (post-review C1).

Reviewer's "single most actionable fix": calibrate epsilon adaptively as
2 * running std(dP) over the last K=20 macro-steps. If the 88% vs 43%
grooming-bias gap survives under adaptive epsilon, the headline finding is
robust to the calibration choice; if it collapses, the finding is contingent
on the production fixed epsilon=0.02 calibration and should be reframed
accordingly.

Runs both fixed (epsilon_mode='fixed', epsilon=0.02) and adaptive
(epsilon_mode='adaptive', K=20, factor=2) on all three grooming strategies
{uniform, upward_biased, coalition} at default sigma=0.30. 100 trials per
cell, 6 cells, 600 trials total.

Output: output/results/followups/adaptive_epsilon_grooming.json
"""
from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

from macaque_sim.main import run_single_trial


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "output" / "results" / "followups" / "adaptive_epsilon_grooming.json"

GROOMING_STRATEGIES = ["uniform", "upward_biased", "coalition"]
EPSILON_MODES = ["fixed", "adaptive"]
N_TRIALS = 100
EPSILON_WINDOW = 20
EPSILON_FACTOR = 2.0


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfwidth = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = []
    grand_t0 = time.perf_counter()

    for strat_idx, strat in enumerate(GROOMING_STRATEGIES):
        for mode_idx, mode in enumerate(EPSILON_MODES):
            n_alive = 0
            n_swaps_total = 0
            t0 = time.perf_counter()
            for run_idx in range(N_TRIALS):
                # Seed offset 60000 keeps these distinct from production
                # grooming_bias seeds AND from the other follow-up sweeps
                # (study_convergence at 90000+, study_coupling at 80000+,
                # study_cooldown at 70000+).
                seed = 60000 + strat_idx * 10000 + mode_idx * 1000 + run_idx
                cfg = {
                    "sigma": 0.30,
                    "grooming_strategy": strat,
                    "random_seed": seed,
                    "epsilon_mode": mode,
                    "epsilon_window": EPSILON_WINDOW,
                    "epsilon_factor": EPSILON_FACTOR,
                }
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        logger = run_single_trial(cfg)
                        n_alive += int(getattr(logger, "alpha_survived", False))
                        n_swaps_total += int(getattr(logger, "num_swaps", 0))
                    except Exception as exc:  # noqa: BLE001
                        print(f"  trial {run_idx} ({strat}, {mode}) raised: {exc}")
            wall = time.perf_counter() - t0
            p = n_alive / N_TRIALS
            lo, hi = wilson_ci(n_alive, N_TRIALS)
            rec = {
                "grooming_strategy": strat,
                "epsilon_mode": mode,
                "epsilon_window": EPSILON_WINDOW,
                "epsilon_factor": EPSILON_FACTOR,
                "n_trials": N_TRIALS,
                "n_alpha_survived": n_alive,
                "alpha_survival_rate": p,
                "wilson_lo": lo,
                "wilson_hi": hi,
                "mean_num_swaps": n_swaps_total / N_TRIALS,
                "wall_clock_s": wall,
            }
            records.append(rec)
            print(f"strat={strat:<14} mode={mode:<8}  "
                  f"{n_alive:>3}/{N_TRIALS} survived ({p:.2f}, "
                  f"Wilson [{lo:.2f}, {hi:.2f}])  mean_swaps={n_swaps_total/N_TRIALS:.1f}  "
                  f"{wall:.1f}s")
            tmp = OUTPUT_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({
                "scenario": "adaptive_epsilon_grooming",
                "n_trials_per_cell": N_TRIALS,
                "epsilon_window": EPSILON_WINDOW,
                "epsilon_factor": EPSILON_FACTOR,
                "grooming_strategies": GROOMING_STRATEGIES,
                "epsilon_modes": EPSILON_MODES,
                "records": records,
            }, indent=2))
            tmp.replace(OUTPUT_PATH)

    grand_wall = time.perf_counter() - grand_t0
    print(f"\nTotal: {len(records)} (strategy, mode) pairs, "
          f"{len(records) * N_TRIALS} trials, {grand_wall:.1f}s wall clock.")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
