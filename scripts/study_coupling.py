"""study_coupling.py -- noise sweep with coupling_mode='mean' (post-review).

Re-runs the production noise sweep at the same 11 sigma values, 100 trials per
cell, but with coupling_mode='mean' so the multi-rate aggregation is the
unbiased macro-interval-mean rather than the production element-wise maximum.

The reviewer's concern (and the disclosure in §3.4) is that the max envelope
biases the centrality input upward in a sigma-dependent way -- so part of the
production noise sweep's sigma transition is deterministic envelope inflation
rather than stochastic destabilization. This experiment quantifies the empirical
difference between the two coupling modes and feeds the §5.4 paragraph + §3.4
caveat update.

Output: output/results/followups/noise_sweep_meancoupling.json
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from macaque_sim.main import run_single_trial


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "output" / "results" / "followups" / "noise_sweep_meancoupling.json"

# Same 11 sigma values as the production noise sweep.
SIGMA_VALUES = list(np.linspace(0.01, 1.0, 11))
N_TRIALS = 100
COUPLING_MODE = "mean"


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

    for sigma_idx, sigma in enumerate(SIGMA_VALUES):
        n_alive = 0
        t0 = time.perf_counter()
        for run_idx in range(N_TRIALS):
            # Seed offset 80000 keeps these distinct from production noise_sweep
            # seeds (0..10*100) and from the convergence-study seeds (90000+).
            seed = 80000 + sigma_idx * 1000 + run_idx
            cfg = {
                "sigma": float(sigma),
                "grooming_strategy": "uniform",
                "random_seed": seed,
                "coupling_mode": COUPLING_MODE,
            }
            try:
                logger = run_single_trial(cfg)
                n_alive += int(getattr(logger, "alpha_survived", False))
            except Exception as exc:  # noqa: BLE001
                print(f"  trial {run_idx} (sigma={sigma:.3f}) raised: {exc}")
        wall = time.perf_counter() - t0
        p = n_alive / N_TRIALS
        lo, hi = wilson_ci(n_alive, N_TRIALS)
        rec = {
            "coupling_mode": COUPLING_MODE,
            "sigma": float(sigma),
            "n_trials": N_TRIALS,
            "n_alpha_survived": n_alive,
            "alpha_survival_rate": p,
            "wilson_lo": lo,
            "wilson_hi": hi,
            "wall_clock_s": wall,
        }
        records.append(rec)
        print(f"sigma={sigma:.3f}  {n_alive}/{N_TRIALS} survived "
              f"({p:.2f}, Wilson [{lo:.2f}, {hi:.2f}]), {wall:.1f}s")
        # Atomic incremental write.
        tmp = OUTPUT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "scenario": "noise_sweep_meancoupling",
            "coupling_mode": COUPLING_MODE,
            "n_trials_per_cell": N_TRIALS,
            "records": records,
        }, indent=2))
        tmp.replace(OUTPUT_PATH)

    grand_wall = time.perf_counter() - grand_t0
    print(f"\nTotal: {len(records)} cells, {len(records) * N_TRIALS} trials, "
          f"{grand_wall:.1f}s wall clock.")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
