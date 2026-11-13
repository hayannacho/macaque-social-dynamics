"""study_cooldown.py -- cooldown sensitivity sweep (post-review).

Mirrors the epsilon sweep (Fig. study6_epsilon_sweep) on cooldown_duration to
show whether the headline numbers depend on this calibration. The reviewer
flagged the production cooldown of 5 ODE time units (~35 days) as biologically
incoherent at the matrilineal-overthrow scale: too long for dyadic post-conflict
reconciliation (Aureli 1989: minutes-to-hours) and too short for inter-overthrow
refractory periods (Oates-O'Brien 2010: >= 6 months; Ehardt-Bernstein 1986:
~1.75 years mean). The sweep is wide on purpose -- two orders of magnitude on
either side of 5 -- so the result is informative whether the gap collapses
across the range or only at the extremes.

Output: output/results/followups/cooldown_sweep.json
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from macaque_sim.main import run_single_trial


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "output" / "results" / "followups" / "cooldown_sweep.json"

# Two orders of magnitude on either side of the production value (5).
COOLDOWN_VALUES = [0.5, 2.0, 5.0, 20.0, 100.0]
GROOMING_STRATEGIES = ["uniform", "coalition"]
N_TRIALS = 100


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
        for cd_idx, cd in enumerate(COOLDOWN_VALUES):
            n_alive = 0
            t0 = time.perf_counter()
            for run_idx in range(N_TRIALS):
                seed = 70000 + strat_idx * 10000 + cd_idx * 1000 + run_idx
                cfg = {
                    "sigma": 0.30,
                    "grooming_strategy": strat,
                    "random_seed": seed,
                    "cooldown_duration": float(cd),
                }
                try:
                    logger = run_single_trial(cfg)
                    n_alive += int(getattr(logger, "alpha_survived", False))
                except Exception as exc:  # noqa: BLE001
                    print(f"  trial {run_idx} ({strat}, cd={cd}) raised: {exc}")
            wall = time.perf_counter() - t0
            p = n_alive / N_TRIALS
            lo, hi = wilson_ci(n_alive, N_TRIALS)
            rec = {
                "grooming_strategy": strat,
                "cooldown_duration": float(cd),
                "n_trials": N_TRIALS,
                "n_alpha_survived": n_alive,
                "alpha_survival_rate": p,
                "wilson_lo": lo,
                "wilson_hi": hi,
                "wall_clock_s": wall,
            }
            records.append(rec)
            print(f"strat={strat:<10} cooldown={cd:>6.1f}  "
                  f"{n_alive}/{N_TRIALS} survived ({p:.2f}, "
                  f"Wilson [{lo:.2f}, {hi:.2f}]), {wall:.1f}s")
            tmp = OUTPUT_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({
                "scenario": "cooldown_sweep",
                "n_trials_per_cell": N_TRIALS,
                "cooldown_values": COOLDOWN_VALUES,
                "grooming_strategies": GROOMING_STRATEGIES,
                "records": records,
            }, indent=2))
            tmp.replace(OUTPUT_PATH)

    grand_wall = time.perf_counter() - grand_t0
    print(f"\nTotal: {len(records)} (strategy, cooldown) pairs, "
          f"{len(records) * N_TRIALS} trials, {grand_wall:.1f}s wall clock.")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
