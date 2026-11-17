"""Plot seeded model output; no empirical macaque observations are used."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from macaque_sim.config.sim_config import default_config
from macaque_sim.main import run_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/strategy-results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    strategies = ("uniform", "upward_biased", "coalition")
    runs, rows, summary = [], [], {}
    for strategy in strategies:
        cfg = default_config()
        cfg.grooming_strategy = strategy
        cfg.T = 10.0
        cfg.validate()
        log = run_simulation(cfg)
        time, power = log.get_time_axis(), log.get_power_history()
        ranks = log.get_rank_history()
        assert len(time) == len(power) == log.step_idx
        assert np.isfinite(power).all() and np.allclose(power.sum(axis=1), 1)
        runs.append((time, power))
        for step, t in enumerate(time):
            for agent in range(cfg.N):
                rows.append((strategy, float(t), agent, int(ranks[step, agent]), float(power[step, agent])))
        summary[strategy] = dict(seed=cfg.random_seed, agents=cfg.N, logged_steps=int(log.step_idx),
                                 final_logged_time=float(time[-1]), rank_change_events=len(log.get_events()))
    with (args.output_dir / "power-trajectories.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("strategy", "model_time", "agent_index", "rank", "normalized_power"))
        writer.writerows(rows)
    (args.output_dir / "run-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.98, top=.73, bottom=.23, wspace=.16)
    fig.text(.07, .94, "Three strategies. One seeded model comparison.", fontsize=21, weight="bold", color="#0f172a")
    fig.text(.07, .865, "30 simulated agents per panel · each line follows one agent · common axes", color="#475569")
    colors = plt.get_cmap("viridis")(np.linspace(.12, .88, 30))
    for ax, strategy, (time, power) in zip(axes, strategies, runs):
        for agent in range(power.shape[1]):
            ax.plot(time, power[:, agent], color=colors[agent], lw=1.1, alpha=.8)
        ax.set_title(strategy.replace("_", " ").title(), loc="left", weight="bold", pad=12)
        ax.grid(axis="y", color="#e2e8f0", lw=.7)
        ax.set_xlabel("Model time")
        ax.set_xlim(0, 10)
        ax.text(.02, .96, f"{summary[strategy]['rank_change_events']} rank-change events", transform=ax.transAxes,
                va="top", fontsize=10, color="#334155")
    axes[0].set_ylabel("Normalized power")
    fig.text(.07, .075, "Exploratory simulation, not observed behavior. One seed; no strategy ranking or Monte Carlo claim.", fontsize=10, color="#475569")
    fig.savefig(args.output_dir / "strategy-results.png", dpi=160, facecolor="white")
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"PASS: {len(rows)} plotted values exported; finite power sums to one at every logged step.")


if __name__ == "__main__":
    main()
