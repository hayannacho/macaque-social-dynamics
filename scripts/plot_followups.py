"""plot_followups.py -- matplotlib figures for the post-review follow-up studies.

Generates:
  - output/figures/followups/convergence.png
  - output/figures/followups/noise_sweep_coupling_comparison.png  (when run after B2)
  - output/figures/followups/cooldown_sweep.png                   (when run after B3)

Each figure is rendered at 300 DPI from the JSON output of the corresponding
study script. The plots are deliberately simple matplotlib line/bar charts
rather than the production plotly pipeline -- the follow-ups are auxiliary
material and don't need the full publication style guide.

Usage:
    python scripts/plot_followups.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "output" / "results" / "followups"
FIGURES_DIR = REPO_ROOT / "output" / "figures" / "followups"


def plot_convergence() -> None:
    src = RESULTS_DIR / "convergence.json"
    if not src.exists():
        print(f"skip: {src} not present")
        return
    payload = json.loads(src.read_text())
    records = payload["records"]

    # Pivot: cells on x-axis (5 cells), three lines (one per dt level).
    cell_order = ["default", "coalition", "sigma_040", "sigma_090", "spike_040"]
    cell_labels = [
        r"default ($\sigma{=}0.30$)",
        r"coalition ($\sigma{=}0.30$)",
        r"$\sigma{=}0.40$",
        r"$\sigma{=}0.90$",
        r"$\sigma{=}0.40$ spike",
    ]
    dt_order = ["production", "halved", "quartered"]
    dt_legend = {
        "production": r"$\Delta t_{sde}{=}0.10,\ \Delta t_{ode}{=}0.50$ (production)",
        "halved":     r"$\Delta t_{sde}{=}0.05,\ \Delta t_{ode}{=}0.25$ (halved)",
        "quartered":  r"$\Delta t_{sde}{=}0.025,\ \Delta t_{ode}{=}0.125$ (quartered)",
    }

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=150)
    x = list(range(len(cell_order)))
    width_offsets = [-0.18, 0.0, 0.18]
    colors = ["#2c7bb6", "#fdae61", "#d7191c"]
    markers = ["o", "s", "^"]

    for offset, dt_label, color, marker in zip(width_offsets, dt_order, colors, markers):
        rows = [r for r in records if r["dt_label"] == dt_label]
        rows_by_cell = {r["cell"]: r for r in rows}
        ys = [rows_by_cell[c]["alpha_survival_rate"] for c in cell_order]
        los = [rows_by_cell[c]["wilson_lo"] for c in cell_order]
        his = [rows_by_cell[c]["wilson_hi"] for c in cell_order]
        xs = [xi + offset for xi in x]
        # Wilson CIs as error bars
        ax.errorbar(
            xs, ys,
            yerr=[[y - lo for y, lo in zip(ys, los)], [hi - y for y, hi in zip(ys, his)]],
            fmt=marker, color=color, ecolor=color, capsize=3, markersize=6,
            label=dt_legend[dt_label], linestyle="None",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cell_labels, fontsize=9)
    ax.set_ylabel("Alpha survival rate")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.5, len(cell_order) - 0.5)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.set_title(
        "Convergence study: alpha-survival per cell across three "
        "$\\Delta t$ refinement levels (50 trials per cell)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "convergence.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_coupling_comparison() -> None:
    src_max = REPO_ROOT / "output" / "results" / "noise_sweep.json"
    src_mean = RESULTS_DIR / "noise_sweep_meancoupling.json"
    if not src_mean.exists():
        print(f"skip: {src_mean} not present (run study_coupling.py first)")
        return
    if not src_max.exists():
        print(f"skip: {src_max} (production noise sweep) not present")
        return

    # Production noise sweep is the existing schema; mean-coupling sweep is the new schema.
    max_payload = json.loads(src_max.read_text())
    mean_payload = json.loads(src_mean.read_text())

    # Production schema (noise_sweep.json) has cells indexed by sigma value.
    def extract_curve_production(payload):
        cells = payload.get("cells", payload.get("results", []))
        sigma_vals, p_hats, los, his = [], [], [], []
        for c in cells:
            params = c.get("param_values", c.get("parameters", {}))
            sigma = params.get("sigma")
            runs = c.get("runs", [])
            if not runs:
                continue
            n = len(runs)
            k = sum(int(r.get("alpha_survived", False)) for r in runs)
            p = k / n
            from math import sqrt
            z = 1.96
            denom = 1.0 + z * z / n
            centre = (p + z * z / (2 * n)) / denom
            half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
            sigma_vals.append(sigma)
            p_hats.append(p)
            los.append(max(0.0, centre - half))
            his.append(min(1.0, centre + half))
        order = sorted(range(len(sigma_vals)), key=lambda i: sigma_vals[i])
        return ([sigma_vals[i] for i in order],
                [p_hats[i] for i in order],
                [los[i] for i in order],
                [his[i] for i in order])

    def extract_curve_records(payload):
        records = payload.get("records", [])
        sigma_vals = [r["sigma"] for r in records]
        p_hats = [r["alpha_survival_rate"] for r in records]
        los = [r["wilson_lo"] for r in records]
        his = [r["wilson_hi"] for r in records]
        order = sorted(range(len(sigma_vals)), key=lambda i: sigma_vals[i])
        return ([sigma_vals[i] for i in order],
                [p_hats[i] for i in order],
                [los[i] for i in order],
                [his[i] for i in order])

    sx, sp, sl, sh = extract_curve_production(max_payload)
    mx, mp, ml, mh = extract_curve_records(mean_payload)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    ax.fill_between(sx, sl, sh, color="#2c7bb6", alpha=0.18)
    ax.plot(sx, sp, "o-", color="#2c7bb6", markersize=5, linewidth=1.4, label="max coupling (production)")
    ax.fill_between(mx, ml, mh, color="#d7191c", alpha=0.18)
    ax.plot(mx, mp, "s--", color="#d7191c", markersize=5, linewidth=1.4, label="mean coupling")
    ax.set_xlabel(r"Social noise $\sigma$")
    ax.set_ylabel("Alpha survival rate")
    ax.set_title(
        "Coupling-mode comparison on the noise sweep "
        "(100 trials per cell; bands = Wilson 95\\% CI)",
        fontsize=10,
    )
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "noise_sweep_coupling_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_cooldown_sweep() -> None:
    src = RESULTS_DIR / "cooldown_sweep.json"
    if not src.exists():
        print(f"skip: {src} not present (run study_cooldown.py first)")
        return
    payload = json.loads(src.read_text())
    records = payload["records"]

    # Two lines, one per grooming strategy.
    by_strategy: dict[str, list[dict]] = {"uniform": [], "coalition": []}
    for r in records:
        by_strategy[r["grooming_strategy"]].append(r)
    for k in by_strategy:
        by_strategy[k].sort(key=lambda r: r["cooldown_duration"])

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    style = {
        "uniform":   ("#2c7bb6", "o-",  "uniform grooming"),
        "coalition": ("#d7191c", "s--", "coalition grooming"),
    }
    for strat, rows in by_strategy.items():
        if not rows:
            continue
        color, fmt, label = style[strat]
        x = [r["cooldown_duration"] for r in rows]
        y = [r["alpha_survival_rate"] for r in rows]
        lo = [r["wilson_lo"] for r in rows]
        hi = [r["wilson_hi"] for r in rows]
        ax.fill_between(x, lo, hi, color=color, alpha=0.18)
        ax.plot(x, y, fmt, color=color, markersize=6, linewidth=1.4, label=label)

    ax.set_xscale("log")
    ax.set_xlabel(r"Cooldown duration (ODE time units, log scale)")
    ax.set_ylabel("Alpha survival rate")
    ax.set_title(
        "Cooldown sensitivity sweep "
        "($\\sigma{=}0.30$, 100 trials per cell; bands = Wilson 95\\% CI)",
        fontsize=10,
    )
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.axvline(5.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out = FIGURES_DIR / "cooldown_sweep.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_adaptive_epsilon() -> None:
    src = RESULTS_DIR / "adaptive_epsilon_grooming.json"
    if not src.exists():
        print(f"skip: {src} not present (run study_adaptive_epsilon.py first)")
        return
    payload = json.loads(src.read_text())
    records = payload["records"]

    strategies = ["uniform", "upward_biased", "coalition"]
    strategy_labels = {
        "uniform": "uniform",
        "upward_biased": "upward-biased",
        "coalition": "coalition",
    }
    by_mode: dict[str, dict[str, dict]] = {"fixed": {}, "adaptive": {}}
    for r in records:
        by_mode[r["epsilon_mode"]][r["grooming_strategy"]] = r

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    x = list(range(len(strategies)))
    width = 0.35
    colors = {"fixed": "#2c7bb6", "adaptive": "#d7191c"}

    for i, mode in enumerate(["fixed", "adaptive"]):
        offset = (-1 if i == 0 else 1) * width / 2
        xs = [xi + offset for xi in x]
        ys = [by_mode[mode][s]["alpha_survival_rate"] for s in strategies]
        los = [by_mode[mode][s]["wilson_lo"] for s in strategies]
        his = [by_mode[mode][s]["wilson_hi"] for s in strategies]
        yerr = [[y - lo for y, lo in zip(ys, los)],
                [hi - y for y, hi in zip(ys, his)]]
        label = (r"fixed $\varepsilon = 0.02$" if mode == "fixed"
                 else r"adaptive $\varepsilon = 2\,\mathrm{std}(\Delta P)$")
        ax.bar(xs, ys, width=width, yerr=yerr, color=colors[mode],
               capsize=4, label=label, edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([strategy_labels[s] for s in strategies])
    ax.set_ylabel("Alpha survival rate")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        r"Adaptive vs.\ fixed $\varepsilon$ on the grooming-bias contrast "
        r"($\sigma{=}0.30$, 100 trials per cell)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIGURES_DIR / "adaptive_epsilon_grooming.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_convergence()
    plot_coupling_comparison()
    plot_cooldown_sweep()
    plot_adaptive_epsilon()


if __name__ == "__main__":
    main()
