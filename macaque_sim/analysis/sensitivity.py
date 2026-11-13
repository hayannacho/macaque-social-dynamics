# sensitivity.py -- Sprint 4 S4-T4 sensitivity analysis over Jenny's MC sweeps.
# Assignee: Dev A (solver-perf half of S4-T4; Dev C owns the biology half)
# See: project_management/tasks/sprint_4/S4-T4_sensitivity_tuning.md

"""
Sensitivity analysis layer on top of Jenny's S4-T1 MC result JSONs.

NOTE (Sprint 4, S4-T4): Jenny's sweep pipeline landed in `run_sprint4_mc.py`
and writes one `ExperimentResults` JSON per scenario into `output/results/`.
This module turns those raw files into the one answer the ticket actually
asks for: which parameter moves alpha-survival the most, and does the
resulting tuned default still sit inside the solver's stable regime?

I'm doing the gradient with plain central differences directly over
Jenny's grid rather than fitting a polynomial or GP surrogate. Why:
sweep resolutions are 11 / 20 / 2 / 1 grid points (sigma / lambda /
grooming_bias / alpha_removal), which is plenty for a finite-difference
derivative but nowhere near enough to justify a surrogate model.

Jenny went with 1D sweeps plus alpha-removal replicates instead of the
doc's illustrative 10x10 example. That's actually a gift here -- a 1D
partial derivative along each sweep axis is exactly what the ticket
means by "Delta(overthrow_rate) / Delta(parameter)". No 2D ridge
direction headaches.

This module covers the three shapes Jenny's results actually come in:
  - Numeric 1D sweep (noise_sweep.sigma, decay_sweep.lambda_decay)
      -> `compute_numeric_gradient`: central-diff d(metric)/d(param),
         plus a range-normalized version so cross-axis ranking is fair.
  - Categorical 1D sweep (grooming_bias.grooming_strategy)
      -> `compute_categorical_contrast`: max - min across categories;
         a derivative doesn't apply because you can't "half-switch" strategies.
  - No-sweep replicate block (alpha_removal)
      -> `summarize_no_sweep`: mean +/- stderr of the metric across MC
         replicates. Doesn't enter the ranking; Dev C's biology half of
         the report cites it as a baseline post-removal survival number.

The Dev-A-specific piece (the solver-perf half of the ticket) lives in
`validate_solver_stability` at the bottom: whenever we propose a tuned
SimConfig, this verifies the solver still has comfortable stability
margin at those parameters and reports the three concrete margins so
the report can quote them directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from macaque_sim.config.scenarios import build_grooming_matrix
from macaque_sim.config.sim_config import SimConfig
from macaque_sim.experiments.monte_carlo import ExperimentCell, ExperimentResults
from macaque_sim.experiments.results_io import load_experiment_results


# --- 1. Loading the four scenario JSONs ---

# Map Jenny's scenario keys to the filenames her driver writes. Kept in sync
# with `_JOBS` in run_sprint4_mc.py; if she ever renames a file over there,
# change it here too (cross-dev coupling alert).
_SCENARIO_FILES = {
    "noise_sweep": "noise_sweep.json",
    "grooming_bias": "grooming_bias.json",
    "decay_sweep": "decay_sweep.json",
    "alpha_removal": "alpha_removal.json",
}


def load_all_results(results_dir: str | Path) -> dict[str, ExperimentResults]:
    """
    Load whichever of Jenny's four S4-T1 JSONs are on disk into a dict keyed
    by scenario name.

    NOTE (Sprint 4, S4-T4): I skip missing files rather than hard-raising.
    Reason: during analysis iteration you often only care about one scenario
    at a time, and re-running the full 8h sweep just to get one file back
    is painful. If the caller cares, they can check which keys came back.
    """
    results_dir = Path(results_dir)
    out: dict[str, ExperimentResults] = {}
    for key, fname in _SCENARIO_FILES.items():
        path = results_dir / fname
        if path.is_file():
            out[key] = load_experiment_results(path)
    return out


# --- 2. Metric extraction per ExperimentCell ---

# Metrics we currently understand how to summarize. Extend as needed -- just
# teach `_cell_metric` how to route the new name.
VALID_METRICS = ("alpha_survival_rate", "mean_num_swaps")


def _cell_metric(cell: ExperimentCell, metric: str) -> float:
    """Pull a single scalar from an ExperimentCell by name."""
    # ExperimentCell already computes these as @property, so we're just
    # routing by string here. Keep the valid set narrow so a typo fails loud.
    if metric == "alpha_survival_rate":
        return float(cell.alpha_survival_rate)
    if metric == "mean_num_swaps":
        return float(cell.mean_num_swaps)
    raise ValueError(f"Unknown metric {metric!r}; valid: {VALID_METRICS}")


# --- 3. Finite-difference gradient on numeric 1D sweeps ---

@dataclass
class GradientSummary:
    """Per-scenario gradient result; the `gradient_normalized` field is what cross-axis ranking consumes."""

    scenario_name: str
    sweep_key: str
    metric: str
    gradient_abs_mean: float     # mean of |d(metric)/d(param)| over the sweep axis, in raw param units
    gradient_normalized: float   # same thing rescaled by param range -- unit-free, safe to rank against other axes
    metric_range: float          # max - min of metric over the sweep; sanity-check on signal strength
    per_cell_metric: list[float]
    per_cell_param: list[float]


def compute_numeric_gradient(
    results: ExperimentResults,
    metric: str = "alpha_survival_rate",
) -> GradientSummary:
    """
    Central-difference gradient of a metric along a 1D numeric sweep axis.

    Biological reading: if this returns a large gradient_normalized for
    sigma, it means small changes in social-noise volatility produce large
    changes in whether the alpha survives. That's the kind of finding the
    ticket wants us to surface.

    gotcha: np.gradient would be the one-liner here, but its default edge
    handling uses forward/backward diffs at the endpoints, and when a
    scenario has only 2 numeric points those edge values dominate the mean.
    I enforce >= 3 points and pass the actual param values as spacing so
    non-uniform grids still behave.

    gotcha: cross-axis ranking ("is sigma or lambda more influential?") only
    makes sense after normalizing by the parameter's own range. A raw
    gradient of 0.5 per unit-sigma over [0.01, 1.0] is NOT directly
    comparable to 0.5 per unit-lambda over [0.05, 1.0]. The
    `gradient_normalized` field does that rescaling by chain rule and is
    what `rank_parameters_by_influence` consumes.
    """
    if len(results.sweep_keys) != 1:
        raise ValueError(
            f"{results.scenario_name}: numeric gradient expects a 1D sweep; "
            f"got sweep_keys={results.sweep_keys}"
        )
    key = results.sweep_keys[0]
    param_values = [float(v) for v in results.sweep_values[key]]

    if len(param_values) < 3:
        # A 2-point sweep is really a contrast, not a gradient -- route it
        # through the categorical path. Fail loud so the caller notices.
        raise ValueError(
            f"{results.scenario_name}: numeric gradient needs >= 3 grid points, "
            f"got {len(param_values)}. Use compute_categorical_contrast instead."
        )

    # Walk cells in sweep order. For 1D sweeps the coords tuple is (i,), so
    # sorting by the first (and only) element gives ascending param order.
    ordered = sorted(results.cells.items(), key=lambda kv: kv[0][0])
    per_cell_metric = [_cell_metric(cell, metric) for _, cell in ordered]

    # gotcha: if a cell crashed mid-sweep its .runs list is empty and the
    # metric comes back NaN. Impute with the sweep mean before taking the
    # gradient so np.gradient doesn't return an all-NaN vector. The original
    # NaNs are still visible to the caller via per_cell_metric.
    arr = np.array(per_cell_metric, dtype=float)
    if not np.all(np.isfinite(arr)):
        filled = arr.copy()
        finite = arr[np.isfinite(arr)]
        fill_value = float(np.mean(finite)) if finite.size else 0.0
        filled[~np.isfinite(filled)] = fill_value
        arr = filled

    # np.gradient with explicit x-coords handles the non-uniform case too.
    grad = np.gradient(arr, np.asarray(param_values, dtype=float))
    grad_abs_mean = float(np.mean(np.abs(grad)))
    param_range = float(max(param_values) - min(param_values))
    metric_range = float(np.max(arr) - np.min(arr))

    # Chain rule: d(metric)/d(param_normalized) where
    # param_normalized = (param - min) / (max - min), so scaling factor is
    # just param_range.
    gradient_normalized = grad_abs_mean * param_range

    return GradientSummary(
        scenario_name=results.scenario_name,
        sweep_key=key,
        metric=metric,
        gradient_abs_mean=grad_abs_mean,
        gradient_normalized=gradient_normalized,
        metric_range=metric_range,
        per_cell_metric=per_cell_metric,
        per_cell_param=param_values,
    )


# --- 4. Categorical contrast for grooming_bias ---

@dataclass
class CategoricalContrast:
    """Result of a categorical 1D sweep; exposes a gradient_normalized so it sorts alongside numeric gradients."""

    scenario_name: str
    sweep_key: str
    metric: str
    categories: list[str]
    per_category_metric: list[float]
    contrast: float  # max(metric) - min(metric) across categories

    @property
    def gradient_normalized(self) -> float:
        # For the cross-scenario ranking: a 2-category contrast of C costs
        # the same as moving normalized-distance 1 along a numeric axis, so
        # the categorical influence is just |contrast|. That's what lets us
        # put "coalition vs upward_biased" on the same ranking as "sigma".
        return abs(self.contrast)


def compute_categorical_contrast(
    results: ExperimentResults,
    metric: str = "alpha_survival_rate",
) -> CategoricalContrast:
    """
    For a categorical 1D sweep (grooming_strategy: upward_biased vs coalition),
    report max(metric) - min(metric). Not a derivative in the calculus sense --
    you can't smoothly half-coalition your way between strategies -- but the
    contrast plays the same role for ranking purposes.

    Why : coalition vs upward_biased is a discrete biological choice about how
    subordinates invest their grooming. The question the ticket cares about is
    "does switching strategy matter more or less than tweaking sigma?", and
    for that we just need a comparable magnitude.
    """
    if len(results.sweep_keys) != 1:
        raise ValueError(
            f"{results.scenario_name}: categorical contrast expects a 1D sweep"
        )
    key = results.sweep_keys[0]
    categories = [str(v) for v in results.sweep_values[key]]

    ordered = sorted(results.cells.items(), key=lambda kv: kv[0][0])
    per_category_metric = [_cell_metric(cell, metric) for _, cell in ordered]

    finite = [m for m in per_category_metric if np.isfinite(m)]
    contrast = float(max(finite) - min(finite)) if finite else float("nan")

    return CategoricalContrast(
        scenario_name=results.scenario_name,
        sweep_key=key,
        metric=metric,
        categories=categories,
        per_category_metric=per_category_metric,
        contrast=contrast,
    )


# --- 5. Replicate-only summary (alpha_removal) ---

@dataclass
class NoSweepSummary:
    """Mean +/- stderr of a metric across MC replicates in a no-sweep scenario."""

    scenario_name: str
    metric: str
    mean: float
    stderr: float
    n_runs: int


def summarize_no_sweep(
    results: ExperimentResults,
    metric: str = "alpha_survival_rate",
) -> NoSweepSummary:
    """
    alpha_removal has no parameter sweep -- it's a single grid cell replicated
    N times to characterize the distribution of post-removal outcomes. Return
    mean +/- stderr so Dev C's biology half of the report can quote a concrete
    baseline survival number for the ticket's "power vacuum" question.

    gotcha: results.cells should have exactly one entry (coords (0,)) for a
    no-sweep scenario. If Jenny ever adds a parameter sweep to alpha_removal
    later, this silently returns the first cell's stats -- worth re-reading
    the scenario file before trusting the number.
    """
    if not results.cells:
        raise ValueError(f"{results.scenario_name}: no cells in results")
    cell = next(iter(results.cells.values()))

    if metric == "alpha_survival_rate":
        vals = np.array([1.0 if r.alpha_survived else 0.0 for r in cell.runs], dtype=float)
    elif metric == "mean_num_swaps":
        vals = np.array([r.num_swaps for r in cell.runs], dtype=float)
    else:
        raise ValueError(f"Unknown metric {metric!r}; valid: {VALID_METRICS}")

    n = int(vals.size)
    mean = float(np.mean(vals)) if n else float("nan")
    # Bessel-corrected std; stderr needs >= 2 runs to be meaningful.
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    stderr = float(std / np.sqrt(n)) if n > 0 else float("nan")

    return NoSweepSummary(
        scenario_name=results.scenario_name,
        metric=metric,
        mean=mean,
        stderr=stderr,
        n_runs=n,
    )


# --- 5b. Wilson CIs for binomial metrics (alpha_survival_rate) ---
#
# Why Wilson and not normal-approx: with n=50 trials per cell and survival
# rates that pin at 1.0 in the low-noise regime and crash near 0 in the
# high-noise regime, the normal approximation gives intervals that cross
# zero or one (which is meaningless). Wilson stays inside [0,1] and degrades
# gracefully at the boundaries -- standard recommendation for binomial CIs
# under small-to-moderate n.

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson 95% CI for a binomial proportion p = k/n.

    Returns (low, high) clipped to [0, 1]. z=1.96 is the default 95% level.
    Returns (nan, nan) if n == 0.
    """
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return float(low), float(high)


def cell_alpha_survival_ci(cell: ExperimentCell, z: float = 1.96) -> tuple[float, float]:
    """Wilson CI on alpha-survival count for a single ExperimentCell."""
    n = len(cell.runs)
    k = sum(1 for r in cell.runs if r.alpha_survived)
    return wilson_ci(k, n, z=z)


# --- 6. Cross-scenario ranking ---

def rank_parameters_by_influence(
    all_results: dict[str, ExperimentResults],
    metric: str = "alpha_survival_rate",
) -> list[tuple[str, str, float]]:
    """
    Return `[(scenario_name, sweep_key_or_label, gradient_normalized), ...]`
    sorted descending by influence magnitude. Skips any scenario that has no
    sweep (those contribute replicate stats, not a gradient).

    This is the direct answer to the S4-T4 acceptance criterion "identify
    top-2 most influential parameters" -- take the first two entries of the
    returned list.

    gotcha: a scenario with a NaN metric everywhere returns a 0 gradient
    rather than being skipped, so it sinks to the bottom of the ranking
    instead of corrupting sort order.
    """
    rows: list[tuple[str, str, float]] = []
    for scen_key, res in all_results.items():
        if not res.sweep_keys:
            # No sweep: can't be ranked as a gradient, just replicate stats.
            continue
        key = res.sweep_keys[0]
        vals = res.sweep_values[key]
        # Detect numeric vs categorical from the first value's type.
        is_numeric = all(isinstance(v, (int, float)) for v in vals)
        try:
            if is_numeric and len(vals) >= 3:
                g = compute_numeric_gradient(res, metric=metric)
                rows.append((scen_key, key, g.gradient_normalized))
            else:
                c = compute_categorical_contrast(res, metric=metric)
                rows.append((scen_key, key, c.gradient_normalized))
        except ValueError:
            # Malformed cell -- push it to the bottom of the ranking with a
            # zero magnitude so a bad sweep doesn't break the whole report.
            rows.append((scen_key, key, 0.0))
    rows.sort(key=lambda row: row[2], reverse=True)
    return rows


# --- 7. Solver-stability verification for a proposed tuned SimConfig ---

@dataclass
class StabilityCheck:
    """
    Result bundle for validate_solver_stability.

    Fields expose the numeric margins themselves (not just pass/fail) so the
    sensitivity report can quote them directly in the tuned-defaults section.
    """

    ok: bool
    messages: list[str]
    dt_sde_margin_ratio: float       # (2/lambda) / dt_sde; want >= 1.10 for safety headroom
    feller_min_ratio: float          # min over off-diag of 2*B_ij / sigma^2; > 1 preferred
    multirate_identity_ratio: float  # (M * dt_sde) / dt_ode; want ~1.0 so envelope and RK4 share an interval


def validate_solver_stability(cfg: SimConfig) -> StabilityCheck:
    """
    Dev-A-owned safety check on any proposed tuned SimConfig. S4-T4 says
    "confirm the tuned defaults don't push the solver into unstable regimes";
    this is that gate.

    Three checks, in descending severity:

      1. HARD -- dt_sde < 2/lambda with at least 10% margin. em_step already
         raises at dt_sde >= 2/lambda (see euler_maruyama.py Section 1), but
         I want explicit headroom so a sweep that jitters dt_sde a little
         doesn't land right on the stability boundary.

      2. SOFT -- Feller ratio 2*B_ij / sigma^2 for the typical grooming matrix
         under cfg.grooming_strategy. Violation is a yellow flag, not red:
         em_step emits a RuntimeWarning rather than raising, and the
         max(0, A) floor catches whatever slips past. But a tuned config
         shouldn't MAKE this worse if we can avoid it.

      3. SOFT -- M * dt_sde ~ dt_ode (multi-rate identity). The envelope in
         multirate_step is the element-wise max across M micro-steps of
         length dt_sde, which is then fed into ONE rk4_step of length dt_ode.
         If M*dt_sde != dt_ode the two clocks drift apart and the envelope
         stops being physically meaningful. Default is 5 * 0.10 = 0.50 = dt_ode.
    """
    messages: list[str] = []
    ok = True

    # --- Check 1: dt_sde stability margin ---
    dt_cap = 2.0 / cfg.lambda_decay
    dt_margin = dt_cap / cfg.dt_sde if cfg.dt_sde > 0 else float("inf")
    if dt_margin < 1.10:
        ok = False
        messages.append(
            f"dt_sde={cfg.dt_sde} is only {dt_margin:.3f}x below 2/lambda "
            f"(={dt_cap:.3f}); want >= 1.10x for safety headroom so a small "
            f"param jitter in a sweep doesn't land on the boundary."
        )

    # --- Check 2: Feller ratio on the typical B ---
    # Build B the way main.py does at t=0 (default rank order). Only the
    # off-diagonal entries are real CIR processes -- diagonal is zero by
    # construction (monkeys don't groom themselves) so including it would
    # always fire the warning.
    feller_min = float("nan")
    try:
        B = build_grooming_matrix(cfg)
        off_diag = B[~np.eye(cfg.N, dtype=bool)]
        if cfg.sigma > 0:
            feller_min = float(np.min(2.0 * off_diag) / (cfg.sigma ** 2))
        else:
            feller_min = float("inf")
    except Exception as e:
        # Build can fail if, e.g., grooming_strategy is "coalition" and the
        # clique is ill-formed at this N. Report it rather than crashing the
        # whole check.
        messages.append(f"Could not build grooming matrix to check Feller: {e!r}")

    if np.isfinite(feller_min) and feller_min <= 1.0:
        # Soft warning: leaves `ok` unchanged because em_step treats this as
        # a RuntimeWarning, not an error. We just surface the number.
        messages.append(
            f"Feller ratio 2*B_min/sigma^2 = {feller_min:.3f} <= 1 at these "
            f"params; non-negativity of A is only enforced by the max(0, ...) "
            f"floor (em_step emits a RuntimeWarning, not an error)."
        )

    # --- Check 3: Multi-rate identity ---
    multi_ratio = (cfg.micro_steps_per_macro * cfg.dt_sde) / cfg.dt_ode
    if not (0.95 <= multi_ratio <= 1.05):
        ok = False
        messages.append(
            f"Multi-rate identity violated: M*dt_sde/dt_ode = {multi_ratio:.3f}; "
            f"want ~1.0 so the envelope-max across M micro-steps corresponds "
            f"to the same wall-clock interval RK4 is advancing over."
        )

    if ok and not messages:
        messages.append("All solver-stability checks pass.")

    return StabilityCheck(
        ok=ok,
        messages=messages,
        dt_sde_margin_ratio=dt_margin,
        feller_min_ratio=feller_min,
        multirate_identity_ratio=multi_ratio,
    )


# --- 8. Markdown rendering for the sensitivity_report.md Numerical Findings ---

def render_ranking_markdown(
    all_results: dict[str, ExperimentResults],
    metric: str = "alpha_survival_rate",
) -> str:
    """
    Render the ranked-influence table as a markdown fragment. Drop this into
    `project_management/reports/sensitivity_report.md` under the Numerical
    Findings section after Jenny's MC sweeps have finished running.
    """
    ranking = rank_parameters_by_influence(all_results, metric=metric)
    if not ranking:
        return "_No sweeps available to rank -- run `python -m macaque_sim.experiments.run_sprint4_mc` first._\n"

    lines = [
        f"| Rank | Scenario | Parameter | Normalized influence on `{metric}` |",
        "|------|----------|-----------|-------------------------------------|",
    ]
    for i, (scen, key, influence) in enumerate(ranking, start=1):
        lines.append(f"| {i} | `{scen}` | `{key}` | {influence:.4f} |")
    return "\n".join(lines) + "\n"


def render_stability_markdown(check: StabilityCheck) -> str:
    """Render a StabilityCheck as a small markdown block for the report."""
    lines = [
        "| Check | Value | Interpretation |",
        "|-------|-------|----------------|",
        f"| `dt_sde` margin vs `2/lambda` | {check.dt_sde_margin_ratio:.3f}x | want >= 1.10x |",
        f"| Feller ratio `2*B_min / sigma^2` | {check.feller_min_ratio:.3f} | > 1 preferred (soft) |",
        f"| Multi-rate identity `M*dt_sde / dt_ode` | {check.multirate_identity_ratio:.3f} | want ~1.0 |",
        "",
        f"Overall: {'PASS' if check.ok else 'FAIL'}",
        "",
    ]
    for msg in check.messages:
        lines.append(f"- {msg}")
    return "\n".join(lines) + "\n"


# --- 9. CLI entry point ---

def _main(argv: list[str] | None = None) -> int:
    """
    Print the sensitivity ranking + solver-stability block for the current
    default_config to stdout. Paste the output into sensitivity_report.md
    once Jenny's MC sweeps have produced JSONs.

        python -m macaque_sim.analysis.sensitivity
        python -m macaque_sim.analysis.sensitivity --results-dir output/results --metric mean_num_swaps
    """
    import argparse

    from macaque_sim.config.sim_config import default_config

    parser = argparse.ArgumentParser(
        description="Sensitivity ranking over Jenny's S4-T1 MC JSONs + solver-stability check on default_config."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("output/results"))
    parser.add_argument("--metric", default="alpha_survival_rate", choices=VALID_METRICS)
    args = parser.parse_args(argv)

    print("## Numerical Findings (auto-generated)\n")
    all_results = load_all_results(args.results_dir)
    if not all_results:
        print(
            f"_No result JSONs found under {args.results_dir}. "
            f"Run `python -m macaque_sim.experiments.run_sprint4_mc` first._\n"
        )
    else:
        print(render_ranking_markdown(all_results, metric=args.metric))
        # Also list no-sweep summaries so alpha_removal's baseline survival
        # rate appears in the report even though it doesn't rank.
        for scen_key, res in all_results.items():
            if not res.sweep_keys:
                s = summarize_no_sweep(res, metric=args.metric)
                print(
                    f"- `{scen_key}` (no sweep): {args.metric} = "
                    f"{s.mean:.3f} +/- {s.stderr:.3f} over n={s.n_runs} replicates"
                )
        print()

    print("## Solver Stability on `default_config()` (auto-generated)\n")
    print(render_stability_markdown(validate_solver_stability(default_config())))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
