# test_sensitivity.py -- Unit tests for the S4-T4 sensitivity analysis layer.
# pytest macaque_sim/tests/test_sensitivity.py -v

import dataclasses

import numpy as np
import pytest

from macaque_sim.analysis.sensitivity import (
    CategoricalContrast,
    GradientSummary,
    NoSweepSummary,
    StabilityCheck,
    cell_alpha_survival_ci,
    compute_categorical_contrast,
    compute_numeric_gradient,
    rank_parameters_by_influence,
    summarize_no_sweep,
    validate_solver_stability,
    wilson_ci,
)
from macaque_sim.config.sim_config import default_config
from macaque_sim.experiments.monte_carlo import (
    ExperimentCell,
    ExperimentResults,
    TrialResult,
)


# --- Synthetic ExperimentResults builders (no real MC required) ---

def _cell(coords: tuple[int, ...], params: dict, survive_rate: float, n_runs: int = 20, num_swaps: int = 0) -> ExperimentCell:
    """
    Build an ExperimentCell whose alpha_survived flags are chosen to produce
    exactly `survive_rate` when averaged, so the downstream metric is
    deterministic (no RNG involved in the fixtures -- repeatability matters
    when debugging a gradient calc).
    """
    # Count how many survivors we need to hit this rate exactly.
    n_alive = int(round(survive_rate * n_runs))
    runs = [
        TrialResult(
            alpha_survived=(i < n_alive),
            num_swaps=int(num_swaps),
            final_ranks=list(range(1, 4)),
            raw_logger=None,
        )
        for i in range(n_runs)
    ]
    return ExperimentCell(coords=coords, param_values=dict(params), runs=runs)


def _numeric_sweep_results(name: str, key: str, values: list[float], survive_rates: list[float]) -> ExperimentResults:
    """Build a 1D numeric-sweep ExperimentResults fixture from hand-picked survival rates."""
    assert len(values) == len(survive_rates), "one survival rate per sweep value"
    cells = {
        (i,): _cell((i,), {key: v}, sr)
        for i, (v, sr) in enumerate(zip(values, survive_rates))
    }
    return ExperimentResults(
        scenario_name=name,
        sweep_keys=[key],
        sweep_values={key: list(values)},
        shape=(len(values),),
        cells=cells,
        elapsed_seconds=0.0,
    )


def _categorical_sweep_results(name: str, key: str, categories: list[str], survive_rates: list[float]) -> ExperimentResults:
    """Build a 1D categorical-sweep ExperimentResults fixture."""
    cells = {
        (i,): _cell((i,), {key: cat}, sr)
        for i, (cat, sr) in enumerate(zip(categories, survive_rates))
    }
    return ExperimentResults(
        scenario_name=name,
        sweep_keys=[key],
        sweep_values={key: list(categories)},
        shape=(len(categories),),
        cells=cells,
        elapsed_seconds=0.0,
    )


def _no_sweep_results(name: str, survive_rate: float, n_runs: int = 40) -> ExperimentResults:
    """Build a no-sweep ExperimentResults fixture (single cell, many replicates)."""
    cell = _cell((0,), {}, survive_rate, n_runs=n_runs)
    return ExperimentResults(
        scenario_name=name,
        sweep_keys=[],
        sweep_values={},
        shape=(1,),
        cells={(0,): cell},
        elapsed_seconds=0.0,
    )


# --- 1. Numeric gradient tests ---

def test_numeric_gradient_monotone_decrease():
    """
    Verify that a monotonically decreasing metric across a numeric sweep
    produces a positive mean absolute gradient of roughly the expected slope.

    Why : this is the core calculation the S4-T4 report rests on. If the
    slope is wrong or sign-flipped, the "which parameter is most
    influential" ranking is garbage and Dev C gets the wrong biological
    story to line up against the literature.

    test: hand-build a 5-point noise sweep where alpha_survival drops from
    1.0 to 0.0 linearly in sigma over [0.0, 1.0]. The expected slope is -1.0
    per unit sigma, so |gradient| should average to 1.0 and gradient_normalized
    (scaled by the 1.0-wide param range) should also be 1.0.
    """
    # Set up a synthetic 5-point sigma sweep with a linear survival curve
    sigmas = [0.0, 0.25, 0.5, 0.75, 1.0]
    survive = [1.0, 0.75, 0.5, 0.25, 0.0]   # alpha-survival drops linearly with sigma
    results = _numeric_sweep_results("noise_sweep", "sigma", sigmas, survive)

    # Run the gradient computation
    g = compute_numeric_gradient(results, metric="alpha_survival_rate")

    # Slope magnitude should be 1.0/unit-sigma (ignoring endpoint rounding)
    assert g.gradient_abs_mean == pytest.approx(1.0, abs=1e-6), (
        f"Expected |slope| ~= 1.0 per unit sigma; got {g.gradient_abs_mean}"
    )
    # Range is 1.0, so normalized == raw here
    assert g.gradient_normalized == pytest.approx(1.0, abs=1e-6)
    # Metric range is 1.0 as well (survival went from 1 to 0)
    assert g.metric_range == pytest.approx(1.0, abs=1e-6)
    # Sanity: the per-cell metrics we put in came back out
    assert g.per_cell_metric == pytest.approx(survive)


def test_numeric_gradient_requires_three_points():
    """
    Verify that numeric gradient refuses to operate on a 2-point sweep.

    Why : central differences are meaningless on 2 points (both endpoints
    are the same forward diff), and a 2-point numeric sweep is really a
    contrast anyway. We want a loud ValueError so the caller routes it
    through compute_categorical_contrast instead of silently getting a
    biased single-difference gradient.

    test: build a 2-point fake numeric sweep and confirm ValueError.
    """
    # Build a 2-point numeric sweep (should be routed through categorical path)
    results = _numeric_sweep_results("tiny_sweep", "sigma", [0.1, 0.2], [0.9, 0.5])

    # Expect compute_numeric_gradient to reject it loudly
    with pytest.raises(ValueError, match=">= 3 grid points"):
        compute_numeric_gradient(results, metric="alpha_survival_rate")


def test_numeric_gradient_normalizes_by_param_range():
    """
    Verify that gradient_normalized rescales so two scenarios with the same
    effect size but different parameter ranges come out equal.

    Why : this is the whole point of normalization. Jenny's sigma sweep is
    over [0.01, 1.0] (range 0.99) and lambda sweep is over [0.05, 1.0]
    (range 0.95). Without normalization, a sigma effect of 0.5 and an
    identical-looking lambda effect of 0.5 would rank differently just
    because of the range difference. After normalization they should match.

    test: two 5-point sweeps where the metric drops from 1.0 to 0.0 linearly,
    but with different param axes. Raw gradient differs; normalized should match.
    """
    # First sweep: sigma over [0.0, 1.0], metric drops 1.0 -> 0.0
    sig = _numeric_sweep_results("a", "sigma", [0.0, 0.25, 0.5, 0.75, 1.0], [1.0, 0.75, 0.5, 0.25, 0.0])
    # Second sweep: lambda over [0.0, 0.5] (half as wide), same metric drop
    lam = _numeric_sweep_results("b", "lambda_decay", [0.0, 0.125, 0.25, 0.375, 0.5], [1.0, 0.75, 0.5, 0.25, 0.0])

    # Raw gradients differ (lambda slope is twice as steep per raw unit)
    g_sig = compute_numeric_gradient(sig)
    g_lam = compute_numeric_gradient(lam)
    assert g_lam.gradient_abs_mean > g_sig.gradient_abs_mean, (
        "sanity check: narrower param range should give larger raw slope"
    )
    # Normalized gradients should agree (same effect per normalized unit of param)
    assert g_sig.gradient_normalized == pytest.approx(g_lam.gradient_normalized, abs=1e-6)


def test_numeric_gradient_imputes_nan_cells():
    """
    Verify that a NaN-metric cell (e.g. a crashed sweep point) is imputed
    with the sweep mean rather than producing an all-NaN gradient.

    Why : Jenny's checkpointing is robust but a sweep cell can still end up
    empty if the simulator crashed mid-write. Rather than propagating NaN
    through the whole analysis and breaking the ranking, the gradient
    function imputes the sweep mean for that cell. The per_cell_metric
    list is untouched so the caller can still see the gap.

    test: build a 5-point sweep with one cell containing zero runs (metric
    -> NaN), and confirm gradient_abs_mean is finite.
    """
    # Build 5-point sweep; drop all runs from the middle cell so its metric is NaN
    sigmas = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = _numeric_sweep_results("noise_sweep", "sigma", sigmas, [1.0, 0.75, 0.5, 0.25, 0.0])
    # Blow away the middle cell's runs to simulate a crashed trial
    results.cells[(2,)] = ExperimentCell(coords=(2,), param_values={"sigma": 0.5}, runs=[])

    # Compute gradient and verify it's finite (imputation ran)
    g = compute_numeric_gradient(results, metric="alpha_survival_rate")
    assert np.isfinite(g.gradient_abs_mean), (
        f"Expected finite gradient after NaN imputation, got {g.gradient_abs_mean}"
    )
    # Per-cell metric list still shows the NaN so caller isn't lied to
    assert not np.isfinite(g.per_cell_metric[2])


# --- 2. Categorical contrast test ---

def test_categorical_contrast_between_two_strategies():
    """
    Verify that compute_categorical_contrast returns max-minus-min across
    categories and exposes a sortable gradient_normalized.

    Why : for grooming_bias (upward_biased vs coalition) we can't compute a
    derivative -- you can't smoothly half-switch strategies -- but we still
    want this scenario to appear in the cross-scenario ranking. The
    |contrast| value plays the same role as gradient_normalized does for
    numeric sweeps.

    test: build a 2-category fixture where survival differs by 0.4 between
    strategies; confirm contrast == 0.4 and gradient_normalized agrees.
    """
    # Build 2-category grooming strategy fixture: coalition survives more than upward_biased
    results = _categorical_sweep_results(
        "grooming_bias", "grooming_strategy",
        ["upward_biased", "coalition"], [0.3, 0.7],
    )

    # Compute the contrast
    c = compute_categorical_contrast(results, metric="alpha_survival_rate")

    # Contrast is max - min across categories
    assert c.contrast == pytest.approx(0.4, abs=1e-6)
    # Ranking-facing property is just |contrast|
    assert c.gradient_normalized == pytest.approx(0.4, abs=1e-6)
    # Preserve category ordering for downstream reporting
    assert c.categories == ["upward_biased", "coalition"]


# --- 3. No-sweep replicate summary test ---

def test_no_sweep_summary_reports_mean_and_stderr():
    """
    Verify that summarize_no_sweep reports the replicate mean and stderr for
    alpha_removal-style single-cell experiments.

    Why : alpha_removal has no parameter sweep (just a single cell replicated
    50 times), so it can't produce a gradient. But Dev C's biology half of
    the report still needs a concrete number for post-removal survival, and
    that number needs a stderr so "x% of the time the alpha gets overthrown"
    has a confidence bound.

    test: build a 40-run single-cell fixture with exactly 25% alpha survival.
    Confirm mean == 0.25, stderr > 0 (40 is enough runs for Bessel-corrected
    std to be meaningful), and n_runs == 40.
    """
    # Build a 40-replicate alpha_removal-style fixture with 25% survival
    results = _no_sweep_results("alpha_removal", survive_rate=0.25, n_runs=40)

    # Summarize replicate statistics
    s = summarize_no_sweep(results, metric="alpha_survival_rate")

    # Mean should be exactly 0.25 (we constructed it that way)
    assert s.mean == pytest.approx(0.25, abs=1e-6)
    # Stderr should be strictly positive for a 25/75 split across 40 runs
    assert s.stderr > 0.0
    # Replicate count should come back correctly
    assert s.n_runs == 40


# --- 4. Cross-scenario ranking test ---

def test_ranking_identifies_dominant_axis():
    """
    Verify that rank_parameters_by_influence puts the more-influential axis
    ahead of the less-influential one.

    Why : the S4-T4 acceptance criterion literally says "sensitivity report
    identifies top-2 most influential parameters". If the sort order is
    wrong we name the wrong parameter as the dominant one and the biology
    discussion points at the wrong lever.

    test: build two numeric sweeps with identical param ranges but different
    metric swings (0.9 vs 0.3). Rank them and confirm the stronger sweep is
    first, the weaker is second, and that the alpha_removal no-sweep block
    does NOT appear in the ranking.
    """
    # Strong sweep: metric drops 0.95 -> 0.05 (influence ~= 0.9)
    strong = _numeric_sweep_results("noise_sweep", "sigma",
                                    [0.0, 0.25, 0.5, 0.75, 1.0],
                                    [0.95, 0.7, 0.5, 0.3, 0.05])
    # Weak sweep: metric drops 0.65 -> 0.35 (influence ~= 0.3)
    weak = _numeric_sweep_results("decay_sweep", "lambda_decay",
                                  [0.0, 0.25, 0.5, 0.75, 1.0],
                                  [0.65, 0.575, 0.5, 0.425, 0.35])
    # No-sweep scenario that should NOT appear in the ranking
    no_sweep = _no_sweep_results("alpha_removal", survive_rate=0.5)

    # Pack them into a dict matching what load_all_results returns
    all_res = {"noise_sweep": strong, "decay_sweep": weak, "alpha_removal": no_sweep}

    # Compute the ranking
    ranking = rank_parameters_by_influence(all_res, metric="alpha_survival_rate")

    # Strong should come first, weak second, alpha_removal absent
    assert len(ranking) == 2, f"Expected 2 ranked rows (sweeps only); got {ranking}"
    assert ranking[0][0] == "noise_sweep", f"Strong axis should rank first; got {ranking[0]}"
    assert ranking[1][0] == "decay_sweep", f"Weak axis should rank second; got {ranking[1]}"
    # The strong axis should have larger influence
    assert ranking[0][2] > ranking[1][2]


# --- 5. Solver-stability validator tests ---

def test_default_config_passes_solver_stability():
    """
    Verify that the current default_config() clears all solver-stability
    checks with comfortable margin.

    Why : this is the regression test that protects the solver from being
    silently pushed into an unstable regime by a future tuning change. If
    someone later edits default_config and this test breaks, they have to
    either justify the new margin or back out the tuning.

    test: call default_config(), pipe it through validate_solver_stability,
    and assert ok is True with the known-good margins at the current
    defaults (dt_margin = 20.0, multi-rate ratio = 1.0).
    """
    # Grab the current baseline config
    cfg = default_config()

    # Run the full stability check
    check = validate_solver_stability(cfg)

    # Overall gate must pass
    assert check.ok is True, f"default_config failed stability check: {check.messages}"
    # dt_sde margin should be very comfortable at the defaults (2 / 0.10 = 20, divided by 0.10 = 20x)
    assert check.dt_sde_margin_ratio >= 1.10
    # Multi-rate identity: 5 * 0.10 / 0.50 = 1.0 exactly
    assert check.multirate_identity_ratio == pytest.approx(1.0, abs=1e-6)


def test_tuned_default_baseline_survives_at_least_70pct():
    """
    Verify that the S4-T4 tuned default_config() produces the "stable but
    contestable" baseline the ticket asks for: alpha survives >70% of MC
    runs (and, by implication, < 100% -- overthrows are possible).

    Why : this is the direct empirical lock-in of the sigma=0.30 retune.
    Without this test, someone could edit sigma back down to a "safe" 0.05
    value at a later sprint, pass all the existing shape/stability tests,
    and silently regress the baseline into the "alpha is untouchable"
    regime that made the original noise_sweep phase transition so
    interesting in the first place.

    test: run 20 MC trials at default_config(), count how many had
    alpha_survived=True, assert the fraction is in (0.70, 1.00]. Lower
    bound is the ticket's explicit acceptance criterion; upper bound is
    the implicit "overthrows must be possible" criterion. Uses the same
    per-run seed bumping Jenny's MC driver does so reproducibility lines
    up with the sensitivity_report numbers.
    """
    # Lazy import so test collection is cheap; run_single_trial pulls a lot of
    # the pipeline in transitively.
    from macaque_sim.config.sim_config import default_config
    from macaque_sim.main import run_single_trial

    # Build the tuned baseline config and seed bump the way the MC runner does
    cfg = default_config()
    base_seed = cfg.random_seed
    n_trials = 20
    survived = 0

    # Run 20 MC trials with distinct seeds; a smaller N trades variance for
    # test wall-clock -- 20 at 0.15s/trial lands us ~3s, acceptable for the
    # normal pytest run.
    for run_i in range(n_trials):
        run_cfg = cfg.to_dict()
        run_cfg["random_seed"] = base_seed + run_i
        logger = run_single_trial(run_cfg)
        if getattr(logger, "alpha_survived", False):
            survived += 1

    # Ticket acceptance: > 70% survival (stable baseline) AND < 100% is the
    # empirical witness that overthrows can happen at these defaults. The
    # upper bound is loose enough that a rare all-survive run doesn't flake
    # the test, but tight enough that a sigma=0 regression regression trips it.
    survival_rate = survived / n_trials
    assert survival_rate > 0.70, (
        f"Tuned default_config() only survived {survived}/{n_trials} runs "
        f"({survival_rate:.1%}); ticket acceptance requires > 70%."
    )
    # Soft sanity: confirm we're not in the trivial sigma=0 always-stable regime.
    # 20 runs with p_survive == 1.0 is possible even at sigma=0.30 (observed
    # survival ~74%), so only flag if 20/20 survived, which would be very
    # strong evidence we regressed back toward a noise-starved default.
    # Binomial prob of 20/20 at p=0.74 is 0.74^20 ~= 0.0024, so this is a
    # 1-in-400 flake even at the correct defaults -- accept that rate.
    assert survived < n_trials, (
        f"All {n_trials}/{n_trials} trials survived; this is possible at the "
        f"tuned defaults but strongly suggests sigma was regressed downward. "
        f"Verify default_config().sigma."
    )


def test_solver_stability_flags_tight_dt_margin():
    """
    Verify that validate_solver_stability fails when dt_sde is pushed to
    within 10% of the 2/lambda stability boundary.

    Why : em_step's hard-raise fires exactly at dt_sde >= 2/lambda. But in
    a parameter sweep we want some headroom so a small jitter in lambda or
    dt doesn't land us right on the boundary. This test locks in that
    headroom requirement.

    test: mutate a config copy so dt_sde is 95% of 2/lambda (5% below cap),
    confirm check.ok is False and the message mentions the boundary.
    """
    # Make a mutable copy of the default config and push dt_sde near the boundary
    cfg = dataclasses.replace(default_config(), dt_sde=0.95 * (2.0 / default_config().lambda_decay))
    # Bypass SimConfig.validate() by constructing directly -- this test is about the
    # sensitivity-layer safety margin, not the config layer's hard stability gate.

    # Expectation: 2/lambda / dt_sde ~= 1.053, which is < 1.10 safety margin
    # NOTE: SimConfig.validate() itself would also warn about a too-tight dt --
    # we're explicitly dodging that here to test the sensitivity-layer margin.
    check = validate_solver_stability(cfg)

    # Overall gate should fail
    assert check.ok is False
    # dt margin field should report the actual ratio
    assert check.dt_sde_margin_ratio < 1.10
    # Message should mention the 2/lambda boundary
    assert any("2/lambda" in m for m in check.messages)


# --- Wilson CI tests ---

def test_wilson_ci_zero_n_returns_nan():
    lo, hi = wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_wilson_ci_brackets_proportion_for_typical_case():
    # 44 / 50 = 0.88, the upward-biased grooming-strategy survival rate.
    lo, hi = wilson_ci(44, 50)
    assert 0.0 <= lo < 0.88 < hi <= 1.0
    # Wilson is asymmetric near boundaries; lower tail is wider here.
    assert (0.88 - lo) > (hi - 0.88)


def test_wilson_ci_at_boundaries_stays_in_unit_interval():
    # All-success boundary: p = 1.0. Wilson never returns hi > 1.
    lo_all, hi_all = wilson_ci(50, 50)
    assert hi_all == 1.0
    assert 0.9 < lo_all < 1.0  # finite lower bound

    # All-failure boundary: p = 0.0. Wilson never returns lo < 0.
    lo_none, hi_none = wilson_ci(0, 50)
    assert lo_none == 0.0
    assert 0.0 < hi_none < 0.1


def test_wilson_ci_widens_with_smaller_n():
    # Same proportion (0.5), smaller n -> wider interval.
    lo_big, hi_big = wilson_ci(50, 100)
    lo_small, hi_small = wilson_ci(5, 10)
    assert (hi_small - lo_small) > (hi_big - lo_big)


def test_cell_alpha_survival_ci_routes_through_wilson():
    cell = _cell((0,), {"sigma": 0.3}, survive_rate=0.6, n_runs=20)
    # 12 / 20 = 0.6 (the _cell helper uses round())
    lo, hi = cell_alpha_survival_ci(cell)
    expected_lo, expected_hi = wilson_ci(12, 20)
    assert lo == pytest.approx(expected_lo)
    assert hi == pytest.approx(expected_hi)
