# test_solvers.py -- Unit tests for Euler-Maruyama, RK4, and multi-rate controller
# pytest macaque_sim/tests/test_solvers.py -v
# python -m pytest macaque_sim/tests/test_solvers.py -v
import dataclasses

import pytest
import numpy as np

from macaque_sim.solvers.euler_maruyama import em_step
from macaque_sim.solvers.multirate import multirate_step
from macaque_sim.solvers.rk4 import (
    SimulationDivergenceError,
    renormalize,
    rk4_step,
)
from macaque_sim.config.sim_config import default_config


def _rng(seed: int = 42) -> np.random.Generator:
    """
    Shared helper that builds a seeded numpy Generator for each test.

    Why : Sprint 2 em_step takes an rng argument; every test needs one. Wrapping
    np.random.default_rng in a helper means if I ever need to swap the generator
    type (e.g. to a PCG64DXSM or a seeded MT19937), I change one line here instead
    of hunting through 15 test bodies.
    """
    return np.random.default_rng(seed)


def test_em_step_shape():
    """
    Verify that the Euler-Maruyama step preserves matrix dimensions.

    Why : The alliance matrix A flows through the entire pipeline
    (power iteration, rank updates, visualization). If the shape changes unexpectedly,
    downstream code will silently broadcast or fail cryptically. Better to catch
    shape mismatches immediately at the solver level.

    test: Feed in an (N, N) alliance matrix and a (N, N) grooming
    matrix, and verify the output is also (N, N). Seems trivial, but it's the
    contract that the rest of the system relies on.
    """
    cfg = default_config()
    N = cfg.N

    # Create baseline alliance and grooming matrices (both N x N)
    A = np.zeros((N, N))
    B = np.zeros((N, N))

    # Step forward once using the Euler-Maruyama scheme
    A_next = em_step(A, B, cfg, _rng())

    # Check that the output shape matches the input shape
    assert A_next.shape == (N, N), f"Expected shape ({N}, {N}), got {A_next.shape}"


def test_em_step_dt_validation():
    """
    Critical stability check: Euler-Maruyama is mathematically unstable if the 
    timestep is too large relative to the mean-reversion rate.
    
    The theory: For Cox-Ingersoll-Ross (CIR) processes, the method only converges 
    mean-square stable when dt < 2/lambda. If we violate this, the numerical solution 
    can blow up to infinity, physically representing "alliances oscillating wildly."
    
    test: If someone configures a timestep that's too large (which 
    could happen by accident in parameter sweeps), we reject it loudly rather than 
    silently producing garbage numbers downstream.
    
    Set dt = 2/lambda exactly (the forbidden boundary), and confirm the 
    solver raises a ValueError before attempting any computation.
    """
    cfg = default_config()
    N = cfg.N
    
    # Push the timestep to the stability limit (exactly dt = 2/lambda)
    cfg.dt_sde = 2.0 / cfg.lambda_decay
    
    # Initialize dummy alliance and grooming matrices
    A = np.zeros((N, N))
    B = np.zeros((N, N))
    
    # Expect the solver to immediately reject this configuration
    with pytest.raises(ValueError, match="Stability violation"):
        em_step(A, B, cfg, _rng())


def test_em_step_non_negative():
    """
    Biological constraint: Alliance strengths cannot be negative.
    
    Why : The biological model assumes alliances are strictly non-negative. 
    Negative alliances don't exist (you can't have "anti-cooperation"). Additionally, 
    the stochastic noise term in the full SDE uses sqrt(A), which requires A >= 0 to be 
    mathematically real. Zero-flooring is our numerical guardrail.
    
    test: Run the solver 10 times sequentially to accumulate numerical 
    errors. The ground-truth math has no noise, so it's deterministic. Even though 
    rapid decay can momentarily push below zero in finite-precision arithmetic, our 
    zero-clamping should catch this and keep everything >= 0.
    
    The scenario: One monkey grooming a group (low effort, B = 0.05), no existing 
    alliances (A starts at zero), and the process should stay non-negative throughout.
    """
    cfg = default_config()
    N = cfg.N
    
    # Start with fresh slate: no pre-existing alliances
    A = np.zeros((N, N))
    
    # Modest, uniform grooming across the troop (except no self-grooming, because 
    # that's biologically nonsensical)
    B = np.full((N, N), 0.05)
    np.fill_diagonal(B, 0.0)
    
    # Integrate forward 10 micro-timesteps to accumulate any floating-point errors
    A_current = A.copy()
    rng = _rng()
    for step_num in range(10):
        A_current = em_step(A_current, B, cfg, rng)
        # After each step, verify we haven't dipped below zero anywhere in the matrix
        assert np.all(A_current >= 0.0), \
            f"Alliance matrix went negative at step {step_num}! Values: {A_current.min()}"


@pytest.mark.parametrize("matrix_type,size_offset,error_pattern", [
    ("A", 5, "alliance matrix A has shape"),
    ("B", -2, "Grooming matrix B has shape"),
])
def test_em_step_shape_mismatch(matrix_type, size_offset, error_pattern):
    """
    Input validation: Catch shape mismatches early for both A and B.
    
    Why : If you accidentally create mismatched matrices (e.g., A as 25x25 when 
    config.N = 30, or B with wrong size), the solver should reject immediately 
    rather than propagate silent broadcasting errors downstream.
    
    test: Pass matrices with the wrong dimensions and verify the solver 
    rejects each with a clear error message.
    
    This parametrized test covers both alliance matrix A and grooming matrix B 
    mismatches with a single test function.
    """
    cfg = default_config()
    N = cfg.N
    
    if matrix_type == "A":
        # Deliberately create A with wrong dimensions
        A = np.zeros((N + size_offset, N + size_offset))
        B = np.zeros((N, N))
    else:  # B
        A = np.zeros((N, N))
        # Create B with wrong dimensions
        B = np.zeros((N + size_offset, N + size_offset))
    
    # Expect the solver to catch this and raise an error
    with pytest.raises(ValueError, match=error_pattern):
        em_step(A, B, cfg, _rng())


def test_em_step_zero_grooming_mean_decays():
    """
    Verify that without grooming maintenance, the *expected* alliance strength decays.

    Why : This test was the Sprint 1 monotonic-decay sanity check -- with B = 0, we
    had dA = -lambda*A*dt and A should shrink every single step. In Sprint 2 the stochastic
    term sigma*sqrt(A)*sqrt(dt)*Z is the same order of magnitude as the drift, so any
    individual step can easily go up. The biological intuition we actually want to
    preserve is that the *mean* alliance level still decays when no grooming is
    happening -- otherwise a free-floating stochastic process is just noise with no
    downward pull, which would be wrong for memory decay.

    test: Start with moderate alliances (0.5 everywhere off-diagonal), zero out the
    grooming matrix B, step forward many times, and assert the off-diagonal mean
    ends up below where it started. Also verify the max(0, ...) floor kept us
    non-negative the whole way.
    """
    cfg = default_config()
    N = cfg.N

    # Start with moderate alliances already in place (no self-loops -- monkeys
    # don't have relationships with themselves).
    A0 = np.full((N, N), 0.5)
    np.fill_diagonal(A0, 0.0)

    # Nobody is grooming anybody this run -- B is zero everywhere.
    B = np.zeros((N, N))

    # Integrate forward 200 micro-steps. That's plenty to let deterministic decay
    # accumulate past the noise wiggle envelope.
    A_current = A0.copy()
    rng = _rng()
    for _ in range(200):
        A_current = em_step(A_current, B, cfg, rng)
        # Non-negativity guardrail should hold every single step even with noise.
        assert np.all(A_current >= 0.0), \
            "Zero-grooming trajectory must stay non-negative (floor guardrail)"

    # Compare starting mean vs ending mean on the off-diagonal entries only --
    # the diagonal is pinned at zero by construction and would skew the average.
    off_diag = ~np.eye(N, dtype=bool)
    assert A_current[off_diag].mean() < A0[off_diag].mean(), \
        "Without grooming, mean alliance should decay over many steps"


def test_em_step_strong_grooming():
    """
    Boundary condition: Intense grooming should strengthen alliances.

    The opposite case from zero_grooming: if the grooming matrix is large (lots of
    social investment), the positive drift should build relationships. Starting from
    scratch (zero alliances) with vigorous grooming should generate positive values.
    """
    cfg = default_config()
    N = cfg.N

    # Start with no pre-existing alliances
    A = np.zeros((N, N))

    # Intense grooming from every monkey to every other monkey
    B = np.full((N, N), 1.0)
    np.fill_diagonal(B, 0.0)  # Except they don't groom themselves

    # Take one step with strong grooming
    A_next = em_step(A, B, cfg, _rng())

    # All elements should remain non-negative (floor guard, and sqrt(A)*Z contributes
    # nothing when starting at zero so there is no first-step noise anyway)
    assert np.all(A_next >= 0.0), \
        "Strong grooming shouldn't produce negative alliances"

    # Off-diagonal elements should have grown from zero (positive drift from grooming)
    off_diag_indices = np.triu_indices_from(A_next, k=1)
    off_diag_values = A_next[off_diag_indices]
    assert np.all(off_diag_values > 0.0), \
        "Off-diagonal alliances should become strictly positive under strong grooming"


# --- Sprint 2 (S2-T1) stochastic CIR tests ---------------------------------


def test_em_step_seeded_determinism():
    """
    Verify that two em_step calls with identically-seeded Generators produce
    bit-for-bit identical outputs.

    Why : Monte Carlo reproducibility depends on this. Every scientific claim we
    make from the phase diagram downstream relies on being able to re-run a
    specific trajectory and get the exact same numbers back. If there's any
    hidden np.random.* global state in the code path, this test will catch it.
    It also protects against the subtler failure mode where two rngs seeded
    the same way somehow evolve differently because of import-order side effects
    or lazy initialization.

    test: Build a moderate (A, B) pair, call em_step twice with two fresh
    Generators both seeded with the same integer, and assert the outputs are
    bitwise equal (np.array_equal, not np.allclose -- we want zero drift).
    """
    cfg = default_config()
    N = cfg.N

    # Moderate alliances already in place, no self-loops (biology).
    A = np.full((N, N), 0.2)
    np.fill_diagonal(A, 0.0)
    # And a realistic grooming matrix to go with them.
    B = np.full((N, N), 0.05)
    np.fill_diagonal(B, 0.0)

    # Two INDEPENDENT Generators, both seeded with 2024. If em_step has any
    # hidden randomness, these two calls will diverge.
    A_next_1 = em_step(A.copy(), B.copy(), cfg, np.random.default_rng(2024))
    A_next_2 = em_step(A.copy(), B.copy(), cfg, np.random.default_rng(2024))

    # np.array_equal, NOT allclose -- for reproducibility we want zero drift.
    assert np.array_equal(A_next_1, A_next_2), \
        "Same seed should produce bitwise identical updates for Monte Carlo reproducibility."


def test_em_step_long_run_non_negative():
    """
    Verify that over 10,000 consecutive em_steps, the alliance matrix never goes
    negative anywhere.

    Why : This is the S2-T1 acceptance criterion -- the spec explicitly calls for
    a long-run non-negativity test because the whole point of the CIR formulation
    (sqrt(A) diffusion + max(0, ...) floor) is to guarantee alliances stay in
    biologically valid territory. A single-step test won't catch slow drift or
    rare-event undershoots; 10k steps will.

    test: Pick a Feller-satisfying (A, B) so the continuous theory is on our
    side, then run em_step 10,000 times from the same rng and assert every
    element of the final A is >= 0 and finite. If the floor ever lets something
    slip through, or if noise accumulation blows up to NaN, this catches it.
    """
    cfg = default_config()
    N = cfg.N

    # Seed the process at a modest alliance level so we're not starting it
    # pathologically close to zero where the floor could mask non-negativity bugs.
    A = np.full((N, N), 0.1)
    np.fill_diagonal(A, 0.0)

    # Pick B so the Feller condition 2*B > sigma^2 holds on every off-diagonal edge.
    # Default sigma = 0.05, so sigma^2 = 0.0025; 2*0.05 = 0.10 > 0.0025, comfortable margin.
    B = np.full((N, N), 0.05)  # 2*B = 0.1 > sigma^2 = 0.0025, Feller satisfied
    np.fill_diagonal(B, 0.0)

    # Run it 10,000 times with a shared rng so noise accumulates over time.
    rng = _rng()
    for _ in range(10_000):
        A = em_step(A, B, cfg, rng)

    # Finiteness check first -- we want to catch inf/NaN explosions separately
    # from the negativity check because the error messages are different.
    assert np.all(np.isfinite(A)), "Alliance matrix should never explode to inf/NaN"
    # Then the actual non-negativity assertion.
    assert np.all(A >= 0.0), "Alliance matrix should stay non-negative over 10k steps"


def test_em_step_mean_reversion():
    """
    Verify that the long-run time-average of A[i,j] converges to the CIR stationary
    mean B[i,j] / lambda.

    Why : This is the other half of the S2-T1 acceptance criteria. It's not enough
    for the process to stay non-negative -- it also has to have the right long-run
    *level*. For the Cox-Ingersoll-Ross SDE, the stationary mean is exactly B/lambda
    (the drift equilibrium). If my implementation has a bug in the drift sign, or
    computes sqrt(A) on the wrong thing, the mean will drift to something else and
    the whole scientific picture downstream (phase diagrams, overthrow frequencies)
    would be off by a constant factor. Catching that here is way cheaper than
    discovering it during the Sprint 4 experiments.

    test: Pick a flat B, start A at the theoretical stationary mean to shorten
    the burn-in, warm up for 1,000 steps, then time-average over 5,000 more
    steps and check the empirical mean is within 15% of B/lambda. The tolerance is
    loose because the max(0, ...) floor introduces a slight positive bias, and
    Monte Carlo noise over 5k samples is still nontrivial.
    """
    cfg = default_config()
    N = cfg.N

    # Flat grooming matrix so the CIR stationary mean B/lambda is the same on every edge.
    # That makes the off-diagonal time-average a well-defined thing to compare
    # against a single theoretical value.
    B_val = 0.04
    B = np.full((N, N), B_val)
    np.fill_diagonal(B, 0.0)

    # Start A at the theoretical stationary mean so we only pay a short burn-in
    # for the process to forget this initial condition rather than a long
    # transient to climb from 0 up to equilibrium.
    expected = B_val / cfg.lambda_decay
    A = np.full((N, N), expected)
    np.fill_diagonal(A, 0.0)

    # A different seed than the default so this test's noise path is independent
    # of the other S2-T1 tests and failures are easier to bisect.
    rng = _rng(seed=7)
    burn_in = 1_000
    samples = 5_000

    # Burn-in: let the process forget the initial condition. We throw these steps
    # away without recording anything.
    for _ in range(burn_in):
        A = em_step(A, B, cfg, rng)

    # Mask the diagonal out before averaging because it's pinned at zero and would
    # drag the mean down.
    off_diag = ~np.eye(N, dtype=bool)
    # Accumulate sample means in a running sum -- avoids allocating a big array of
    # (samples, N, N) slices just to call .mean() at the end.
    running_sum = 0.0
    for _ in range(samples):
        A = em_step(A, B, cfg, rng)
        running_sum += A[off_diag].mean()

    empirical_mean = running_sum / samples
    # Tolerance chosen loosely -- Monte-Carlo noise plus the floor bias (max(0, ...) clamp)
    # both nudge the empirical mean, but it should be within ~15% of B/lambda.
    assert abs(empirical_mean - expected) < 0.15 * expected, \
        f"Mean {empirical_mean:.4f} should converge to B/lambda = {expected:.4f}"


# --- Sprint 2 (S2-T2) multi-rate controller tests --------------------------


def _stable_multirate_config():
    """
    Build a config that keeps RK4 inside its stability region for the alliance
    magnitudes these tests use.

    NOTE (Sprint 2): the default_config has `dt_ode = 0.5` with a CIR stationary
    alliance mean of B/lambda ~= 0.5, which gives h*lambda_max(A) roughly 7+ -- well above the
    ~2.78 RK4 stability boundary. In practice, default_config() + multirate
    diverges within a handful of macro-steps, which is exactly the kind of edge
    case Dev B and I will have to characterize in S3-T4 stability validation.
    But for *this* unit test I want to measure the multirate plumbing itself,
    not the RK4 stability boundary, so I shrink sigma and dt_ode until I'm
    comfortably inside the stable region. Think of it as isolating the unit
    under test from a known-sharp-edge parameter regime.
    """
    return dataclasses.replace(
        default_config(),
        sigma=0.02,
        dt_sde=0.05,
        dt_ode=0.10,
        lambda_decay=0.10,
        micro_steps_per_macro=2,   # matches dt_ode/dt_sde = 0.10/0.05
    )


def _well_posed_ab(cfg):
    """
    Build a (A, B) pair that satisfies the Feller condition so the stochastic CIR
    term behaves as theory predicts.

    Why : a couple of the multirate tests below run the solver many times, and I
    don't want the Feller warning to fire on every single step -- it both clutters
    the pytest output and is a false positive since we're inside the Feller
    regime on purpose.
    """
    N = cfg.N
    # Small uniform starting alliances, zero diagonal (no self-grooming).
    A = np.full((N, N), 0.05)
    np.fill_diagonal(A, 0.0)
    # B small enough that 2*B = 0.02 > sigma^2 = 0.0004 -- Feller satisfied with
    # about an order of magnitude of margin.
    B = np.full((N, N), 0.01)  # 2*B = 0.02 > sigma^2 = 0.0004
    np.fill_diagonal(B, 0.0)
    return A, B


def _uniform_power(N):
    """
    Start with a normalized uniform power vector -- every monkey equally strong.

    Why : this is the maximum-entropy initial condition, which means any
    subsequent drift in P is caused by the simulation itself rather than by
    us seeding the test with a pre-baked hierarchy.
    """
    P = np.ones(N) / N
    return P


def test_multirate_step_shapes_and_normalization():
    """
    Verify that one multirate_step call preserves shapes, stays finite, and leaves
    P summing to exactly 1.

    Why : the macro-step is the interface main.py's sim loop calls once per
    dt_ode. If it returns wrong shapes, NaNs, or unnormalized P, every single
    downstream consumer (hysteresis, logger, perturbations) will either crash
    or produce nonsense. This is the shape/normalization contract test.

    test: One call with a single (A, P, B) triple on the stable config, then
    assert output shapes match the inputs, every entry is finite, and P
    renormalized correctly.
    """
    cfg = _stable_multirate_config()
    N = cfg.N
    # Build Feller-safe state and a uniform starting P.
    A, B = _well_posed_ab(cfg)
    P = _uniform_power(N)

    # One macro-step through the controller.
    A_next, P_next = multirate_step(A, P, B, cfg, _rng())

    # Shape contract: A_next is (N,N) and P_next is (N,), same as the inputs.
    assert A_next.shape == (N, N)
    assert P_next.shape == (N,)
    # Finiteness contract: no NaN, no inf.
    assert np.all(np.isfinite(A_next)), "A_next must be finite"
    assert np.all(np.isfinite(P_next)), "P_next must be finite"
    # Normalization contract: P_next.sum() must be 1 to within floating point.
    # Dev B's rk4_step does the renorm internally; if multirate ever starts
    # double-normalizing (or skipping it), this assertion fires.
    assert abs(P_next.sum() - 1.0) < 1e-10, \
        f"P_next should renormalize to 1, got sum={P_next.sum():.12f}"


def test_multirate_step_non_negative():
    """
    Verify that A stays non-negative and P stays normalized across 50 consecutive
    macro-steps.

    Why : a single macro-step proving shapes is one thing, but the real sim runs
    hundreds to thousands of these back-to-back. If there's a slow leak in
    either invariant -- e.g. the floor only holds for small noise, or the
    renormalization drifts slightly and compounds -- we won't see it in a
    single-step test. Running 50 steps catches compounding bugs without
    being so slow that it drags down pytest runs.

    test: Start from the Feller-safe config, step 50 times, assert every step
    keeps A >= 0 and P.sum() ~= 1.
    """
    cfg = _stable_multirate_config()
    N = cfg.N
    # Feller-safe starting state.
    A, B = _well_posed_ab(cfg)
    P = _uniform_power(N)

    rng = _rng()
    # 50 macro-steps is a good balance -- long enough to compound bugs, short
    # enough that the test stays under a few hundred ms.
    for _ in range(50):
        A, P = multirate_step(A, P, B, cfg, rng)
        # After every macro-step: A non-negative, P normalized. Assertions are
        # inside the loop so we know exactly which step broke the invariant.
        assert np.all(A >= 0.0), "Alliance matrix must stay non-negative"
        assert abs(P.sum() - 1.0) < 1e-9, "P must stay normalized each macro-step"


def test_multirate_step_m_equals_one_equivalence():
    """
    Verify that when M = 1 (dt_ode == dt_sde), multirate_step produces the same
    result as calling em_step followed by rk4_step directly.

    Why : this is the S2-T2 acceptance criterion, and it's a great regression
    guard. If someone (future me, Dev B, anyone) changes the envelope logic to
    something subtly different, or forgets that Dev B's rk4_step renormalizes
    internally, this test will fire. The M=1 case is the simplest possible
    multirate configuration so the equivalence is unambiguous.

    test: Override dt_sde = dt_ode so M = 1. Run multirate_step with an
    rng seeded to 1234. Reset a fresh Generator with the same seed and
    run em_step + rk4_step manually. Assert the two paths produce the same
    A (bitwise) and the same P (allclose with zero tolerance).
    """
    # Force M = 1 by setting micro_steps_per_macro directly (and aligning dt_sde
    # and dt_ode so the equivalence check makes physical sense). Everything else
    # inherited from the stable multirate config so RK4 doesn't blow up.
    cfg = dataclasses.replace(
        _stable_multirate_config(),
        dt_sde=0.1,
        dt_ode=0.1,
        micro_steps_per_macro=1,
    )
    N = cfg.N
    A, B = _well_posed_ab(cfg)
    P = _uniform_power(N)

    # Path 1: run through the multirate controller with a freshly-seeded rng.
    rng_mr = np.random.default_rng(1234)
    A_mr, P_mr = multirate_step(A.copy(), P.copy(), B.copy(), cfg, rng_mr)

    # Path 2: reset a brand-new Generator with the SAME seed, then run the two
    # solver calls manually in the same order multirate does internally.
    rng_direct = np.random.default_rng(1234)
    A_direct = em_step(A.copy(), B.copy(), cfg, rng_direct)
    P_direct = rk4_step(A_direct, P.copy(), cfg.dt_ode)

    # A should be bitwise identical because the envelope with M=1 is just the
    # single post-step matrix (see multirate.py seeding logic).
    assert np.array_equal(A_mr, A_direct), \
        "With M=1 the multirate A update should equal a single em_step"
    # P comes from the same rk4_step call in both paths, so zero-tolerance
    # allclose is fine here too.
    assert np.allclose(P_mr, P_direct, atol=0, rtol=0), \
        "With M=1 the multirate P update should equal a single rk4_step on the post-step A"


def test_multirate_step_mean_coupling_collapses_to_max_at_M_equals_1():
    """
    Verify that coupling_mode="mean" produces bit-identical output to "max"
    when M = 1, since with a single post-step matrix the "envelope" under
    either mode is just that matrix.

    Why : this is the analogue of test_multirate_step_m_equals_one_equivalence
    for the post-review coupling_mode field. The two modes diverge only when
    the running aggregate sees more than one matrix; at M = 1 they MUST agree.
    Locks in the invariant that flipping coupling_mode never breaks the M=1
    collapse property.
    """
    base = dataclasses.replace(
        _stable_multirate_config(),
        dt_sde=0.1,
        dt_ode=0.1,
        micro_steps_per_macro=1,
    )
    cfg_max = dataclasses.replace(base, coupling_mode="max")
    cfg_mean = dataclasses.replace(base, coupling_mode="mean")
    N = base.N
    A, B = _well_posed_ab(base)
    P = _uniform_power(N)

    rng_max = np.random.default_rng(4321)
    A_max, P_max = multirate_step(A.copy(), P.copy(), B.copy(), cfg_max, rng_max)

    rng_mean = np.random.default_rng(4321)
    A_mean, P_mean = multirate_step(A.copy(), P.copy(), B.copy(), cfg_mean, rng_mean)

    assert np.array_equal(A_max, A_mean), \
        "M=1: coupling_mode='mean' must produce identical A to 'max'"
    assert np.allclose(P_max, P_mean, atol=0, rtol=0), \
        "M=1: coupling_mode='mean' must produce identical P to 'max'"


def test_multirate_step_mean_coupling_differs_from_max_at_M_gt_1():
    """
    Sanity: with M > 1 and nonzero noise, the mean and max envelopes must
    actually differ (otherwise the post-review coupling_mode field is doing
    no work). The max envelope is the running element-wise maximum; the mean
    is the unbiased average. They agree only on degenerate inputs.
    """
    base = _stable_multirate_config()  # M=2, dt_sde=0.05, dt_ode=0.10
    cfg_max = dataclasses.replace(base, coupling_mode="max")
    cfg_mean = dataclasses.replace(base, coupling_mode="mean")
    N = base.N
    A, B = _well_posed_ab(base)
    P = _uniform_power(N)

    rng_max = np.random.default_rng(7777)
    A_max, P_max = multirate_step(A.copy(), P.copy(), B.copy(), cfg_max, rng_max)
    rng_mean = np.random.default_rng(7777)
    A_mean, P_mean = multirate_step(A.copy(), P.copy(), B.copy(), cfg_mean, rng_mean)

    # Both modes consume the same RNG draws (em_step is identical between modes).
    # The final A_current returned by multirate_step is the M-th micro-step
    # state regardless of envelope mode, so A_max and A_mean DO match by
    # design (the envelope is internal to multirate_step). The RK4 input
    # matrix differs, so P_max != P_mean.
    assert np.array_equal(A_max, A_mean), \
        "Final A_current is the same in both modes (envelope is internal)."
    assert not np.allclose(P_max, P_mean, atol=1e-12), \
        "P should differ between max and mean coupling at M>1 (different RK4 input)."


def test_adaptive_epsilon_tracker_basic_behavior():
    """
    Verify EpsilonTracker basic invariants (post-review C1).

    Why : the discrete-event layer relies on this tracker returning the static
    fallback when fewer than 2 deltas have been observed (since std requires
    at least 2 samples), and 2 * std(pooled deltas) once enough history is
    in the rolling window. Locks in the contract main.py depends on.
    """
    from macaque_sim.events.epsilon_tracker import EpsilonTracker

    tracker = EpsilonTracker(window_size=4, factor=2.0)
    fallback = 0.02

    # Empty buffer: returns fallback.
    assert tracker.current_epsilon(fallback) == fallback

    # One sample: still no delta available, returns fallback.
    tracker.update(np.array([0.1, 0.2, 0.7]))
    assert tracker.current_epsilon(fallback) == fallback

    # Second sample: one delta available; std of one sample is 0 (ddof=1
    # would be NaN), so the tracker should still return the fallback.
    tracker.update(np.array([0.15, 0.18, 0.67]))
    eps = tracker.current_epsilon(fallback)
    # Either fallback (if guarded) or 2 * std of the 3-element delta vector.
    assert eps == fallback or eps > 0

    # Several more samples: tracker now returns 2 * std of pooled deltas.
    for shift in [0.02, -0.01, 0.005, -0.003]:
        tracker.update(np.array([0.15 + shift, 0.18 - shift, 0.67]))
    eps = tracker.current_epsilon(fallback)
    assert 0 < eps < 1.0


def test_adaptive_epsilon_pipeline_changes_swap_behavior():
    """
    Sanity: with the same seed, epsilon_mode='adaptive' produces a different
    trajectory than 'fixed' under coalition grooming -- confirming the adaptive
    branch is actually wired through to check_overthrows. Without this test,
    the field could silently fall back to fixed and pass production behavior.
    """
    from macaque_sim.main import run_single_trial
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg_fixed = {"sigma": 0.30, "grooming_strategy": "coalition",
                     "random_seed": 12345, "epsilon_mode": "fixed"}
        cfg_adapt = {"sigma": 0.30, "grooming_strategy": "coalition",
                     "random_seed": 12345, "epsilon_mode": "adaptive",
                     "epsilon_window": 20, "epsilon_factor": 2.0}
        lf = run_single_trial(cfg_fixed)
        la = run_single_trial(cfg_adapt)
    # Same seed, different epsilon mode: the swap counts should differ. If they
    # match, adaptive is silently no-op'ing and the wiring is broken.
    assert lf.num_swaps != la.num_swaps, (
        f"adaptive epsilon should change swap count vs fixed; got "
        f"fixed={lf.num_swaps}, adaptive={la.num_swaps}"
    )


def test_multirate_step_shape_validation():
    """
    Verify that shape mismatches between (A, P, B) and config.N are caught before
    any solver work happens.

    Why : same philosophy as the em_step shape checks. If the sim loop hands us
    mismatched arrays (because someone truncated N somewhere upstream), I want
    the ValueError to fire here, at the controller boundary, not after M
    micro-steps silently broadcast wrong shapes and RK4 dies with a cryptic
    "matmul: Input operand 1 does not have enough dimensions" message.

    test: Deliberately pass in wrong-sized arrays for each of A, P, B one at a
    time and assert multirate_step raises with a clear shape-name message.
    """
    cfg = default_config()
    N = cfg.N
    # Start from well-formed state so only the one bad argument trips the check.
    A, B = _well_posed_ab(cfg)
    P = _uniform_power(N)

    # A with wrong shape -> should mention "Alliance matrix A has shape".
    with pytest.raises(ValueError, match="Alliance matrix A has shape"):
        multirate_step(np.zeros((N + 1, N + 1)), P, B, cfg, _rng())

    # P with wrong shape -> should mention "Power vector P has shape".
    with pytest.raises(ValueError, match="Power vector P has shape"):
        multirate_step(A, np.zeros(N + 1), B, cfg, _rng())

    # B with wrong shape -> should mention "Grooming matrix B has shape".
    with pytest.raises(ValueError, match="Grooming matrix B has shape"):
        multirate_step(A, P, np.zeros((N + 1, N + 1)), cfg, _rng())


# NOTE: the old test_multirate_ratio_validation test was removed along with
# the _micro_steps_per_macro helper. M now lives on SimConfig as
# `micro_steps_per_macro` and SimConfig.validate() owns the "M must be a
# positive int" check -- that's Dev C's territory, and belongs in
# test_config.py if anyone ever wants to regression-test it.


# =============================================================================
# Sprint 2 — RK4 power-iteration ODE (S2-T3) and renormalization (S2-T4) tests
# =============================================================================


def test_rk4_step_sum_to_one_after_step():
    """
    Verify that rk4_step always returns a power vector that sums to exactly 1.

    Why : the whole point of the Rayleigh normalizer in the ODE is that in
    continuous time sum(P) is an invariant. In discrete RK4 the invariant leaks
    at O(h) per stage, so rk4_step has to lean on renormalize() at the end. If
    the sum drifts, downstream hysteresis comparisons across P_i start using
    different "units" per timestep and rank swaps go haywire.

    test: Feed a well-conditioned symmetric A and a uniform P, step once, and
    check the L1 sum is 1 within machine epsilon.
    """
    N = 10
    rng = np.random.default_rng(0)
    A = rng.random((N, N)) * 0.1
    np.fill_diagonal(A, 0.0)
    P = np.ones(N) / N

    P_next = rk4_step(A, P, h=0.5)

    assert P_next.shape == (N,)
    assert np.isclose(P_next.sum(), 1.0, atol=1e-12)


def test_rk4_step_non_negative_output():
    """
    Verify that rk4_step never produces negative power scores.

    Why : P_i is an eigenvector-centrality proxy -- Perron-Frobenius says it
    stays non-negative when A is non-negative, and the hysteresis engine treats
    P_i as a raw "social standing" scalar. A negative entry would turn the
    ordering inside out and quietly corrupt every swap comparison downstream.

    test: Step once on a non-negative A from a uniform P. The renormalize()
    call at the tail of rk4_step should have clamped any tiny negative drift.
    """
    N = 8
    rng = np.random.default_rng(1)
    A = rng.random((N, N)) * 0.1
    np.fill_diagonal(A, 0.0)
    P = np.ones(N) / N

    P_next = rk4_step(A, P, h=0.1)

    assert np.all(P_next >= 0.0)


def test_rk4_step_converges_to_dominant_eigenvector():
    """
    Verify that for a fixed symmetric A, repeated rk4_step calls drive P toward
    the dominant eigenvector of A.

    Why : this is the whole reason we chose the continuous power-iteration ODE
    over a static eigensolve -- the ODE is supposed to track the leading
    eigenvector of A as A drifts. If it doesn't even converge to the right
    vector for a frozen A, the time-varying case has no chance.

    Biological parallel: under a *stable* social environment (constant A), the
    centrality scores should settle into the monkey-by-monkey dominance
    ordering that A encodes, not bounce around forever.

    test: Pick a small symmetric positive A with a known dominant eigenvector
    (computed via numpy.linalg.eig, used only as a ground truth for the test),
    step 1000 times with a modest h, and assert the wedge norm of the final P
    against that eigenvector is below tolerance.
    """
    # Keep N small so numpy.linalg.eig is fast and tolerances are tight.
    A = np.array([
        [0.0, 0.4, 0.3, 0.2],
        [0.4, 0.0, 0.5, 0.1],
        [0.3, 0.5, 0.0, 0.6],
        [0.2, 0.1, 0.6, 0.0],
    ])
    # Ground truth: pick eigenvector of the dominant (largest-abs) eigenvalue.
    eigvals, eigvecs = np.linalg.eig(A)
    dom = int(np.argmax(np.abs(eigvals)))
    target = np.abs(eigvecs[:, dom])
    target = target / target.sum()

    P = np.ones(4) / 4.0
    for _ in range(1000):
        P = rk4_step(A, P, h=0.05)

    # Compare up to L1 renormalization; eigenvectors are only defined up to scale.
    err = np.linalg.norm(P - target, ord=1)
    assert err < 1e-6, f"RK4 did not converge to dominant eigvec; L1 err={err:.3e}"


def test_rk4_step_substeps_under_hot_A():
    """
    Verify that rk4_step stays finite and normalized even when h * rho(A) is
    well past the vanilla RK4 stability radius.

    Why : this is the S3-T1 regression. default_config has dt_ode=0.5 and the
    envelope A at equilibrium has row-sums ~29, which puts h*rho around 15 --
    way past RK4's ~2.8 stability limit on the power-iteration linearization.
    Before the internal sub-stepping fix, this silently produced NaN. The
    adaptive sub-step count should keep us inside the stable region.

    test: Build an A with big row-sums on purpose, step once with a large h,
    and assert the output is finite and sums to 1.
    """
    N = 10
    # Row-sums ~= N-1 ~= 9 for this construction; pair with h=1.0 -> h*rho ~= 9.
    A = np.ones((N, N)) - np.eye(N)
    P = np.ones(N) / N

    P_next = rk4_step(A, P, h=1.0)

    assert np.all(np.isfinite(P_next)), "RK4 produced non-finite entries on hot A"
    assert np.isclose(P_next.sum(), 1.0, atol=1e-12)


# --- Renormalize (S2-T4) ------------------------------------------------------


def test_renormalize_normal_vector():
    """
    Verify that a well-behaved P renormalizes to sum exactly 1.

    Why : the baseline case -- every rk4 sub-step ends in this branch. If the
    normal path drifts by even 1e-8, that error compounds over T/dt_ode steps
    and hysteresis starts firing on numerical noise.

    test: Take a uniform vector scaled to sum 1.3, renormalize, check sum == 1.
    """
    P = np.full(5, 0.26)  # sums to 1.3
    Q = renormalize(P)

    assert np.isclose(Q.sum(), 1.0, atol=1e-15)
    # All entries equal since input was uniform.
    assert np.allclose(Q, np.full(5, 0.2))


def test_renormalize_negative_entry_clamps():
    """
    Verify that a single sub-zero entry is clamped to 0 and the rest sum to 1.

    Why : RK4 intermediate stages can briefly dip P_i negative when a monkey's
    power is very low; the clamp is the recoverable-error path. We explicitly
    do NOT raise here -- raising would kill Monte Carlo trials on a routine
    floating-point flinch.

    test: Hand-build a vector with one small negative entry, renormalize, check
    that the offending entry is now 0 and the rest still sum to 1.
    """
    P = np.array([0.5, -1e-10, 0.3, 0.2])
    Q = renormalize(P)

    assert Q[1] == 0.0
    assert np.isclose(Q.sum(), 1.0, atol=1e-15)


def test_renormalize_all_zero_raises():
    """
    Verify that an all-zero P raises SimulationDivergenceError.

    Why : all-zero means the system has lost *every* unit of social influence
    simultaneously, which our continuous dynamics physically can't produce. If
    we see this, something catastrophic happened upstream (full-troop alpha
    removal, NaN overflow in the SDE). Raising cleanly lets Dev E's Monte Carlo
    runner log+skip the trial instead of silently poisoning aggregate metrics.

    test: Renormalize np.zeros(4); expect SimulationDivergenceError.
    """
    P = np.zeros(4)
    with pytest.raises(SimulationDivergenceError):
        renormalize(P)


def test_renormalize_nan_raises():
    """
    Verify that a NaN anywhere in P raises SimulationDivergenceError.

    Why : NaN poisons every subsequent sum, comparison, and np.argmax. It
    doesn't "heal" -- once it's there the whole trial is garbage. Same
    bail-the-trial rationale as the zero-sum case.

    test: Inject a NaN into an otherwise healthy P, renormalize, expect the
    divergence exception.
    """
    P = np.array([0.25, 0.25, np.nan, 0.25, 0.25])
    with pytest.raises(SimulationDivergenceError):
        renormalize(P)
