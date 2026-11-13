# test_stability.py -- Solver stability validation (S3-T4)
# pytest macaque_sim/tests/test_stability.py -v

"""
Stress the coupled SDE/ODE system against edge-case parameter combinations
that are known to break naive numerical solvers. These aren't correctness
tests -- they're "does the solver silently produce garbage under duress"
tests, which is the whole point of the Sprint 3 stability gate.

Why separate from test_solvers.py: unit solver tests use well-conditioned
inputs. These tests deliberately stay in the hostile regime (high sigma, low
lambda, boundary dt, large N) so we can document the solver's stable envelope
for Sprint 4's Monte Carlo sweeps.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from macaque_sim.config.scenarios import build_grooming_matrix
from macaque_sim.config.sim_config import default_config
from macaque_sim.initializer.mri_init import initialize
from macaque_sim.main import run_simulation
from macaque_sim.solvers.euler_maruyama import em_step


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- EM mean-square stability boundary ---------------------------------------


def test_em_rejects_dt_at_stability_boundary():
    """
    Verify that dt_sde == 2/lambda is rejected (strict-less-than constraint).

    Why : the CIR Euler-Maruyama mean-square stability theorem requires
    dt < 2/lambda strictly. Allowing equality is a silent corruption source --
    the method is neutrally stable there, which in practice means "diverges
    the first time noise pushes it outside the basin". We want the guard
    in em_step to fire at the boundary, not just past it.

    test: set dt_sde exactly at the boundary, expect ValueError.
    """
    cfg = default_config()
    cfg.dt_sde = 2.0 / cfg.lambda_decay
    A = np.zeros((cfg.N, cfg.N))
    B = np.zeros((cfg.N, cfg.N))

    with pytest.raises(ValueError, match="Stability violation"):
        em_step(A, B, cfg, _rng())


def test_em_accepts_dt_just_below_stability_boundary():
    """
    Verify that dt_sde just under 2/lambda is accepted and produces a finite,
    non-negative A.

    Why : the complementary test to the boundary rejection. Without this we'd
    never know if someone over-tightened the guard and started rejecting
    configs that are actually stable.

    test: set dt_sde = 0.99 * 2/lambda, run 500 steps, assert finite and >=0.
    """
    cfg = default_config()
    cfg.dt_sde = 0.99 * (2.0 / cfg.lambda_decay)
    B = build_grooming_matrix(cfg)
    A, _, _ = initialize(cfg, B)

    rng = _rng()
    for _ in range(500):
        A = em_step(A, B, cfg, rng)

    assert np.all(np.isfinite(A)), "A went non-finite just below the stability boundary"
    assert np.all(A >= 0.0), "A went negative just below the stability boundary"


# --- hostile parameter regimes ------------------------------------------------


def test_high_sigma_stays_bounded():
    """
    Verify that 5x default sigma does not produce NaN, inf, or negative A over
    a long SDE trajectory.

    Why : high-sigma regimes are where the Feller condition fails and the
    sqrt(A)*dW diffusion can push A below zero every step. The max(0, ...)
    floor is our only safety net there. If it's ever removed or the diffusion
    grows a sign bug, this test catches it.

    test: sigma = 5x default, 1000 SDE steps starting from the MRI init A,
    assert finite + non-negative.
    """
    cfg = default_config()
    cfg.sigma = cfg.sigma * 5.0  # high stress
    B = build_grooming_matrix(cfg)
    A, _, _ = initialize(cfg, B)

    rng = _rng()
    for _ in range(1000):
        A = em_step(A, B, cfg, rng)

    assert np.all(np.isfinite(A)), "High sigma produced non-finite A"
    assert np.all(A >= 0.0), "High sigma produced negative A (floor failed)"


def test_low_lambda_keeps_alliance_finite():
    """
    Verify that very low lambda (slow alliance decay) does not cause A to
    blow up to infinity.

    Why : with lambda close to zero, alliances accumulate without dissipating.
    The mean-reversion equilibrium B/lambda drifts toward infinity, and any
    bug in the drift term scaling would surface as runaway growth. This pins
    down that the SDE's drift is still acting as a restoring force.

    test: lambda = 0.01 (10x slower than default). Also have to bump dt_sde
    down so we stay inside 2/lambda. Run, assert A remains finite after
    500 steps -- not necessarily small, just not inf/nan.
    """
    cfg = default_config()
    cfg.lambda_decay = 0.01
    cfg.dt_sde = 0.05  # well under 2/lambda = 200
    B = build_grooming_matrix(cfg)
    A, _, _ = initialize(cfg, B)

    rng = _rng()
    for _ in range(500):
        A = em_step(A, B, cfg, rng)

    assert np.all(np.isfinite(A)), "Low lambda let A go non-finite"


# --- full-pipeline stress -----------------------------------------------------


def test_large_troop_completes_within_budget():
    """
    Verify that a N=50 full-pipeline run finishes in a reasonable wall-clock
    budget and produces clean output.

    Why : N=50 is the upper end of the documented troop-size range. The
    alliance matrix scales O(N^2) and RK4 scales O(N^2) per stage, so if
    someone introduces an accidental O(N^3) loop (e.g. Python-level nested
    iteration over pairs), this test's runtime balloons from seconds to
    minutes. Hard numeric budget catches it.

    test: N=50, T=10 (short), run_simulation, assert under 30s wall-clock
    and the usual finite/sum-to-one invariants hold. Budget is generous on
    purpose -- the point is catching 10x regressions, not fine-grained perf.
    """
    cfg = default_config()
    cfg.N = 50
    cfg.T = 10.0

    t0 = time.perf_counter()
    log = run_simulation(cfg)
    elapsed = time.perf_counter() - t0

    assert elapsed < 30.0, f"N=50 / T=10 run took {elapsed:.1f}s; suspect O(N^3) regression"
    P_hist = log.get_power_history()
    assert np.all(np.isfinite(P_hist))
    assert np.allclose(P_hist.sum(axis=1), 1.0, atol=1e-10)
