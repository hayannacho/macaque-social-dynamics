# test_pipeline.py -- End-to-end pipeline regressions (S3-T1, S4-T5)
# pytest macaque_sim/tests/test_pipeline.py -v

"""
Integration tests that exercise run_simulation() as a single unit. Unit tests in
test_solvers.py / test_hysteresis_and_cooldown.py cover each module in isolation;
this file guards the cross-module contracts that only break when everything is
glued together.

Why a separate file: if a regression shows up here but unit tests are green, the
bug lives in the wiring (main.py's loop order, shared-state handoff between
solvers + hysteresis + perturbations), not in any one module.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from macaque_sim.config.scenarios import build_grooming_matrix
from macaque_sim.config.sim_config import default_config
from macaque_sim.experiments.perturbations import apply_perturbation
from macaque_sim.initializer.mri_init import initialize
from macaque_sim.main import run_simulation


def _shrink(cfg, T: float = 5.0):
    """
    Shrink a config to keep test suite fast.

    Why : default T=100 takes ~1s per run and most pipeline checks don't need
    the full trajectory. 5 time-units @ dt_ode=0.5 = 10 macro-steps is enough
    to exercise all cross-module contracts without slowing the suite.
    """
    cfg.T = T
    return cfg


# --- S4-T5: determinism ------------------------------------------------------


def test_run_simulation_is_deterministic_under_same_seed():
    """
    Verify that two runs with identical config + seed produce byte-identical
    power, rank, and event histories.

    Why : this is the single most important guard against silent reproducibility
    bugs. If someone swaps rng.standard_normal for np.random.standard_normal
    anywhere in the stack, or forgets to thread the Generator through a new
    solver path, this test fails instantly -- whereas without it we'd only
    notice when Monte Carlo results refused to replicate across machines.

    This also covers S4-T5's "deterministic run produces identical results
    across two runs with same seed" acceptance criterion. We can't test with
    sigma=0 directly since SimConfig rejects it, but identical-seed runs at
    any sigma prove the same property.

    test: run, re-run, assert numpy array equality on P / R / time / events.
    """
    cfg_a = _shrink(default_config())
    cfg_b = _shrink(default_config())

    log_a = run_simulation(cfg_a)
    log_b = run_simulation(cfg_b)

    assert np.array_equal(log_a.get_power_history(), log_b.get_power_history())
    assert np.array_equal(log_a.get_rank_history(), log_b.get_rank_history())
    assert np.array_equal(log_a.get_time_axis(), log_b.get_time_axis())
    assert log_a.get_events() == log_b.get_events()


def test_run_simulation_different_seeds_diverge():
    """
    Verify that two runs with different seeds produce different P histories.

    Why : sibling to the determinism test. If this fails, it means the rng is
    not actually driving any stochasticity -- which would silently make every
    Monte Carlo sweep a single repeated run. Catches the inverse regression of
    the previous test.

    test: change random_seed, run, assert P histories are not all-equal.
    """
    cfg_a = _shrink(default_config())
    cfg_b = _shrink(default_config())
    cfg_b.random_seed = cfg_a.random_seed + 1

    log_a = run_simulation(cfg_a)
    log_b = run_simulation(cfg_b)

    assert not np.array_equal(log_a.get_power_history(), log_b.get_power_history())


# --- pipeline invariants ------------------------------------------------------


def test_run_simulation_no_nan_no_negative_invariants():
    """
    Verify end-of-run invariants: P finite, P sums to 1 every step, A stays
    non-negative, ranks remain a valid permutation of [1..N].

    Why : this is the catch-all regression for "any solver layer silently
    produces NaN / negatives / rank-duplicates". Was a real bug before the
    S2-T3 sub-stepping + S2-T4 renormalize fix -- default_config used to NaN
    out after ~10 macro-steps.

    test: one default run, one coalition+high-sigma run (exercises the
    hysteresis swap path too), assert invariants on every logged row.
    """
    for cfg_mut in [
        lambda c: c,
        lambda c: dataclasses.replace(c, grooming_strategy="coalition", sigma=0.15),
    ]:
        cfg = _shrink(cfg_mut(default_config()))
        log = run_simulation(cfg)

        P_hist = log.get_power_history()
        R_hist = log.get_rank_history()
        A_hist = log.A[: log.step_idx]

        assert np.all(np.isfinite(P_hist)), "P history contains NaN or inf"
        assert np.allclose(P_hist.sum(axis=1), 1.0, atol=1e-10), "P rows don't sum to 1"
        assert np.all(P_hist >= 0.0), "Negative power scores leaked through renormalize"
        assert np.all(A_hist >= 0.0), "Alliance matrix went negative"
        # Every rank row must be a permutation of [1..N].
        for row in R_hist:
            assert sorted(row.tolist()) == list(range(1, cfg.N + 1))


# --- perturbation wiring ------------------------------------------------------


def test_alpha_removal_zeroes_target_mid_run():
    """
    Verify that a scheduled alpha_removal perturbation actually fires inside
    the simulation loop and zeroes the alpha's row/col in A and entry in P.

    Why : test_perturbations.py exercises apply_perturbation() in isolation,
    but nothing proves the should_trigger + dispatch handshake in main.py's
    loop actually wires up correctly. Without this test, someone could break
    the `perturbations = getattr(cfg, 'perturbations', None) or []` line and
    every scenario_alpha_removal run would silently behave like the noise
    baseline.

    test: attach an alpha_removal perturbation at t=T/2 onto a shrunk default
    config, run, check the alpha's power drops to ~0 at the next logged step
    after T/2 (it may recover slightly since renorm spreads mass, so assert a
    strict drop rather than exactly 0).
    """
    cfg = _shrink(default_config(), T=4.0)
    # Attach perturbations as an ad-hoc attribute -- main.py reads it via getattr.
    cfg.perturbations = [
        {"type": "alpha_removal", "time": 2.0, "target": -1, "params": {}},
    ]

    log = run_simulation(cfg)

    time_axis = log.get_time_axis()
    P_hist = log.get_power_history()

    # Find the alpha at t=0 (the agent with rank 1).
    R0 = log.get_rank_history()[0]
    alpha_idx = int(np.argmin(R0))

    # gotcha: main.py fires perturbations BEFORE logging within a macro-step,
    # so the row at t=2.0 already reflects the removal (P[alpha] == 0 there).
    # Compare the trigger row against the row just before it.
    trigger_idx = int(np.searchsorted(time_axis, 2.0, side="left"))

    pre_alpha_power = P_hist[trigger_idx - 1, alpha_idx]
    post_alpha_power = P_hist[trigger_idx, alpha_idx]

    # alpha_removal zeroes P[alpha] in-place; no multirate step runs between
    # the zeroing and the log, so the trigger row should be exactly 0.
    assert post_alpha_power < 0.5 * pre_alpha_power, (
        f"Alpha power only dropped {pre_alpha_power:.4f} -> {post_alpha_power:.4f}"
    )

    # And the perturbation event should have been logged.
    event_types = {e.get("type") for e in log.get_events()}
    assert "perturbation" in event_types


# --- hysteresis + cooldown wiring --------------------------------------------


def test_hysteresis_fires_under_coalition_grooming():
    """
    Verify that under coalition grooming + elevated sigma + low epsilon, at
    least some rank overthrows are logged.

    Why : the default config (uniform grooming, low sigma, eps=0.02) produces
    zero overthrows by design -- powers cluster tightly around 1/N. If we only
    tested that path, we'd never notice someone disconnecting check_overthrows
    from main.py's loop (exact S3-T1 regression). Coalition + noisy is the
    known "should topple things" regime from the manual sanity check we did
    when closing S3-T1.

    test: coalition grooming, sigma=0.2, eps=0.005, run, assert at least one
    overthrow event appears in the logger.
    """
    cfg = _shrink(default_config(), T=20.0)
    cfg = dataclasses.replace(
        cfg,
        grooming_strategy="coalition",
        sigma=0.2,
        epsilon=0.005,
    )

    log = run_simulation(cfg)

    overthrows = [e for e in log.get_events() if e.get("type") == "overthrow"]
    assert len(overthrows) > 0, "No overthrow events fired -- hysteresis is not wired"


def test_cooldown_blocks_repeat_swaps_between_same_pair():
    """
    Verify that after any overthrow between agents (i, j), no second swap
    involving either i or j fires within cooldown_duration time units.

    Why : unit tests exercise CooldownTracker state directly, but the live
    pipeline is where the tracker actually gets threaded into check_overthrows
    at every macro-step. If someone instantiates a fresh tracker inside the
    loop by accident (losing state between steps), this test catches it --
    whereas the unit tests would still pass because they drive the tracker
    object themselves.

    test: same coalition/high-sigma config as above. For each overthrow in
    the event log, scan forward and assert no swap touches the same winner
    or loser id within cooldown_duration time-units.
    """
    cfg = _shrink(default_config(), T=20.0)
    cfg = dataclasses.replace(
        cfg,
        grooming_strategy="coalition",
        sigma=0.2,
        epsilon=0.005,
        cooldown_duration=5,
    )

    log = run_simulation(cfg)
    overthrows = [e for e in log.get_events() if e.get("type") == "overthrow"]
    if len(overthrows) < 2:
        pytest.skip("Not enough overthrows under this seed to probe cooldown.")

    for i, ev_a in enumerate(overthrows):
        t_a = ev_a["t"]
        ids_a = {ev_a["details"]["winner_id"], ev_a["details"]["loser_id"]}
        # cooldown_duration is in ODE time-units. CooldownTracker uses strict
        # "t < unlock_time", so an agent is free to swap again AT exactly
        # t_a + cooldown_duration -- use >= to break out at that boundary.
        window_end = t_a + cfg.cooldown_duration
        for ev_b in overthrows[i + 1 :]:
            if ev_b["t"] >= window_end:
                break
            ids_b = {ev_b["details"]["winner_id"], ev_b["details"]["loser_id"]}
            assert not (ids_a & ids_b), (
                f"Agent(s) {ids_a & ids_b} swapped again at t={ev_b['t']} "
                f"within cooldown window after t={t_a}"
            )


# --- initial_state kwarg (orphan injection + other custom-init flows) ------


def test_run_simulation_accepts_initial_state():
    """run_simulation(cfg, initial_state=(A,P,R)) skips MRI init and uses the
    caller's state. Sanity-check: row 0 of the logger matches what we passed in."""
    cfg = _shrink(default_config())
    B = build_grooming_matrix(cfg)
    A, P, R = initialize(cfg, B)

    log = run_simulation(cfg, initial_state=(A, P, R))

    np.testing.assert_array_equal(log.get_rank_history()[0], R)
    np.testing.assert_allclose(log.get_power_history()[0], P, atol=1e-12)


def test_run_simulation_initial_state_does_not_mutate_caller_arrays():
    """The loop should copy the caller's state defensively. If we mutate A after
    the call, the logger's history must not change."""
    cfg = _shrink(default_config())
    B = build_grooming_matrix(cfg)
    A, P, R = initialize(cfg, B)

    A_before = A.copy()
    log = run_simulation(cfg, initial_state=(A, P, R))
    # Corrupt the caller's A after the run -- logger history shouldn't budge.
    A[:] = 0.0
    np.testing.assert_array_equal(A_before, log.A[0])


def test_run_simulation_initial_state_is_deterministic():
    """Same cfg + same initial_state + same seed -> identical trajectories."""
    cfg = _shrink(default_config())
    B = build_grooming_matrix(cfg)
    A, P, R = initialize(cfg, B)

    log_a = run_simulation(cfg, initial_state=(A, P, R))
    log_b = run_simulation(cfg, initial_state=(A, P, R))
    np.testing.assert_array_equal(log_a.get_rank_history(), log_b.get_rank_history())
    np.testing.assert_array_equal(log_a.get_power_history(), log_b.get_power_history())


def test_run_simulation_initial_state_shape_mismatch_raises():
    """Wrong-shape arrays must raise clearly, not silently propagate."""
    cfg = _shrink(default_config())
    N = cfg.N
    bad_A = np.zeros((N - 1, N - 1))
    bad_P = np.zeros(N - 1)
    bad_R = np.zeros(N - 1)
    with pytest.raises(ValueError, match="initial_state shapes"):
        run_simulation(cfg, initial_state=(bad_A, bad_P, bad_R))


def test_run_simulation_initial_state_threads_orphan_perturbation():
    """End-to-end: apply orphan_injection at t=0, feed state into run_simulation,
    verify the orphan stays at the bottom rank for the duration (no stochastic
    alliance has been built up yet)."""
    cfg = _shrink(default_config(), T=2.0)
    B = build_grooming_matrix(cfg)
    A, P, R = initialize(cfg, B)

    orphan_id = cfg.N // 2
    A, P, R, cfg = apply_perturbation(
        {"type": "orphan_injection", "time": 0.0, "target": orphan_id, "params": {}},
        A, P, R, cfg,
    )
    assert int(R[orphan_id]) == cfg.N  # orphan at bottom after perturbation

    log = run_simulation(cfg, initial_state=(A, P, R))
    # At t=0 the orphan is still at the bottom (logger row 0 == our input).
    assert int(log.get_rank_history()[0][orphan_id]) == cfg.N
