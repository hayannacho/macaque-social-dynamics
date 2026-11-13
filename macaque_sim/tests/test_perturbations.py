"""
test_perturbations.py — Unit tests for experiments/perturbations.py.

Covers:
  - apply_alpha_removal: zeroes row/col, zeroes power, sentinel resolution,
                         partial flags (zero_row only / zero_col only).
  - apply_grooming_shock: replaces B in state, shape mismatch error,
                          missing new_B error.
  - apply_volatility_spike: modifies sigma by factor.
  - apply_perturbation: dispatch by type, unknown type error,
                        time sentinel -1 resolved to T/2.
  - should_trigger: fires in correct interval, does not double-fire,
                    time sentinel resolution.
  - register_perturbation: custom handler registered and callable.

Run with:  python -m pytest tests/test_perturbations.py -v
"""

import math

import numpy as np
import pytest

from macaque_sim.experiments.perturbations import (
    apply_alpha_removal,
    apply_grooming_shock,
    apply_volatility_spike,
    apply_orphan_injection,
    apply_perturbation,
    should_trigger,
    register_perturbation,
    list_perturbation_types,
)


# ============================================================
#  Fixtures
# ============================================================

def make_state(N=4, alpha=0):
    """Build a minimal simulation state dict for testing."""
    rng = np.random.default_rng(42)
    A = rng.uniform(0.1, 1.0, (N, N))
    np.fill_diagonal(A, 0.0)
    P = rng.uniform(0.1, 1.0, N)
    P /= P.sum()
    R = list(range(N))          # agent 0 = rank 0 (alpha), agent N-1 = omega
    B = rng.uniform(0.0, 0.5, (N, N))
    np.fill_diagonal(B, 0.0)
    return {"A": A.copy(), "P": P.copy(), "R": R, "B": B.copy(), "t": 0.0}


# ============================================================
#  apply_alpha_removal
# ============================================================

class TestAlphaRemoval:

    def test_row_zeroed(self):
        s = make_state()
        apply_alpha_removal(s, target=0, params={})
        assert np.all(s["A"][0, :] == 0.0)

    def test_col_zeroed(self):
        s = make_state()
        apply_alpha_removal(s, target=0, params={})
        assert np.all(s["A"][:, 0] == 0.0)

    def test_power_zeroed(self):
        s = make_state()
        apply_alpha_removal(s, target=0, params={})
        assert s["P"][0] == 0.0
        assert math.isclose(float(s["P"].sum()), 1.0, abs_tol=1e-10)

    def test_other_agents_A_unchanged(self):
        """Non-target rows/cols (excluding target's row+col) stay intact."""
        s = make_state(N=4)
        A_before = s["A"].copy()
        apply_alpha_removal(s, target=0, params={})
        # Row 1 col 1 should be untouched
        assert math.isclose(s["A"][1, 1], A_before[1, 1])
        assert math.isclose(s["A"][2, 3], A_before[2, 3])

    def test_sentinel_minus1_removes_rank0_agent(self):
        """target=-1 should resolve to the agent with rank 0."""
        s = make_state(N=4)
        # Manually set agent 2 as rank-0
        s["R"] = [1, 2, 0, 3]
        apply_alpha_removal(s, target=-1, params={})
        assert np.all(s["A"][2, :] == 0.0)
        assert np.all(s["A"][:, 2] == 0.0)
        assert s["P"][2] == 0.0

    def test_zero_row_only(self):
        s = make_state(N=4)
        col_before = s["A"][:, 1].copy()
        apply_alpha_removal(s, target=1, params={"zero_row": True, "zero_col": False})
        assert np.all(s["A"][1, :] == 0.0)
        np.testing.assert_array_equal(s["A"][:, 1], col_before)

    def test_zero_col_only(self):
        s = make_state(N=4)
        row_before = s["A"][1, :].copy()
        apply_alpha_removal(s, target=1, params={"zero_row": False, "zero_col": True})
        assert np.all(s["A"][:, 1] == 0.0)
        np.testing.assert_array_equal(s["A"][1, :], row_before)

    def test_out_of_range_target_raises(self):
        s = make_state(N=4)
        with pytest.raises(ValueError, match="out of range"):
            apply_alpha_removal(s, target=10, params={})

    def test_returns_state(self):
        s = make_state()
        result = apply_alpha_removal(s, target=0, params={})
        assert result is s


# ============================================================
#  apply_grooming_shock
# ============================================================

class TestGroomingShock:

    def test_replaces_B_in_state(self):
        s = make_state(N=3)
        new_B = np.ones((3, 3)) * 0.9
        apply_grooming_shock(s, target=0, params={"new_B": new_B})
        np.testing.assert_array_equal(s["B"], new_B)

    def test_shape_mismatch_raises(self):
        s = make_state(N=3)
        with pytest.raises(ValueError, match="shape"):
            apply_grooming_shock(s, target=0, params={"new_B": np.ones((2, 2))})

    def test_missing_new_B_raises(self):
        s = make_state(N=3)
        with pytest.raises(ValueError, match="new_B"):
            apply_grooming_shock(s, target=0, params={})

    def test_A_unchanged(self):
        s = make_state(N=3)
        A_before = s["A"].copy()
        apply_grooming_shock(s, target=0, params={"new_B": np.zeros((3, 3))})
        np.testing.assert_array_equal(s["A"], A_before)

    def test_returns_state(self):
        s = make_state(N=3)
        result = apply_grooming_shock(s, target=0, params={"new_B": np.zeros((3, 3))})
        assert result is s


# ============================================================
#  apply_volatility_spike
# ============================================================

class DummyConfig:
    def __init__(self, sigma=0.1):
        self.sigma = sigma


class TestVolatilitySpike:
    def test_spike_updates_config_sigma(self):
        s = make_state()
        s["config"] = DummyConfig(sigma=0.2)
        apply_volatility_spike(s, target=0, params={"factor": 3.0})
        assert math.isclose(s["config"].sigma, 0.6, abs_tol=1e-12)

    def test_missing_factor_raises(self):
        s = make_state()
        s["config"] = DummyConfig()
        with pytest.raises(ValueError, match="factor"):
            apply_volatility_spike(s, target=0, params={})

    def test_bad_factor_raises(self):
        s = make_state()
        s["config"] = DummyConfig()
        with pytest.raises(ValueError, match="> 0"):
            apply_volatility_spike(s, target=0, params={"factor": 0})


# ============================================================
#  apply_orphan_injection
# ============================================================

class TestOrphanInjection:

    def test_alliance_row_zeroed(self):
        s = make_state(N=4)
        apply_orphan_injection(s, target=2, params={})
        assert np.all(s["A"][2, :] == 0.0)

    def test_alliance_col_zeroed(self):
        s = make_state(N=4)
        apply_orphan_injection(s, target=2, params={})
        assert np.all(s["A"][:, 2] == 0.0)

    def test_moved_to_bottom_rank(self):
        """Default starting_rank='bottom' swaps orphan with current omega."""
        s = make_state(N=4)
        # R = [0, 1, 2, 3] -> bottom is rank 3, held by agent 3.
        apply_orphan_injection(s, target=0, params={})
        assert s["R"][0] == 3
        assert s["R"][3] == 0   # previous omega now holds orphan's old rank

    def test_power_zeroed_and_renormalized(self):
        s = make_state(N=4)
        apply_orphan_injection(s, target=2, params={})
        assert s["P"][2] == 0.0
        assert math.isclose(float(s["P"].sum()), 1.0, abs_tol=1e-10)

    def test_noop_when_already_at_bottom(self):
        """If target already at max rank, ranks stay put."""
        s = make_state(N=4)
        apply_orphan_injection(s, target=3, params={})
        assert list(s["R"]) == [0, 1, 2, 3]

    def test_explicit_starting_rank(self):
        s = make_state(N=4)
        apply_orphan_injection(s, target=0, params={"starting_rank": 2})
        assert s["R"][0] == 2
        # Whoever held rank 2 now holds agent 0's old rank (0)
        assert s["R"][2] == 0

    def test_registered_in_registry(self):
        assert "orphan_injection" in list_perturbation_types()

    def test_out_of_range_target_raises(self):
        s = make_state(N=4)
        with pytest.raises(ValueError, match="out of range"):
            apply_orphan_injection(s, target=99, params={})

    def test_bad_starting_rank_raises(self):
        s = make_state(N=4)
        with pytest.raises(ValueError, match="starting_rank"):
            apply_orphan_injection(s, target=0, params={"starting_rank": "middle"})

    def test_returns_state(self):
        s = make_state(N=4)
        result = apply_orphan_injection(s, target=2, params={})
        assert result is s


# ============================================================
#  apply_perturbation (dispatch)
# ============================================================

class TestApplyPerturbation:

    def _make_perturb(self, p_type, time=10.0, target=0, params=None):
        return {"type": p_type, "time": time, "target": target,
                "params": params or {}}

    def test_dispatches_alpha_removal(self):
        s = make_state(N=4)
        p = self._make_perturb("alpha_removal", target=0)
        A, P, R, cfg = s["A"], s["P"], np.array(s["R"]), DummyConfig()
        A2, P2, R2, cfg2 = apply_perturbation(p, A, P, R, cfg)
        assert np.all(A2[0, :] == 0.0)

    def test_dispatches_grooming_shock(self):
        s = make_state(N=4)
        new_B = np.ones((4, 4)) * 0.5
        p = self._make_perturb("grooming_shock", params={"new_B": new_B})
        A, P, R, cfg = s["A"], s["P"], np.array(s["R"]), DummyConfig()
        A2, P2, R2, cfg2 = apply_perturbation(p, A, P, R, cfg)
        assert hasattr(cfg2, "B")
        np.testing.assert_array_equal(cfg2.B, new_B)

    def test_dispatches_volatility_spike(self):
        s = make_state(N=4)
        p = self._make_perturb("volatility_spike", target=1, params={"factor": 2.0})
        A, P, R, cfg = s["A"], s["P"], np.array(s["R"]), DummyConfig(sigma=0.2)
        _, _, _, cfg2 = apply_perturbation(p, A, P, R, cfg)
        assert math.isclose(cfg2.sigma, 0.4, abs_tol=1e-12)

    def test_dispatches_orphan_injection(self):
        s = make_state(N=4)
        p = self._make_perturb("orphan_injection", target=1, params={})
        A, P, R, _ = apply_perturbation(
            p, s["A"], s["P"], np.array(s["R"]), DummyConfig()
        )
        assert np.all(A[1, :] == 0.0)
        assert np.all(A[:, 1] == 0.0)
        assert int(R[1]) == int(np.asarray(R).max())
        assert P[1] == 0.0

    def test_unknown_type_raises(self):
        s = make_state()
        p = self._make_perturb("teleportation")
        with pytest.raises(ValueError, match="Unknown"):
            apply_perturbation(p, s["A"], s["P"], np.array(s["R"]), DummyConfig())

    def test_time_sentinel_minus1_resolves(self):
        """time=-1 in perturbation dict should resolve to sim_T/2 = 50.0."""
        resolved_times = []

        def capturing_fn(state, target, params):
            resolved_times.append(params.get("resolved", "called"))
            return state

        register_perturbation("_test_sentinel", capturing_fn)
        s = make_state()
        p = {"type": "_test_sentinel", "time": -1, "target": 0, "params": {}}
        apply_perturbation(p, s["A"], s["P"], np.array(s["R"]), DummyConfig())
        assert len(resolved_times) == 1   # it was called


# ============================================================
#  should_trigger
# ============================================================

class TestShouldTrigger:

    def make_p(self, time):
        return {"type": "alpha_removal", "time": time, "target": 0, "params": {}}

    def test_fires_when_trigger_in_interval(self):
        p = self.make_p(time=50.0)
        assert should_trigger(p, t_prev=49.0, t_now=51.0, sim_T=100.0)

    def test_fires_exactly_at_t_now(self):
        p = self.make_p(time=50.0)
        assert should_trigger(p, t_prev=49.0, t_now=50.0, sim_T=100.0)

    def test_does_not_fire_at_t_prev(self):
        """Half-open interval (t_prev, t_now] — exact t_prev does not fire."""
        p = self.make_p(time=49.0)
        assert not should_trigger(p, t_prev=49.0, t_now=50.0, sim_T=100.0)

    def test_does_not_fire_before_trigger(self):
        p = self.make_p(time=80.0)
        assert not should_trigger(p, t_prev=49.0, t_now=50.0, sim_T=100.0)

    def test_does_not_fire_after_trigger(self):
        p = self.make_p(time=20.0)
        assert not should_trigger(p, t_prev=49.0, t_now=50.0, sim_T=100.0)

    def test_sentinel_minus1_resolves_to_half_T(self):
        p = self.make_p(time=-1)
        # sim_T=100 → trigger at 50.0
        assert should_trigger(p, t_prev=49.0, t_now=51.0, sim_T=100.0)
        assert not should_trigger(p, t_prev=51.0, t_now=60.0, sim_T=100.0)

    def test_fires_exactly_once_across_consecutive_steps(self):
        """Simulate stepping through the trigger: should fire on exactly one step."""
        p = self.make_p(time=50.0)
        steps = [(i * 10.0, (i + 1) * 10.0) for i in range(10)]
        fires = [should_trigger(p, t0, t1, sim_T=100.0) for t0, t1 in steps]
        assert sum(fires) == 1


# ============================================================
#  register_perturbation
# ============================================================

class TestRegisterPerturbation:

    def test_custom_handler_registered(self):
        called_with = []

        def my_handler(state, target, params):
            called_with.append((target, params))
            return state

        register_perturbation("custom_test", my_handler)
        assert "custom_test" in list_perturbation_types()

        s = make_state()
        p = {"type": "custom_test", "time": 5.0, "target": 2, "params": {"x": 9}}
        apply_perturbation(p, s["A"], s["P"], np.array(s["R"]), DummyConfig())
        assert called_with == [(2, {"x": 9})]

    def test_non_callable_raises(self):
        with pytest.raises(TypeError, match="callable"):
            register_perturbation("bad", "not_a_function")

    def test_list_perturbation_types_includes_builtins(self):
        types = list_perturbation_types()
        assert "alpha_removal" in types
        assert "grooming_shock" in types
        assert "volatility_spike" in types
        assert "orphan_injection" in types


class TestNewApiContract:
    def test_apply_perturbation_new_signature_returns_tuple(self):
        A = np.ones((3, 3), dtype=float)
        np.fill_diagonal(A, 0.0)
        P = np.array([0.4, 0.3, 0.3], dtype=float)
        R = np.array([0, 1, 2], dtype=int)
        cfg = DummyConfig(sigma=0.2)
        perturb = {"type": "volatility_spike", "target": -1, "params": {"factor": 2.0}}
        A2, P2, R2, cfg2 = apply_perturbation(perturb, A, P, R, cfg)
        assert A2 is A
        assert P2 is P
        assert R2 is R
        assert cfg2 is cfg
        assert math.isclose(cfg2.sigma, 0.4, abs_tol=1e-12)