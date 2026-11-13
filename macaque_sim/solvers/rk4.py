# rk4.py -- ODE stepper (RK4) + renormalization
# Assignee: Dev B (S1-T5 skeleton, S2-T3 full ODE, S2-T4 renorm)
# See: project_management/tasks/sprint_2/S2-T3_rk4_power_iteration.md
# See: project_management/tasks/sprint_2/S2-T4_renormalization.md

"""
Power-iteration ODE solver for the macaque centrality score P(t).

The continuous equation we're discretizing:

    dP_i/dt = sum_j(A_ij * P_j) - P_i * sum_{k,j}(A_{kj} * P_j)

First term is "influence flowing in from my allies"; second term is a Rayleigh
normalizer that analytically keeps sum(P) = 1. RK4 gives us O(h^5) local
truncation error, plenty for the ODE side of the multirate loop.

gotcha (Sprint 2, S2-T3): the Rayleigh normalizer keeps P on the simplex in
*continuous* time. In discrete RK4 with a big h, the intermediate stages
(P_t1, P_t2, P_t3) can leave the simplex, which blows up the next stage's
matmul and feeds NaN back up through k4. With default_config() the envelope
A has row-sums ~29 at equilibrium and dt_ode=0.5 -- way past the RK4
stability radius for this linearization. So I adaptively sub-step inside
rk4_step based on a cheap spectral-radius estimate. See `_substep_count`.
"""

import warnings

import numpy as np


class SimulationDivergenceError(RuntimeError):
    """
    Raised when renormalize() sees a P vector we can't recover from.

    Why : all-zero means every monkey lost all influence simultaneously, which
    is mathematically impossible under the CIR+power-iteration dynamics --
    it only happens if something upstream (alpha_removal of the *whole* troop,
    NaN overflow in the SDE, etc.) has broken the invariant. Better to bail
    the whole trial cleanly so the Monte Carlo runner can log+skip, rather
    than quietly propagating NaN for the rest of the sweep.
    """


def renormalize(P: np.ndarray) -> np.ndarray:
    """
    Force sum(P) == 1 exactly, with guards against the three ways this can fail.

    Order matters here:
      1. NaN check first -- a single NaN poisons any sum or comparison that
         follows, so detect it before we try to do math.
      2. Negative-clamp next -- RK4 intermediates can dip tiny amounts below
         zero from floating-point noise. Floor to 0 and carry on.
      3. Zero-sum last -- after the clamp it's still possible everything
         collapsed to 0, in which case we've lost all information and raise.
    """
    P = np.asarray(P, dtype=float)

    # gotcha: np.isnan on a masked/object array silently returns False. We're
    # guaranteed float here because of the asarray cast above, so np.isnan is safe.
    if np.any(np.isnan(P)):
        raise SimulationDivergenceError(
            "Power vector contains NaN; upstream solver diverged."
        )

    # Negative entries get clamped rather than raising -- this is the expected
    # failure mode when RK4 overshoots a near-zero P_i, and it's recoverable.
    if np.any(P < 0):
        P = np.maximum(0.0, P)

    total = float(P.sum())
    if total <= 0.0:
        raise SimulationDivergenceError(
            f"Power vector sum is {total}; no valid renormalization exists."
        )

    return P / total


def _power_deriv(A: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    The right-hand side of the power-iteration ODE.

        f(A, P)[i] = (A @ P)[i] - P[i] * (1^T A P)

    gotcha (Dev A, 2026-04-09): the older inline version did `P_t @ (one @ A @ P_t)`
    with four `@`s -- but `one @ A @ P_t` is already a scalar, and matmul'ing a
    vector with a 0-d scalar raises ValueError. Pulled the derivative out into
    its own helper and used `*` for the scalar multiply so it can't regress.
    """
    AP = A @ P
    # 1^T A P is the total flux; just sum AP instead of building a ones-vector.
    rayleigh = float(AP.sum())
    return AP - P * rayleigh


def _rk4_single(A: np.ndarray, P: np.ndarray, h: float) -> np.ndarray:
    """One un-renormalized RK4 step. Split out so sub-stepping can reuse it."""
    k1 = _power_deriv(A, P)
    k2 = _power_deriv(A, P + 0.5 * h * k1)
    k3 = _power_deriv(A, P + 0.5 * h * k2)
    k4 = _power_deriv(A, P + h * k3)
    return P + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _substep_count(A: np.ndarray, h: float, budget: float = 1.5) -> int:
    """
    How many sub-steps we need to stay inside RK4's stability region.

    The power-iteration ODE linearizes around any fixed A with spectral radius
    rho(A); classical RK4 is stable when |h * rho| <~ 2.8. Use the max row sum
    as a cheap Gershgorin upper bound on rho, and aim for h_sub * rho <= budget
    (1.5 leaves comfortable headroom under 2.8 and avoids wasted steps when A
    happens to be well-conditioned).
    """
    # np.abs() because A is non-negative in the live pipeline, but this keeps
    # the bound honest if a test hands us a signed A.
    rho_upper = float(np.max(np.abs(A).sum(axis=1)))
    if rho_upper <= 0.0:
        # A is all-zero (post-alpha-removal on a tiny troop, etc.) -- one step is fine.
        return 1
    return max(1, int(np.ceil(h * rho_upper / budget)))


def rk4_step(A: np.ndarray, P_t: np.ndarray, h: float) -> np.ndarray:
    """
    Advance P one ODE macro-step using classical RK4 on the power-iteration ODE.

    Transparently sub-steps internally when A is too hot for the requested h,
    re-projecting onto the simplex after each sub-step so the Rayleigh normalizer
    stays honest in the discrete world.

    Parameters
    ----------
    A : (N, N) ndarray
        Alliance matrix driving the ODE. In the live pipeline this is the
        envelope-max A from the multirate controller.
    P_t : (N,) ndarray
        Power vector at start of step. Expected to sum to ~1.
    h : float
        Macro-step size (config.dt_ode in the live pipeline).

    Returns
    -------
    (N,) ndarray summing to exactly 1.
    """
    n_sub = _substep_count(A, h)
    h_sub = h / n_sub

    P = P_t
    for _ in range(n_sub):
        P = _rk4_single(A, P, h_sub)
        # Between sub-steps, pull P back onto the simplex. Without this the
        # sub-step trick buys nothing -- the whole point is keeping intermediate
        # magnitudes sane for the next matmul.
        P = renormalize(P)

    return P


def check_converge(A: np.ndarray, P: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Is P an eigenvector of A? Cheap wedge-norm check.

    Uses the fact that A @ P is parallel to P iff the outer products
    (A@P) ⊗ P and P ⊗ (A@P) are symmetric, i.e. their difference has zero norm.
    Way cheaper than an eigendecomposition and never used in the live sim --
    purely a diagnostic for the S2-T3 convergence test.
    """
    AP = A @ P
    wedge = np.outer(AP, P) - np.outer(P, AP)
    return bool(np.linalg.norm(wedge) < tol)
