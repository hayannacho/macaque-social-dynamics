# euler_maruyama.py -- SDE stepper (CIR square-root diffusion)
# Assignee: Dev A (S1-T4 skeleton, S2-T1 full implementation)
# See: project_management/tasks/sprint_1/S1-T4_em_skeleton.md
# See: project_management/tasks/sprint_2/S2-T1_em_cir_full.md

import warnings
import numpy as np

# In Sprint 1, we are using the SimConfig class from the config module
# so that our solver has strict access to exactly what parameters drove this run.
# If a circular dependency issue arises later during the integration sprint,
# this could be refactored into a generic dict.
from macaque_sim.config.sim_config import SimConfig


def em_step(
    A: np.ndarray,
    B: np.ndarray,
    config: SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Takes a single micro-step using Euler-Maruyama to update our macaque alliances (A).

    NOTE (Sprint 2, S2-T1): the Sprint 1 stub is gone -- I now actually do the thing the
    math was always supposed to do. Every edge gets a random Wiener kick on top of the
    tug-of-war drift. The continuous SDE we're approximating is:

        dA_ij = (B_ij - lambda * A_ij) * dt + sigma * sqrt(A_ij) * dW

    and the Euler-Maruyama discretization I'm implementing below is:

        A_next[i,j] = A[i,j] + (B[i,j] - lambda * A[i,j]) * dt_sde
                             + sigma * sqrt(max(0, A[i,j])) * sqrt(dt_sde) * Z[i,j]

    with Z[i,j] ~ N(0,1). The critical change from Sprint 1 is that noise is now
    drawn from a caller-supplied `rng` (a numpy Generator) -- Monte Carlo reproducibility
    depends on this, so every downstream call-site has to plumb it through.

    Biological intuition: the sqrt(A) factor on the diffusion is what keeps this from
    going haywire near zero. When two monkeys barely have any bond left, the noise kicks
    shrink with them -- you can't randomly gain/lose a huge chunk of a relationship that
    basically doesn't exist. Feller's condition (2*B > sigma^2) is the formal guarantee of
    strict positivity; I warn (not raise) when it's violated, and a hard max(0, ...)
    floor catches whatever the continuous theory doesn't.
    """

    # --- 1. Sanity Checks ---
    N = config.N

    # gotcha: If A and B don't match the config's troop size N, we'll get silent broadcasting
    # bugs later when computing eigenvector centrality. Failing loud and early here.
    if A.shape != (N, N):
        raise ValueError(f"Whoa, alliance matrix A has shape {A.shape}, but config.N is {N}")

    if B.shape != (N, N):
        raise ValueError(f"Grooming matrix B has shape {B.shape}, but config.N is {N}")

    # HUGE gotcha for CIR equations: Euler-Maruyama is notoriously unstable if you step
    # time too fast relative to the mean-reversion pull. The math physically explodes if
    # dt >= 2/lambda. I'm hard-stopping the run if the config parameters violate this.
    if config.dt_sde >= (2.0 / config.lambda_decay):
        raise ValueError(
            f"Stability violation! dt_sde ({config.dt_sde}) must be < 2/lambda ({2.0 / config.lambda_decay}). "
            f"If we run this, the alliances will oscillate to infinity. Tweak the config."
        )

    # gotcha: Feller's condition 2*B_ij > sigma^2 is the theoretical guarantee that the
    # continuous CIR stays strictly positive. If it's violated our sqrt(A)*dW noise
    # can overwhelm the drift, the process *wants* to go negative, and the only thing
    # stopping that is the hard floor further down. That's a numerical crutch, not
    # math, so I want the caller to see a yellow flag -- but not to crash, because
    # some scenario sweeps (high sigma, low B) are deliberately pushing into this regime
    # and need to keep running. Warn, don't raise.
    #
    # I'm also masking out the diagonal before the check. Monkeys don't groom
    # themselves, so B_ii == 0 by construction -- if I let that hit the Feller test
    # it would fire on literally every single call and nobody would notice the
    # legitimate warnings. Only off-diagonal edges are real CIR processes.
    off_diag = ~np.eye(N, dtype=bool)
    if np.any(2.0 * B[off_diag] <= config.sigma ** 2):
        warnings.warn(
            "Feller condition violated: 2*B_ij <= sigma^2 for at least one off-diagonal "
            "edge; non-negativity of A is only enforced by the max(0, ...) floor.",
            RuntimeWarning,
            stacklevel=2,
        )

    # --- 2. The Deterministic Drift ---

    # we're calculating the non-random part of the update here.
    # Think of it as a tug-of-war: B is the actual grooming happening (pulling relationship up),
    # and lambda * A is the monkeys naturally forgetting each other (pulling it down).
    drift = (B - config.lambda_decay * A) * config.dt_sde

    # --- 3. The Stochastic Diffusion (Wiener increment) ---

    # One independent N(0,1) draw per edge -- this is the "random kick" each pair
    # of monkeys gets this step. We pull from the caller's rng on purpose: any
    # Monte Carlo sweep that seeds the generator the same way MUST get the same
    # numbers back. Do NOT reach for np.random.* globals here, that would silently
    # break reproducibility and I'd never find out until the phase diagram looked wrong.
    Z = rng.standard_normal((N, N))

    # And now the actual discrete Wiener increment dW = sqrt(dt) * Z, scaled by the
    # CIR diffusion coefficient sigma * sqrt(A). The max(0, A) inside the sqrt is pure
    # paranoia -- floating point in the drift step can nudge A a tiny bit below zero
    # *before* the floor clamp runs, and sqrt of a negative blows up the whole matrix.
    # Cheaper to clamp inside the sqrt than to pay for NaN debugging later.
    diffusion = config.sigma * np.sqrt(np.maximum(0.0, A)) * np.sqrt(config.dt_sde) * Z

    # Standard forward Euler-Maruyama jump: drift + diffusion, same as the stub but
    # now with the noise term actually doing something.
    A_next = A + drift + diffusion

    # --- 4. The "No Anti-Alliances" Floor ---

    # Mathematically the real CIR process uses sqrt(A) to dampen noise when
    # relationships hit zero, naturally preventing negative numbers.
    # But we're discretizing continuous time into chunky steps, so rapid decay can and
    # WILL randomly undershoot past zero -- especially once the Sprint 2 noise is
    # turned on and sigma*sqrt(A)*sqrt(dt)*Z can swing both directions.
    # Dev B's RK4 solver goes completely nuts if we feed it negative relationships.
    # Rather than doing something computationally expensive, I'm just hard-clamping the
    # floor to 0.0. "Anti-alliances" don't exist in our model anyway.
    A_next = np.maximum(0.0, A_next)

    return A_next

