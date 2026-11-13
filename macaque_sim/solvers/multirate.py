# multirate.py -- Multi-rate coupling controller (SDE micro <-> ODE macro)
# Assignee: Dev A (S2-T2)
# See: project_management/tasks/sprint_2/S2-T2_multirate_controller.md

"""
Multi-rate integration controller -- the piece of the pipeline that bridges my SDE
and Dev B's ODE so they can share a simulation loop without fighting each other.

The motivating problem: Euler-Maruyama has strong convergence order 0.5, so it
needs a tiny dt_sde to not blow up. RK4, on the other hand, converges at O(dt^5)
and is happy with a much coarser dt_ode. If I just ran both at dt_sde everything
works but we waste a ton of CPU on RK4 steps nobody asked for. If I ran both at
dt_ode the SDE explodes. The answer is to run each at the step size it likes, and
fold the results together once per macro-step.

Here's what one macro-step actually does (this is the whole algorithm):

    1. Run M = config.micro_steps_per_macro micro-steps of em_step(A, B, cfg, rng) on
       the alliance matrix A, advancing it the full macro-interval in little pieces.
    2. Track the element-wise MAXIMUM of A across those M intermediate states (the
       "envelope"). This is the subtle bit -- spec says maximum, not average. The
       reason is biological: the rank overthrows we care about are triggered by
       spikes in alliance strength, not averages. Averaging would smear a spike
       across M steps and the power iteration would never see it.
    3. Feed that envelope into ONE rk4_step call on the power vector P, advancing
       P the full dt_ode macro-interval.
    4. Return (latest A from the micro loop, updated P from RK4).

gotcha: Dev B's rk4_step already renormalizes P internally before it returns, so
this controller intentionally does NOT renormalize again. If I did, I'd be dividing
a sum that's already 1.0 -- harmless numerically but misleading if anyone reads the
code later and assumes renormalization happens here.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from macaque_sim.config.sim_config import SimConfig
from macaque_sim.solvers.euler_maruyama import em_step
from macaque_sim.solvers.rk4 import rk4_step

# M (SDE micro-steps per ODE macro-step) now lives on SimConfig as
# `micro_steps_per_macro` -- added by Dev C for this ticket. Read it directly.


def multirate_step(
    A: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    config: SimConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advance the coupled (A, P) state by one macro-step of length `config.dt_ode`.

    This is the function main.py's sim loop calls once per macro-step. It wraps up
    M em_step micro-calls and a single rk4_step macro-call behind one interface so
    the sim loop doesn't have to know about the SDE/ODE rate mismatch.

    Parameters
    ----------
    A : (N, N) ndarray
        Current alliance matrix.
    P : (N,) ndarray
        Current power vector; assumed to already sum to 1.
    B : (N, N) ndarray
        Grooming / baseline matrix for the alliance SDE.
    config : SimConfig
        Provides dt_sde, dt_ode, N, lambda_decay, sigma.
    rng : numpy Generator
        Shared RNG so that Monte Carlo seeding stays reproducible end-to-end.

    Returns
    -------
    A_next : (N, N) ndarray
        Alliance matrix after the final micro-step. This is the "current" A to
        hand back to the sim loop -- NOT the envelope matrix (which is a
        construction detail we use to drive RK4, not physical state).
    P_next : (N,) ndarray
        Power vector after one RK4 macro-step driven by the envelope.
    """
    N = config.N

    # --- 1. Sanity Checks (same style as em_step) ---
    # gotcha: if any of A, P, B disagree on N with the config, I want the ValueError
    # to fire HERE, before I spend M micro-steps scribbling on mismatched arrays and
    # then watch RK4 explode downstream with a much less helpful error.
    if A.shape != (N, N):
        raise ValueError(f"Alliance matrix A has shape {A.shape}, expected {(N, N)}")
    if P.shape != (N,):
        raise ValueError(f"Power vector P has shape {P.shape}, expected {(N,)}")
    if B.shape != (N, N):
        raise ValueError(f"Grooming matrix B has shape {B.shape}, expected {(N, N)}")

    # M lives on SimConfig (added by Dev C); just read it straight out.
    M = config.micro_steps_per_macro

    # --- 2. Run M micro-steps, tracking the envelope ---
    #
    # The envelope folds the M post-step matrices into a single matrix passed to
    # RK4. Default mode "max" (production) takes the element-wise max -- the
    # original biological intuition is that overthrows are triggered by
    # transient power spikes that an average would smooth away.
    #
    # Mode "mean" was added post-review (final_report.tex §5.4 mean-vs-max
    # coupling experiment) to probe how much of the sigma transition is the
    # max-envelope's positive sigma-dependent bias vs. genuine stochastic
    # destabilization. The mean envelope is the unbiased estimator of the
    # macro-interval-averaged A; max is biased upward.
    #
    # I deliberately do NOT seed the running aggregate with the starting A --
    # that's the previous macro-interval's state, not this one's. Seeding with
    # the first post-step matrix has a nice side-effect: when M = 1, the
    # envelope is exactly the single post-step A under both modes, which
    # means multirate_step collapses to "em_step then rk4_step" regardless of
    # coupling_mode (the S2-T2 acceptance criterion).
    coupling_mode = getattr(config, "coupling_mode", "max")
    A_envelope: np.ndarray | None = None
    A_current = A

    for _ in range(M):
        # Advance A by one SDE micro-step. em_step already re-draws noise from rng,
        # so every iteration sees independent Z.
        A_current = em_step(A_current, B, config, rng)

        # Fold the new state into the running aggregate. First iteration seeds
        # it; subsequent iterations either take the element-wise max in-place
        # ("max" mode) or accumulate the running sum ("mean" mode), to avoid
        # allocating a fresh (N, N) matrix each iteration.
        if A_envelope is None:
            A_envelope = A_current.copy()
        elif coupling_mode == "mean":
            A_envelope += A_current
        else:  # "max"
            np.maximum(A_envelope, A_current, out=A_envelope)

    # In mean mode A_envelope is the running SUM at this point; divide by M to
    # finish the aggregation. Done outside the inner loop because dividing in
    # every iteration would be wrong (the i-th iteration would see a partial
    # sum / M, not the running mean over i seen samples).
    if coupling_mode == "mean":
        assert A_envelope is not None  # M >= 1 by validation
        A_envelope /= M

    # --- 3. One RK4 macro-step on P, driven by the envelope ---
    #
    # This is the "SDE hands the envelope to the ODE" handoff described in the
    # docstring. Dev B's rk4_step signature is (A, P, h) and it renormalizes
    # internally, so we pass the envelope directly and don't touch P_next after
    # the call. If Dev B ever changes that signature or removes the internal
    # renorm, this line has to change too -- cross-dev coupling alert.
    P_next = rk4_step(A_envelope, P, config.dt_ode)

    # We return A_current (the final micro-step state), NOT A_envelope. The
    # envelope is an RK4-driving construct only; the thing that lives forward
    # in the sim loop is the actual post-noise alliance matrix.
    return A_current, P_next
