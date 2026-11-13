# perturbations.py — Mid-simulation shocks (alpha removal, grooming shock, σ spike)
# Assignee: Dev E (S3-T2)
# See: project_management/tasks/sprint_3/S3-T2_perturbation_injector.md

"""
perturbations.py — Mid-simulation perturbation injector.

Primary S3-T2 API:
    apply_perturbation(perturb_dict, A, P, R, config) -> (A, P, R, config)

Supported perturbation types:
  - "alpha_removal"     : zero row/col in A, zero P[target], renormalize P
  - "grooming_shock"    : replace grooming matrix B
  - "volatility_spike"  : multiply sigma by factor (optionally temporary)
  - "orphan_injection"  : zero target row/col in A, drop target to bottom rank,
                          zero P[target], renormalize P. Biologically: a
                          macaque without matriline support slides to omega.

AI Usage: After completing all logic for the perturbation types, I used AI to generate the docstrings and comments for the code.
Also asked for refactoring the code to make it more readable and maintainable.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# State type alias (for documentation clarity)
# ---------------------------------------------------------------------------
# state = {
#   "A": np.ndarray shape (N, N)  — alliance matrix
#   "P": np.ndarray shape (N,)    — power scores
#   "R": list[int] shape (N,)     — rank array (R[i] = rank of agent i)
#   "t": float                    — current simulation time
# }


# ---------------------------------------------------------------------------
# Individual perturbation functions
# ---------------------------------------------------------------------------

def _renormalize_power(P: np.ndarray) -> np.ndarray:
    total = float(P.sum())
    if total > 0:
        P[:] = P / total
    else:
        P[:] = 1.0 / len(P)
    return P


def apply_alpha_removal(state: dict, target: int, params: dict) -> dict:
    """
    Remove an agent from the alliance network by zeroing their row and column
    in A(t).  Their power score is also set to zero.

    This simulates death, transfer, or removal in a captive breeding context.
    The hysteresis engine (Dev D) will detect the resulting power vacuum and
    trigger rank reassignment on the next simulation step.

    Parameters
    ----------
    state  : simulation state dict (mutated in place).
    target : agent index to remove.  Pass -1 to remove the current rank-0
             agent (resolved dynamically from state["R"]).
    params : dict — supports:
        "zero_row" : bool (default True)  — zero out row target in A.
        "zero_col" : bool (default True)  — zero out col target in A.

    Returns
    -------
    Modified state dict.
    """
    A = state["A"]
    P = state["P"]
    R = state["R"]
    N = A.shape[0]

    # Resolve sentinel: -1 means "current alpha" (rank 1)
    if target == -1:
        target = int(np.argmin(R))   # agent holding the minimum rank number (alpha)

    if not (0 <= target < N):
        raise ValueError(f"target agent {target} out of range [0, {N}).")

    zero_row = params.get("zero_row", True)
    zero_col = params.get("zero_col", True)

    if zero_row:
        A[target, :] = 0.0
    if zero_col:
        A[:, target] = 0.0

    P[target] = 0.0
    _renormalize_power(P)

    return state


def apply_grooming_shock(
    state: dict,
    target: int,
    params: dict,
) -> dict:
    """
    Replace the grooming input matrix B_ij with an alternative matrix,
    simulating a sudden shift in grooming strategy.

    The new B matrix is provided via params["new_B"].  This is stored in
    state["B"] so the SDE integrator picks it up on the next micro-step.

    Parameters
    ----------
    state  : simulation state dict (mutated in place).
    target : unused for this perturbation type (pass 0 or -1).
    params : dict — requires:
        "new_B" : np.ndarray shape (N, N) — replacement grooming matrix.

    Returns
    -------
    Modified state dict.
    """
    if "new_B" not in params:
        raise ValueError("apply_grooming_shock requires params['new_B'].")

    new_B = np.asarray(params["new_B"], dtype=float)
    expected_shape = state["A"].shape

    if new_B.shape != expected_shape:
        raise ValueError(
            f"new_B shape {new_B.shape} does not match A shape {expected_shape}."
        )

    state["B"] = new_B
    return state


def apply_orphan_injection(state: dict, target: int, params: dict) -> dict:
    """
    Inject an orphan: a monkey that has lost its matriline anchor.

    Biologically, macaque rank is matrilineally inherited and contingent on
    the mother being alive and socially active to enforce it. An orphan (or
    transferred/unaffiliated immigrant) has no maternal coalition backup and
    slides to the bottom of the hierarchy as a default — field data (Bernstein
    & Ehardt, 1985 and others) supports this strongly.

    Effects applied to state in place:
      - A[target, :] and A[:, target] zeroed (no inherited alliances).
      - target agent moved to the bottom rank (swapped with whoever currently
        holds max(R)). Override via params["starting_rank"].
      - P[target] zeroed and P renormalized to sum to 1.

    Parameters
    ----------
    state  : simulation state dict (mutated in place). Requires "A", "P", "R".
    target : agent index to orphan.
    params : dict — supports:
        "starting_rank" : "bottom" (default), or explicit int rank value.
                          "bottom" places orphan at max(R) and swaps with the
                          agent currently there.

    Returns
    -------
    Modified state dict.
    """
    A = state["A"]
    P = state["P"]
    R = state["R"]
    N = A.shape[0]

    if not (0 <= target < N):
        raise ValueError(f"target agent {target} out of range [0, {N}).")

    A[target, :] = 0.0
    A[:, target] = 0.0

    R_arr = np.asarray(R)
    starting_rank = params.get("starting_rank", "bottom")
    if starting_rank == "bottom":
        target_rank_value = int(R_arr.max())
    elif isinstance(starting_rank, (int, np.integer)):
        target_rank_value = int(starting_rank)
    else:
        raise ValueError(
            f"starting_rank must be 'bottom' or int; got {starting_rank!r}."
        )

    orphan_current_rank = int(R_arr[target])
    if orphan_current_rank != target_rank_value:
        holders = np.where(R_arr == target_rank_value)[0]
        if len(holders) == 0:
            raise ValueError(
                f"No agent currently holds rank {target_rank_value}; cannot swap."
            )
        holder = int(holders[0])
        if isinstance(R, list):
            R[target], R[holder] = target_rank_value, orphan_current_rank
        else:
            R[target] = target_rank_value
            R[holder] = orphan_current_rank

    P[target] = 0.0
    _renormalize_power(P)

    return state


def apply_volatility_spike(state: dict, target: int, params: dict) -> dict:
    """
    Multiply config/state sigma by a scalar factor.

    params:
      - factor: required positive multiplier
      - duration: optional metadata for loop-level scheduling (stored only)
    """
    if "factor" not in params:
        raise ValueError("apply_volatility_spike requires params['factor'].")
    factor = float(params["factor"])
    if factor <= 0:
        raise ValueError(f"factor must be > 0; got {factor}.")

    # Support both state["config"] style and lightweight state dict.
    cfg = state.get("config")
    if cfg is not None and hasattr(cfg, "sigma"):
        cfg.sigma = float(cfg.sigma) * factor
        if "duration" in params:
            setattr(cfg, "_sigma_spike_duration", float(params["duration"]))
        return state

    if "sigma" in state:
        state["sigma"] = float(state["sigma"]) * factor
        if "duration" in params:
            state["_sigma_spike_duration"] = float(params["duration"])
        return state

    raise ValueError("volatility_spike needs state['config'].sigma or state['sigma'].")


# ---------------------------------------------------------------------------
# Dispatch table and main entry point
# ---------------------------------------------------------------------------

_PERTURBATION_REGISTRY: dict[str, callable] = {
    "alpha_removal":  apply_alpha_removal,
    "grooming_shock": apply_grooming_shock,
    "volatility_spike": apply_volatility_spike,
    "orphan_injection": apply_orphan_injection,
}


def apply_perturbation(
    perturbation: dict,
    A: np.ndarray,
    P: np.ndarray,
    R,
    config,
) -> tuple[np.ndarray, np.ndarray, any, any]:
    """
    Dispatch and apply a single perturbation to the simulation state.

    This is the function called by the main simulation loop at each macro-step
    when checking whether any scheduled perturbation should fire.

    Parameters
    ----------
    state        : current simulation state dict {"A", "P", "R", "t", ...}.
    perturbation : one entry from ExperimentScenario.perturbations, with keys:
                     "type"   : str   — perturbation kind
                     "time"   : float — trigger time (-1 means T/2)
                     "target" : int   — agent index (-1 for dynamic resolution)
                     "params" : dict  — perturbation-specific keyword args
    sim_T        : total simulation duration, used to resolve time sentinel -1.

    Returns
    -------
    Modified state dict.

    Raises
    ------
    ValueError  : if perturbation type is unknown.
    """
    p_type = perturbation["type"]
    target = perturbation.get("target", -1)
    params = dict(perturbation.get("params", {}))

    state = {"A": A, "P": P, "R": R, "config": config}

    if p_type not in _PERTURBATION_REGISTRY:
        raise ValueError(
            f"Unknown perturbation type '{p_type}'. "
            f"Available: {sorted(_PERTURBATION_REGISTRY.keys())}."
        )
    fn = _PERTURBATION_REGISTRY[p_type]
    fn(state, target, params)

    # Grooming shock can replace B on config if present
    if "B" in state:
        try:
            setattr(config, "B", state["B"])
        except Exception:
            pass
    return state["A"], state["P"], state["R"], config


def should_trigger(
    perturbation: dict,
    t_prev: float,
    t_now: float,
    sim_T: float,
) -> bool:
    """
    Return True if a perturbation should fire in the interval (t_prev, t_now].

    Uses a half-open interval so each perturbation fires exactly once even
    if the simulation step lands exactly on the trigger time.

    Parameters
    ----------
    perturbation : perturbation dict from ExperimentScenario.perturbations.
    t_prev       : simulation time at start of current macro-step.
    t_now        : simulation time at end of current macro-step.
    sim_T        : total simulation duration (for resolving sentinel -1).
    """
    p_time = perturbation["time"]
    trigger = sim_T / 2.0 if p_time == -1 else float(p_time)
    return t_prev < trigger <= t_now


def register_perturbation(name: str, fn: callable) -> None:
    """
    Register a custom perturbation handler at runtime.

    Allows future experiments to add new perturbation types without modifying
    this file.  The handler must have signature: fn(state, target, params) -> state.

    Parameters
    ----------
    name : str      — perturbation type string used in scenario dicts.
    fn   : callable — handler function.
    """
    if not callable(fn):
        raise TypeError(f"fn must be callable; got {type(fn)}.")
    _PERTURBATION_REGISTRY[name] = fn


def list_perturbation_types() -> list[str]:
    """Return all currently registered perturbation type strings."""
    return sorted(_PERTURBATION_REGISTRY.keys())