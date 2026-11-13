# hysteresis.py — Overthrow detection with epsilon threshold
# Assignee: Dev D (S2-T5)
# See: project_management/tasks/sprint_2/S2-T5_hysteresis_engine.md
# Convention: R[i] = rank of agent i (0 = highest / alpha)
# ChatGPT was used to debug and refactor this code. Some comments were also refined by ChatGPT.

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from macaque_sim.config.sim_config import SimConfig

def check_overthrows(
    t,
    P: np.ndarray,
    R: np.ndarray,
    config: SimConfig,
    cooldown_tracker: Any,
    epsilon_override: float | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Check whether any discrete rank overthrows should occur at the current timestep.

    This function is the bridge between:
    - continuous power scores P(t), and
    - discrete ordinal ranks R[t].

    Core idea
    ---------
    A lower-status agent i (subordinate) can overthrow a higher-status agent j
    (dominant) only if:

        1. R[i] > R[j]
           meaning i is currently below j in the rank hierarchy

        2. P[i] >= P[j] + epsilon
           meaning i's power exceeds j's power by at least the hysteresis margin

        3. neither i nor j is on cooldown
           meaning neither agent is temporarily locked out from participating
           in another rank swap

    Multi-swap handling
    -------------------
    It is possible that multiple valid overthrow candidates exist in the same
    timestep. In that case, this function uses greedy priority:

        - choose the valid pair with the largest power differential first
        - perform that swap
        - recompute the remaining valid pairs under the updated ranks
        - repeat until no valid swaps remain

    Parameters
    ----------
    P : np.ndarray, shape (N,)
        Current power scores.

    R : np.ndarray, shape (N,)
        Current discrete rank array.
        Convention assumed here:
            lower rank number = higher status
        Example (matches the rest of the codebase; see mri_init.assign_initial_ranks):
            R[k] = 1 means agent k is alpha
            R[k] = 2 means beta

    config : SimConfig
        Configuration object that must provide the hysteresis margin (epsilon).

    cooldown_tracker : Any
        Object describing which agents are currently on cooldown.

        Read:
            - cooldown_tracker.is_on_cooldown(t, agent_id)

        Write/update:
            - cooldown_tracker.start_cooldown(t, winner_id, loser_id)

    Returns
    -------
    R_new : np.ndarray, shape (N,)
        Updated rank array after processing all valid swaps for this timestep.

    events : list[dict]
        List of swap detail dictionaries.

        This matches logger.py, where the caller can do:
            logger.record_event(t, "swap", event)

        Each event dict contains:
            - winner_id  : ID of the subordinate who moved up
            - loser_id   : ID of the dominant who moved down
            - old_rank_w : winner's rank before the swap
            - new_rank_w : winner's rank after the swap
    """

    # Convert power scores to a float numpy array.
    # This guarantees consistent numeric behavior even if the caller passed
    # a Python list or some other array-like structure.
    # Soft Guard
    P = np.asarray(P, dtype=float)

    # Copy the rank array so we do not mutate the caller's input in place.
    # Also ensure that it is a float numpy array
    R_new = np.asarray(R).copy()

    # Sanity checks, both inputs must be 1D and aligned in length.
    # Helps perevent messy debugging later
    if P.ndim != 1 or R_new.ndim != 1:
        raise ValueError("P and R must both be 1D arrays.")
    if P.shape[0] != R_new.shape[0]:
        raise ValueError("P and R must have the same length.")

    # Extract the hysteresis margin epsilon. If the caller supplies an override
    # (used by main.py when config.epsilon_mode == "adaptive" and the rolling
    # EpsilonTracker has computed a current value), trust that; otherwise fall
    # back to the static config.epsilon. Production behavior is unchanged
    # because main.py only passes an override when running in adaptive mode.
    epsilon = float(epsilon_override) if epsilon_override is not None else config.epsilon

    # This list stores all swap events that occur during this call.
    # Each entry is a detail dict that matches logger.record_event(..., "swap", details).
    events: List[Dict[str, Any]] = []

    # Greedy multi-swap loop:
    # Repeatedly find the single best valid overthrow, apply it, and then
    # re-evaluate all pairs from scratch under the updated ranks.
    while True:
        best_pair = None
        best_delta = -np.inf

        n = len(P)

        # Iterate over every ordered pair (i, j) using brute force (no optimization).
        # We interpret i as a potential subordinate challenger and j as a
        # potential dominant candidate.
        for i in range(n):
            for j in range(n):
                # Skip comparisons to itself.
                if i == j:
                    continue

                # Only consider pairs where i is currently lower in status than j.
                # smaller rank number = better rank
                # so R[i] > R[j] means i is subordinate and j is dominant.
                if R_new[i] <= R_new[j]:
                    continue

                # If either participant is currently on cooldown, the pair is
                # ineligible for swapping this timestep.
                if cooldown_tracker.is_on_cooldown(t, i):
                    continue
                if cooldown_tracker.is_on_cooldown(t, j):
                    continue

                # Compute how far the subordinate exceeds the dominant in power.
                delta = P[i] - P[j]

                # Accept the pair only if it meets the hysteresis threshold.
                #
                # We use >= epsilon because the acceptance criteria explicitly says
                # the swap should fire when P[sub] - P[dom] >= epsilon.
                #
                # Among all valid pairs, retain the one with the largest delta.
                if delta >= epsilon and delta > best_delta:
                    best_delta = delta
                    best_pair = (i, j)

        # If no valid pair was found, the system is stable for this timestep.
        if best_pair is None:
            break

        # Unpack the winning subordinate and losing dominant.
        winner_id, loser_id = best_pair

        # Record the winner's old rank and future new rank before swapping.
        # These are the fields requested in the event output spec.
        old_rank_w = int(R_new[winner_id])
        new_rank_w = int(R_new[loser_id])

        # Perform the discrete rank swap.
        R_new[winner_id], R_new[loser_id] = R_new[loser_id], R_new[winner_id]

        # Build the event detail dict.
        event = {
            "winner_id": int(winner_id),
            "loser_id": int(loser_id),
            "old_rank_w": old_rank_w,
            "new_rank_w": new_rank_w,
        }
        events.append(event)

        # After a swap, both agents should be placed on cooldown.
        cooldown_tracker.start_cooldown(t, winner_id, loser_id)

        # The while-loop then restarts and recomputes all possible pairs under the
        # updated hierarchy. This is what enforces the greedy multi-swap behavior.

    return R_new, events