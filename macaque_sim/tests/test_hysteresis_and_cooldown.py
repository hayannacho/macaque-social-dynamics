#ChatGPT was used to suggest tests, refactor test code and refine comments 
import numpy as np

from macaque_sim.config.sim_config import default_config
from macaque_sim.events.cooldown import CooldownTracker
from macaque_sim.events.hysteresis import check_overthrows


def test_no_swap_when_scores_are_close():
    """
    If the subordinate's power does not exceed the dominant's power
    by at least epsilon, no swap should occur.
    """

    # Build config and set epsilon explicitly
    cfg = default_config()
    cfg.epsilon = 0.02

    # Create cooldown tracker
    cooldown_tracker = CooldownTracker(cfg)

    # Rank convention:
    # lower number = higher status
    # So here:
    # agent 0 has rank 1 (subordinate)
    # agent 1 has rank 0 (dominant)
    R = np.array([1, 0])

    # Subordinate exceeds dominant by only 0.01 < epsilon
    P = np.array([0.51, 0.50])

    R_new, events = check_overthrows(
        t=10,
        P=P,
        R=R,
        config=cfg,
        cooldown_tracker=cooldown_tracker,
    )

    # No swap should occur
    assert np.array_equal(R_new, R)

    # No event should be recorded
    assert events == []


def test_swap_fires_at_threshold_and_event_fields_are_present():
    """
    If the subordinate exceeds the dominant by exactly epsilon,
    the swap should occur.

    Also verify that the returned event dict contains all required fields.
    """

    cfg = default_config()
    cfg.epsilon = 0.02

    cooldown_tracker = CooldownTracker(cfg)

    # agent 0 is subordinate, agent 1 is dominant
    R = np.array([1, 0])

    # delta = 0.52 - 0.50 = 0.02 = epsilon
    # So the swap should fire
    P = np.array([0.52, 0.50])

    R_new, events = check_overthrows(
        t=10,
        P=P,
        R=R,
        config=cfg,
        cooldown_tracker=cooldown_tracker,
    )

    # Ranks should be swapped
    expected_R = np.array([0, 1])
    assert np.array_equal(R_new, expected_R)

    # Exactly one swap event should be recorded
    assert len(events) == 1

    event = events[0]

    # Check required event fields
    assert "winner_id" in event
    assert "loser_id" in event
    assert "old_rank_w" in event
    assert "new_rank_w" in event

    # Check field values for this specific case
    assert event["winner_id"] == 0
    assert event["loser_id"] == 1
    assert event["old_rank_w"] == 1
    assert event["new_rank_w"] == 0


def test_cooldown_blocks_reswap():
    """
    After a swap, both involved agents should be placed on cooldown.

    Even if the old loser becomes strong enough to overthrow back immediately,
    the re-swap must be blocked while cooldown is active.
    """

    cfg = default_config()
    cfg.epsilon = 0.02
    cfg.cooldown_duration = 5

    cooldown_tracker = CooldownTracker(cfg)

    # Initial state:
    # agent 0 subordinate, agent 1 dominant
    R = np.array([1, 0])

    # First timestep: agent 0 exceeds agent 1 by epsilon, so swap should happen
    P_first = np.array([0.52, 0.50])

    R_after_first, events_first = check_overthrows(
        t=10,
        P=P_first,
        R=R,
        config=cfg,
        cooldown_tracker=cooldown_tracker,
    )

    # Confirm first swap happened
    assert np.array_equal(R_after_first, np.array([0, 1]))
    assert len(events_first) == 1

    # Next timestep:
    # Now agent 1 is subordinate and has enough power to overthrow back
    # But cooldown should block the re-swap
    P_second = np.array([0.50, 0.53])

    R_after_second, events_second = check_overthrows(
        t=11,
        P=P_second,
        R=R_after_first,
        config=cfg,
        cooldown_tracker=cooldown_tracker,
    )

    # No re-swap should occur because both agents are still on cooldown
    assert np.array_equal(R_after_second, R_after_first)
    assert events_second == []