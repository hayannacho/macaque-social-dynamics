#ChatGPT was used to suggest tests, refactor test code and refine comments 
from macaque_sim.config.sim_config import default_config
from macaque_sim.events.cooldown import CooldownTracker


def test_agent_on_cooldown_returns_true():
    """
    After starting cooldown, the involved agents should be reported
    as being on cooldown for all t < unlock_time.
    """

    # Create config and override cooldown length for test clarity
    cfg = default_config()
    cfg.cooldown_duration = 5

    tracker = CooldownTracker(cfg)

    # Start cooldown at t = 10 for agents 2 and 7
    # unlock_time = 10 + 5 = 15
    tracker.start_cooldown(t=10, agent_a=2, agent_b=7)

    # Immediately after swap → both must be on cooldown
    assert tracker.is_on_cooldown(t=10, agent_id=2) is True
    assert tracker.is_on_cooldown(t=10, agent_id=7) is True

    # Still before unlock_time → still on cooldown
    assert tracker.is_on_cooldown(t=12, agent_id=2) is True
    assert tracker.is_on_cooldown(t=12, agent_id=7) is True


def test_cooldown_expires_after_cooldown_duration():
    """
    After cooldown_duration has elapsed, the agent should no longer
    be considered on cooldown.
    """

    cfg = default_config()
    cfg.cooldown_duration = 5

    tracker = CooldownTracker(cfg)

    # Start cooldown at t = 10 → unlock_time = 15
    tracker.start_cooldown(t=10, agent_a=2, agent_b=7)

    # Before unlock_time → still blocked
    assert tracker.is_on_cooldown(t=14, agent_id=2) is True
    assert tracker.is_on_cooldown(t=14, agent_id=7) is True

    # At unlock_time → cooldown expires
    # Condition is: t < unlock_time → False when t == unlock_time
    assert tracker.is_on_cooldown(t=15, agent_id=2) is False
    assert tracker.is_on_cooldown(t=15, agent_id=7) is False


def test_multiple_agents_can_be_on_cooldown_simultaneously():
    """
    Ensure that multiple independent cooldowns can coexist and
    expire at different times without interfering with each other.
    """

    cfg = default_config()
    cfg.cooldown_duration = 5

    tracker = CooldownTracker(cfg)

    # First swap at t = 4 → unlock_time = 9
    tracker.start_cooldown(t=4, agent_a=1, agent_b=3)

    # Second swap at t = 6 → unlock_time = 11
    tracker.start_cooldown(t=6, agent_a=5, agent_b=6)

    # At t = 7:
    # Both cooldown windows are active → all four agents blocked
    assert tracker.is_on_cooldown(t=7, agent_id=1) is True
    assert tracker.is_on_cooldown(t=7, agent_id=3) is True
    assert tracker.is_on_cooldown(t=7, agent_id=5) is True
    assert tracker.is_on_cooldown(t=7, agent_id=6) is True

    # An unrelated agent should not be affected
    assert tracker.is_on_cooldown(t=7, agent_id=8) is False

    # At t = 10:
    # First cooldown expired (unlock_time = 9)
    # Second cooldown still active (unlock_time = 11)
    assert tracker.is_on_cooldown(t=10, agent_id=1) is False
    assert tracker.is_on_cooldown(t=10, agent_id=3) is False
    assert tracker.is_on_cooldown(t=10, agent_id=5) is True
    assert tracker.is_on_cooldown(t=10, agent_id=6) is True