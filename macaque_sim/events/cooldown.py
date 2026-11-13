# cooldown.py — Post-conflict lockout tracker
# Assignee: Dev D (S2-T6)
# See: project_management/tasks/sprint_2/S2-T6_cooldown.md
# ChatGPT was used to debug and refactor this code. Some comments were also refined by ChatGPT.
from __future__ import annotations

from macaque_sim.config.sim_config import SimConfig

class CooldownTracker:
    """
    State tracker for post-conflict cooldowns.

    Cooldown state stored as:
        {agent_id: unlock_time}
    where unlock_time is in the same "time units" as t passed in (ODE time).
    """

    def __init__(self, config: SimConfig):
        # Duration is in the same ODE time units as t passed to start_cooldown /
        # is_on_cooldown. NOT a macro-step count -- unlock = t + cooldown_duration.
        self.cooldown_duration = config.cooldown_duration
        self.unlock_times = {}  # {agent_id: unlock_time}

    def start_cooldown(self, t, agent_a: int, agent_b: int) -> None:
        """
        Start cooldown for both agents at time t.
        """

        # Cleanup expired entries
        expired_agents = [
            agent_id
            for agent_id, unlock_time in self.unlock_times.items()
            if t >= unlock_time
        ]
        for agent_id in expired_agents:
            del self.unlock_times[agent_id]

        # Start cooldown for current agents
        unlock_time = t + self.cooldown_duration
        self.unlock_times[agent_a] = unlock_time
        self.unlock_times[agent_b] = unlock_time

    def is_on_cooldown(self, t, agent_id: int) -> bool:
        """
        Check if agent is on cooldown at time t.
        """
        if agent_id not in self.unlock_times:
            return False

        unlock_time = self.unlock_times[agent_id]
        return t < unlock_time