"""Common agent interface. Every agent controls BOTH survivors and returns a joint action.

Decentralized agents (A*, independent Q-learners) are simply two per-survivor
policies wrapped in one Agent. Planners receive `env` so they can snapshot it
(env.get_state()) -- non-planning agents must not touch it (fairness rule).
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Agent(ABC):
    name: str = "agent"
    #: what this agent is allowed to see -- documented for the fairness rule
    observes: str = "full state"

    def reset(self, seed: int | None = None) -> None:
        """Called at the start of each episode."""

    @abstractmethod
    def act(self, obs: dict, env) -> tuple[int, int]:
        """Return (a1, a2), each in 0..5."""

    def observe(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        """Optional hook for learning agents."""
