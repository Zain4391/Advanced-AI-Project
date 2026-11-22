from __future__ import annotations

import numpy as np

from env.config import N_ACTIONS

from .base import Agent


class RandomAgent(Agent):
    name = "random"
    observes = "nothing"

    def __init__(self, seed: int = 0):
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng((self._seed, seed))

    def act(self, obs, env) -> tuple[int, int]:
        a = self.rng.integers(N_ACTIONS, size=2)
        return int(a[0]), int(a[1])
