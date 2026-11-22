"""Headless smoke test: the renderer draws a full random episode without errors."""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from env import OutbreakEnv  # noqa: E402
from env.render_pygame import Renderer  # noqa: E402


def test_renderer_runs_full_episode():
    pygame.init()
    env = OutbreakEnv()
    env.reset(seed=1)
    r = Renderer(env.config, cell=32)
    screen = pygame.Surface(r.size)
    rng = np.random.default_rng(0)
    now, done = 0.0, False
    while not done:
        prev = env.state.copy()
        _, _, term, trunc, info = env.step(tuple(rng.integers(6, size=2)))
        r.on_step(prev, env.state, info, now=now)
        for dt in (0.0, 0.08, 0.5):
            r.draw(screen, env.state, pending=(None, 1), seed=1, now=now + dt)
        now += 1.0
        done = term or trunc
    r.draw(screen, env.state, banner=("END", "sub", (255, 0, 0)), now=now + 5)
