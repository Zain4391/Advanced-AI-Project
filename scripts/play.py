"""Play the game with the keyboard (two humans, simultaneous turns).

    python scripts/play.py --seed 0

Survivor 1 (blue):   arrow keys move, SPACE shoot, ENTER wait
Survivor 2 (orange): W A S D move,    F shoot,     Q wait
The game advances one step once both survivors have chosen (evacuated survivors auto-wait).
R restart same seed | N next seed | ESC quit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame  # noqa: E402

from env import DOWN, LEFT, RIGHT, SHOOT, UP, WAIT, EnvConfig, OutbreakEnv  # noqa: E402
from env.render_pygame import C, Renderer  # noqa: E402

KEYS_S1 = {pygame.K_UP: UP, pygame.K_DOWN: DOWN, pygame.K_LEFT: LEFT, pygame.K_RIGHT: RIGHT,
           pygame.K_SPACE: SHOOT, pygame.K_RETURN: WAIT, pygame.K_KP_ENTER: WAIT}
KEYS_S2 = {pygame.K_w: UP, pygame.K_s: DOWN, pygame.K_a: LEFT, pygame.K_d: RIGHT,
           pygame.K_f: SHOOT, pygame.K_q: WAIT}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=8, help="grid size")
    ap.add_argument("--cell", type=int, default=80, help="tile size in pixels (multiples of 16 look crispest)")
    args = ap.parse_args()

    env = OutbreakEnv(EnvConfig(grid_size=args.size))
    seed = args.seed
    env.reset(seed=seed)

    pygame.init()
    renderer = Renderer(env.config, cell=args.cell)
    screen = pygame.display.set_mode(renderer.size)
    pygame.display.set_caption("Zombie Outbreak")
    clock = pygame.time.Clock()

    pending: list[int | None] = [None, None]
    total, banner = 0.0, None
    running = True
    while running:
        s = env.state
        for i in (0, 1):
            if not s.active[i] and pending[i] is None:  # evacuated or dead: auto-wait
                pending[i] = WAIT
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_r, pygame.K_n):
                    seed += event.key == pygame.K_n
                    env.reset(seed=seed)
                    renderer.reset()
                    pending, total, banner = [None, None], 0.0, None
                elif not s.done:
                    if event.key in KEYS_S1:
                        pending[0] = KEYS_S1[event.key]
                    elif event.key in KEYS_S2:
                        pending[1] = KEYS_S2[event.key]

        if not env.state.done and None not in pending:
            prev = env.state.copy()
            _, r, term, trunc, info = env.step(tuple(pending))
            renderer.on_step(prev, env.state, info, action=tuple(pending))
            total += r
            pending = [None, None]
            if info["won"]:
                banner = ("EVACUATED!", f"Both survivors made it out in {env.state.t} steps.  R retry  N next map", C["good"])
            elif info["partial"]:
                banner = ("ONE GOT OUT", "One survivor escaped; the other didn't make it.  R retry  N next map", C["warn"])
            elif info["lost"]:
                banner = ("OVERRUN", "Both survivors went down.  R retry  N next map", C["bad"])
            elif trunc:
                banner = ("OUT OF TIME", "The last chopper left without you.  R retry  N next map", C["warn"])

        renderer.draw(screen, env.state, pending=pending, seed=seed, total_return=total, banner=banner)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
