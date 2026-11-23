"""Watch an agent play by itself, with its planned paths overlaid.

    python scripts/watch.py --agent astar --seed 2000
    python scripts/watch.py --agent astar --auto            # keep cycling through test maps

SPACE pause/resume | RIGHT one step (while paused) | UP/DOWN speed | P toggle paths
R restart map | N next map | ESC quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame  # noqa: E402

from agents import AGENTS, make_agent  # noqa: E402
from env import DEFAULT_CONFIG, NO_COORD_CONFIG, SHIFTED_CONFIGS, OutbreakEnv  # noqa: E402
from env.render_pygame import C, Renderer  # noqa: E402

CONFIGS = {"default": DEFAULT_CONFIG, "no_coord": NO_COORD_CONFIG, **SHIFTED_CONFIGS}
SPEEDS = [0.5, 1, 2, 4, 8, 16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="astar", choices=sorted(AGENTS))
    ap.add_argument("--seed", type=int, default=2000, help="map seed (2000-2099 = test split)")
    ap.add_argument("--config", default="default", choices=sorted(CONFIGS))
    ap.add_argument("--speed", type=float, default=4, help="steps per second")
    ap.add_argument("--auto", action="store_true", help="advance to the next map after each episode")
    ap.add_argument("--cell", type=int, default=80)
    ap.add_argument("--checkpoint", default=None, help="trained agent file, e.g. checkpoints/qlearning/default_seed0.pkl")
    args = ap.parse_args()

    env = OutbreakEnv(CONFIGS[args.config])
    agent = make_agent(args.agent, **({"checkpoint": args.checkpoint} if args.checkpoint else {}))
    seed = args.seed

    pygame.init()
    renderer = Renderer(env.config, cell=args.cell)
    screen = pygame.display.set_mode(renderer.size)
    pygame.display.set_caption(f"Zombie Outbreak - watching {args.agent}")
    clock = pygame.time.Clock()
    speed_idx = min(range(len(SPEEDS)), key=lambda k: abs(SPEEDS[k] - args.speed))

    def start(sd):
        o, _ = env.reset(seed=sd)
        agent.reset(sd)
        renderer.reset()
        return o

    obs = start(seed)
    total, banner, last = 0.0, None, [None, None]
    paused, show_paths, single = False, True, False
    next_step_at, done_at = time.perf_counter(), None
    stats = {"episodes": 0, "wins": 0}

    running = True
    while running:
        now = time.perf_counter()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_RIGHT:
                    single = True
                elif k == pygame.K_UP:
                    speed_idx = min(speed_idx + 1, len(SPEEDS) - 1)
                elif k == pygame.K_DOWN:
                    speed_idx = max(speed_idx - 1, 0)
                elif k == pygame.K_p:
                    show_paths = not show_paths
                elif k in (pygame.K_r, pygame.K_n):
                    seed += k == pygame.K_n
                    obs = start(seed)
                    total, banner, last, done_at = 0.0, None, [None, None], None

        if env.state.done:
            if args.auto and done_at is not None and now - done_at > 2.0:
                seed += 1
                obs = start(seed)
                total, banner, last, done_at = 0.0, None, [None, None], None
        elif (not paused and now >= next_step_at) or (paused and single):
            single = False
            action = agent.act(obs, env)
            prev = env.state.copy()
            obs, r, term, trunc, info = env.step(action)
            renderer.on_step(prev, env.state, info, action=action, now=now)
            agent.observe(None, action, r, obs, term, trunc)
            total += r
            last = list(action)
            next_step_at = now + 1.0 / SPEEDS[speed_idx]
            if term or trunc:
                done_at = now
                stats["episodes"] += 1
                stats["wins"] += int(info["won"])
                if info["won"]:
                    banner = ("EVACUATED!", f"Both survivors out in {env.state.t} steps", C["good"])
                elif info["partial"]:
                    banner = ("ONE GOT OUT", "One survivor escaped, one fell", C["warn"])
                elif info["lost"]:
                    banner = ("OVERRUN", "Both survivors went down", C["bad"])
                else:
                    banner = ("OUT OF TIME", "100-step limit reached", C["warn"])

        speed = SPEEDS[speed_idx]
        state_txt = "PAUSED" if paused else f"{speed:g} steps/s"
        controls = [
            ("", f"{state_txt}   wins {stats['wins']}/{stats['episodes']}", C["warn"] if paused else C["muted"]),
            ("", "Space pause   Right step   Up/Down speed", C["muted"]),
            ("", "P paths   R restart   N next map   Esc quit", C["muted"]),
        ]
        renderer.draw(screen, env.state, pending=last, seed=seed, total_return=total, banner=banner,
                      paths=getattr(agent, "paths", None) if show_paths else None,
                      subtitle=f"Agent: {args.agent}   config: {args.config}",
                      controls=controls, action_label="DID")
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
