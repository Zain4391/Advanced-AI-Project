"""Planner cost: ms per MPC decision over a grid of N (samples) and H (horizon).

    python -m experiments.time_mpc                       # N in {100, 200, 400}, H in {6, 8, 12}
    python -m experiments.time_mpc --n 200 --h 8 --kwargs "iters=3"

Times act() on a fixed set of states (the first few steps of some val maps, reached
with A*), so every configuration sees the same positions.
"""
from __future__ import annotations

import argparse
import time

from agents import AStarAgent, MPCAgent
from env import DEFAULT_CONFIG, OutbreakEnv
from experiments.run_benchmark import parse_kwargs
from experiments.seeds import VAL


def bench_states(n_maps: int = 4, steps=(0, 3, 6)) -> list[OutbreakEnv]:
    envs = []
    for seed in VAL[:n_maps]:
        env = OutbreakEnv(DEFAULT_CONFIG)
        obs, _ = env.reset(seed=seed)
        guide = AStarAgent()
        guide.reset(seed)
        for t in range(max(steps) + 1):
            if t in steps:
                envs.append(env.clone())
            obs, _, term, trunc, _ = env.step(guide.act(obs, env))
            if term or trunc:
                break
    return envs


def time_config(envs, **kw) -> float:
    agent = MPCAgent(**kw)
    agent.reset(0)
    t0 = time.perf_counter()
    for e in envs:
        agent.act(None, e)
    return 1000.0 * (time.perf_counter() - t0) / len(envs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[100, 200, 400])
    ap.add_argument("--h", type=int, nargs="+", default=[6, 8, 12])
    ap.add_argument("--kwargs", default="", help="extra MPCAgent kwargs, e.g. iters=3")
    args = ap.parse_args()
    extra = parse_kwargs(args.kwargs)
    envs = bench_states()
    print(f"ms per decision over {len(envs)} states {extra or ''}")
    print("N \\ H " + "".join(f"{h:>8}" for h in args.h))
    for n in args.n:
        cells = [time_config(envs, n_samples=n, horizon=h, **extra) for h in args.h]
        print(f"{n:<6}" + "".join(f"{c:8.0f}" for c in cells))


if __name__ == "__main__":
    main()
