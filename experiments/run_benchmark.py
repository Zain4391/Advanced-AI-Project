"""Evaluate an agent headless on a fixed seed split and write a CSV.

    python -m experiments.run_benchmark --agent astar                       # Exp 1, val split
    python -m experiments.run_benchmark --agent astar --split test
    python -m experiments.run_benchmark --agent astar --split shifted --config infection_2p
    python -m experiments.run_benchmark --agent astar --config no_coord     # coordination ablation

Output: results/{exp}_{agent}_{config}_{split}.csv  (one row per episode) + printed summary.
"""
from __future__ import annotations

import argparse
import time

from agents import AGENTS, make_agent
from env import DEFAULT_CONFIG, NO_COORD_CONFIG, SHIFTED_CONFIGS
from experiments.harness import evaluate, summarize, write_csv
from experiments.seeds import SPLITS

CONFIGS = {"default": DEFAULT_CONFIG, "no_coord": NO_COORD_CONFIG, **SHIFTED_CONFIGS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="astar", choices=sorted(AGENTS))
    ap.add_argument("--split", default="val", choices=sorted(SPLITS))
    ap.add_argument("--config", default="default", choices=sorted(CONFIGS))
    ap.add_argument("--episodes", type=int, default=None, help="limit number of seeds")
    ap.add_argument("--exp", default="exp1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=None, help="trained agent file (e.g. Q-tables)")
    ap.add_argument("--agent-seed", type=int, default=0)
    args = ap.parse_args()

    seeds = SPLITS[args.split][: args.episodes]
    t0 = time.perf_counter()
    kw = {"seed": args.agent_seed}
    if args.checkpoint:
        kw["checkpoint"] = args.checkpoint
    rows = evaluate(make_agent(args.agent, **kw), CONFIGS[args.config], seeds, label=args.agent,
                    condition=args.config, split=args.split, agent_seed=args.agent_seed,
                    checkpoint=args.checkpoint or "")
    suffix = f"_seed{args.agent_seed}" if args.checkpoint or args.agent_seed else ""
    out = args.out or f"results/{args.exp}_{args.agent}_{args.config}_{args.split}{suffix}.csv"
    write_csv(rows, out)

    s = summarize(rows)
    print(f"{args.agent} | config={args.config} split={args.split} | {s['n']} episodes "
          f"in {time.perf_counter() - t0:.1f}s -> {out}")
    for k in ("success", "partial", "lost", "truncated"):
        print(f"  {k:<18} {s[k + '_mean']:7.1%}")
    for k in ("return", "steps", "deaths", "damage", "kills", "pickups", "mean_partner_dist", "ms_per_decision"):
        print(f"  {k:<18} {s[k + '_mean']:7.2f} +/- {s[k + '_std']:.2f}")


if __name__ == "__main__":
    main()
