"""Evaluate an agent headless on a fixed seed split and write a CSV.

    python -m experiments.run_benchmark --agent astar                       # Exp 1, val split
    python -m experiments.run_benchmark --agent astar --split test
    python -m experiments.run_benchmark --agent astar --split shifted --config infection_2p
    python -m experiments.run_benchmark --agent astar --config no_coord     # coordination ablation
    python -m experiments.run_benchmark --agent mpc --exp exp3 --agent-kwargs "n_samples=200,horizon=8" --workers 4

Output: results/{exp}_{agent}_{config}_{split}.csv  (one row per episode) + printed summary.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from agents import AGENTS, make_agent
from env import DEFAULT_CONFIG, NO_COORD_CONFIG, SHIFTED_CONFIGS
from experiments.harness import evaluate, summarize, write_csv
from experiments.seeds import SPLITS

CONFIGS = {"default": DEFAULT_CONFIG, "no_coord": NO_COORD_CONFIG, **SHIFTED_CONFIGS}


def parse_kwargs(text: str | None) -> dict:
    """'n_samples=200,leaf=false,p_hint=0.5' -> {'n_samples': 200, 'leaf': False, 'p_hint': 0.5}."""
    out = {}
    for item in filter(None, (t.strip() for t in (text or "").split(","))):
        key, _, val = item.partition("=")
        low = val.strip().lower()
        if low in ("true", "false"):
            v = low == "true"
        elif low in ("none", "null"):
            v = None
        else:
            try:
                v = int(val)
            except ValueError:
                try:
                    v = float(val)
                except ValueError:
                    v = val.strip()
        out[key.strip()] = v
    return out


def _eval_chunk(job: tuple) -> list[dict]:
    """One worker: build the agent fresh and evaluate a chunk of map seeds."""
    agent_name, agent_kw, config_name, seeds, extra = job
    agent = make_agent(agent_name, **agent_kw)
    extra = {**extra, **getattr(agent, "csv_extra", {})}
    return evaluate(agent, CONFIGS[config_name], seeds, label=agent_name, **extra)


def run(agent_name: str, agent_kw: dict, config_name: str, seeds, workers: int = 1, **extra) -> list[dict]:
    """Evaluate on `seeds`, split over `workers` spawn-safe processes. Rows are sorted by seed,
    so the result matches workers=1 (agents reseed from the map seed in reset())."""
    seeds = list(seeds)
    workers = max(1, min(int(workers), len(seeds)))
    extra = {**extra, "workers": workers}
    if workers == 1:
        rows = _eval_chunk((agent_name, agent_kw, config_name, seeds, extra))
    else:
        chunks = [seeds[k::workers] for k in range(workers)]  # interleaved: balances map difficulty
        jobs = [(agent_name, agent_kw, config_name, c, extra) for c in chunks]
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            rows = [r for part in pool.map(_eval_chunk, jobs) for r in part]
    return sorted(rows, key=lambda r: r["seed"])


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
    ap.add_argument("--agent-kwargs", default="", help='constructor kwargs, e.g. "n_samples=200,horizon=8,iters=1"')
    ap.add_argument("--workers", type=int, default=1, help="processes; map seeds are split across them")
    args = ap.parse_args()

    seeds = SPLITS[args.split][: args.episodes]
    t0 = time.perf_counter()
    kw = {"seed": args.agent_seed, **parse_kwargs(args.agent_kwargs)}
    if args.checkpoint:
        kw["checkpoint"] = args.checkpoint
    rows = run(args.agent, kw, args.config, seeds, args.workers,
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
