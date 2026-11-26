"""Why do survivors die? Replays an agent on a seed split and attributes every death.

    python -m experiments.diagnose_failures --agent mpc --agent-kwargs "n_samples=200,horizon=8" --workers 4
    python -m experiments.diagnose_failures --agent astar

For each survivor death it records the damage sources over the whole episode
(infected tile vs zombie contact) and on the fatal step, whether the partner had
already left, and how far from the exit the survivor died. It also counts truncated
episodes and "gate waits" (both survivors alive, gate still closed, at least one
next to the exit) as a sign of gate deadlocks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from agents import make_agent
from env import OutbreakEnv
from experiments.run_benchmark import CONFIGS, parse_kwargs
from experiments.seeds import SPLITS


def _damage_sources(s, i) -> tuple[int, int]:
    """(infection, zombie) damage survivor i takes in the state just reached (step order 7)."""
    r, c = s.surv_pos[i]
    inf = int(s.infection[r, c])
    z = s.zombie_pos[s.zombie_alive]
    zom = int(len(z) > 0 and (np.abs(z - s.surv_pos[i]).sum(axis=1) == 1).any())
    return inf, zom


def diagnose_episode(env: OutbreakEnv, agent, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    agent.reset(seed)
    inf_dmg, zom_dmg = [0, 0], [0, 0]
    deaths, gate_wait, done = [], 0, False
    while not done:
        s = env.state
        active = s.active.copy()
        if active.all() and not s.gate_open and (np.abs(s.surv_pos - s.exit_pos).sum(1) <= 1).any():
            gate_wait += 1
        obs, _, term, trunc, info = env.step(agent.act(obs, env))
        done = term or trunc
        s = env.state
        for i in (0, 1):
            if not active[i] or info["events"]["damage"][i] == 0:
                continue
            a, b = _damage_sources(s, i)
            inf_dmg[i] += a
            zom_dmg[i] += b
            if info["events"]["deaths"][i]:
                deaths.append({
                    "inf_total": inf_dmg[i], "zom_total": zom_dmg[i], "fatal": ("inf" if a else "") + ("zom" if b else ""),
                    "partner_out": bool(s.evacuated[1 - i]), "partner_dead": bool(s.dead[1 - i]),
                    "exit_dist": int(s.exit_dist[s.surv_pos[i, 0], s.surv_pos[i, 1]]), "t": s.t,
                })
    return {"seed": seed, "won": info["won"], "partial": info["partial"], "lost": info["lost"],
            "truncated": bool(trunc), "gate_wait": gate_wait, "gate_open": s.gate_open, "deaths": deaths,
            "t_end": s.t, "end_active": s.active.tolist(), "end_evacuated": s.evacuated.tolist(),
            "end_exit_dist": [int(s.exit_dist[r, c]) for r, c in s.surv_pos],
            "end_partner_dist": int(np.abs(s.surv_pos[0] - s.surv_pos[1]).sum()),
            "end_infected_frac": float(s.infection[~s.walls].mean())}


def _chunk(job):
    agent_name, kw, config, seeds = job
    env, agent = OutbreakEnv(CONFIGS[config]), make_agent(agent_name, **kw)
    return [diagnose_episode(env, agent, sd) for sd in seeds]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="mpc")
    ap.add_argument("--agent-kwargs", default="")
    ap.add_argument("--split", default="val")
    ap.add_argument("--config", default="default")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--json", default=None, help="also write every episode record to this file")
    args = ap.parse_args()
    seeds = SPLITS[args.split][: args.episodes]
    kw = parse_kwargs(args.agent_kwargs)
    w = max(1, min(args.workers, len(seeds)))
    jobs = [(args.agent, kw, args.config, seeds[k::w]) for k in range(w)]
    if w == 1:
        eps = _chunk(jobs[0])
    else:
        with ProcessPoolExecutor(w, mp_context=get_context("spawn")) as pool:
            eps = [e for part in pool.map(_chunk, jobs) for e in part]

    eps.sort(key=lambda e: e["seed"])
    if args.json:
        import json
        from pathlib import Path
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(eps, indent=1, default=bool))
    deaths = [d for e in eps for d in e["deaths"]]
    n = len(eps)
    print(f"{args.agent} {kw or ''} on {n} {args.split} maps")
    print(f"  won {sum(e['won'] for e in eps)}, partial {sum(e['partial'] for e in eps)}, "
          f"lost {sum(e['lost'] for e in eps)}, truncated {sum(e['truncated'] for e in eps)}")
    print(f"  deaths: {len(deaths)}")
    if deaths:
        kinds = Counter()
        for d in deaths:
            if d["zom_total"] and d["inf_total"]:
                kinds["both sources"] += 1
            elif d["zom_total"]:
                kinds["zombies only"] += 1
            else:
                kinds["infection only"] += 1
        print("  damage history of the dead:", dict(kinds))
        print("  fatal blow:", dict(Counter(d["fatal"] for d in deaths)))
        print(f"  total damage taken by the dead: infection {sum(d['inf_total'] for d in deaths)}, "
              f"zombies {sum(d['zom_total'] for d in deaths)}")
        print(f"  died after partner evacuated: {sum(d['partner_out'] for d in deaths)}; "
              f"exit distance at death: median {np.median([d['exit_dist'] for d in deaths]):.0f}; "
              f"t at death: median {np.median([d['t'] for d in deaths]):.0f}")
        near = sum(d["exit_dist"] <= 2 for d in deaths)
        print(f"  died within 2 tiles of the exit: {near}")
    gw = [e["gate_wait"] for e in eps]
    print(f"  gate waits (both alive, gate closed, one next to exit): mean {np.mean(gw):.1f} steps, "
          f"episodes with >= 5: {sum(g >= 5 for g in gw)}; gate never opened in {sum(not e['gate_open'] for e in eps)}")


if __name__ == "__main__":
    main()
