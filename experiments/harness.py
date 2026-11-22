"""Evaluation harness shared by every experiment.

    rows = evaluate(agent, config, seeds, label="astar")
    write_csv(rows, "results/exp1_astar.csv")
    print(summarize(rows))

One row per episode; all Section-7 per-episode metrics are logged here so every
experiment produces comparable CSVs.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from env import DEFAULT_CONFIG, EnvConfig, OutbreakEnv

METRIC_KEYS = ("return", "success", "partial", "deaths", "steps", "lost", "truncated", "kills", "damage",
               "pickups", "mean_partner_dist", "gate_open_step", "ms_per_decision")


def run_episode(env: OutbreakEnv, agent, seed: int, record: bool = False) -> dict:
    obs, info = env.reset(seed=seed)
    agent.reset(seed)
    total, kills, damage, pickups = 0.0, 0, 0, 0
    dists, gate_step, decide_time, n_decisions = [], -1, 0.0, 0
    trajectory = [] if record else None
    terminated = truncated = False
    while not (terminated or truncated):
        t0 = time.perf_counter()
        action = agent.act(obs, env)
        decide_time += time.perf_counter() - t0
        n_decisions += 1
        next_obs, r, terminated, truncated, info = env.step(action)
        agent.observe(obs, action, r, next_obs, terminated, truncated)
        if record:
            trajectory.append((action, r))
        total += r
        ev = info["events"]
        kills += ev["kills"]
        damage += sum(ev["damage"])
        pickups += ev["pickups"]
        s = env.state
        if s.active.all():
            dists.append(int(np.abs(s.surv_pos[0] - s.surv_pos[1]).sum()))
        if info["gate_open"] and gate_step < 0:
            gate_step = s.t
        obs = next_obs
    row = {
        "seed": seed,
        "return": total,
        "success": int(info["won"]),
        "steps": env.state.t,
        "lost": int(info["lost"]),
        "partial": int(info["partial"]),
        "deaths": sum(info["dead"]),
        "truncated": int(truncated),
        "kills": kills,
        "damage": damage,
        "pickups": pickups,
        "mean_partner_dist": float(np.mean(dists)) if dists else 0.0,
        "gate_open_step": gate_step,
        "ms_per_decision": 1000.0 * decide_time / max(n_decisions, 1),
    }
    if record:
        row["trajectory"] = trajectory
    return row


def evaluate(agent, config: EnvConfig | None = None, seeds=(), label: str = "", **extra) -> list[dict]:
    env = OutbreakEnv(config or DEFAULT_CONFIG)
    rows = []
    for seed in seeds:
        row = run_episode(env, agent, seed)
        row.update({"agent": label or agent.name, **extra})
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for k in METRIC_KEYS:
        v = np.array([r[k] for r in rows], dtype=float)
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_std"] = float(v.std())
    return out


def write_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r if k != "trajectory"))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":  # smoke test: python -m experiments.harness
    from agents import RandomAgent
    from experiments.seeds import VAL

    rows = evaluate(RandomAgent(seed=0), DEFAULT_CONFIG, VAL[:50], label="random")
    s = summarize(rows)
    print(f"random agent on {s['n']} val seeds: "
          f"return {s['return_mean']:.2f} +/- {s['return_std']:.2f}, "
          f"success {s['success_mean']:.2%}, steps {s['steps_mean']:.1f}, "
          f"{s['ms_per_decision_mean']:.3f} ms/decision")
