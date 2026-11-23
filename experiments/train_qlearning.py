"""Train two independent Q-learners together (Exp 2) and log the team learning curve.

    python -m experiments.train_qlearning --episodes 30000 --run-seed 0
    python -m experiments.train_qlearning --episodes 30000 --run-seed 1
    python -m experiments.train_qlearning --episodes 30000 --run-seed 2
    python -m analysis.plots curve        # mean +/- std learning curve over run seeds

Training maps: the TRAIN split (seeds 0-999), reshuffled each pass with the run seed.
Every --eval-every episodes, the greedy policy (epsilon = 0, no updates) is evaluated on
experiments.seeds.CURVE_EVAL (fixed validation maps), as Requirements.md section 7 asks.

Outputs
  results/exp2_qlearning_curve_<config>_seed<k>.csv   learning curve (+ env steps, for sample efficiency)
  checkpoints/qlearning/<config>_seed<k>.pkl           final Q-tables
  checkpoints/qlearning/<config>_seed<k>_ep<N>.pkl     early snapshot for the before/after demo
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from agents import QLearningAgent
from env import DEFAULT_CONFIG, NO_COORD_CONFIG, WAIT, OutbreakEnv
from experiments.harness import evaluate, summarize
from experiments.seeds import CURVE_EVAL, TRAIN, VAL

CONFIGS = {"default": DEFAULT_CONFIG, "no_coord": NO_COORD_CONFIG}


def epsilon_at(ep: int, n: int, start: float, end: float, decay_frac: float) -> float:
    """Linear decay from start to end over the first decay_frac of training, then constant."""
    t = min(1.0, ep / max(1, decay_frac * n))
    return start + (end - start) * t


def potential(env: OutbreakEnv, i: int) -> float:
    """1 - (BFS distance to exit) / grid area, in [0, 1]. Used only when --shaping > 0."""
    s = env.state
    r, c = s.surv_pos[i]
    # Phi must be >= 0 and grow toward the exit. With a negative potential (e.g. -dist),
    # standing still earns (gamma - 1) * Phi > 0, which cancels the step cost and teaches
    # the agent that bumping into walls is free. Here staying put costs (1 - gamma) * Phi.
    return 1.0 - float(min(s.exit_dist[r, c], s.walls.size)) / s.walls.size


def train(cfg, *, episodes: int, run_seed: int = 0, alpha: float = 0.1, gamma: float = 0.95,
          eps_start: float = 1.0, eps_end: float = 0.05, eps_decay_frac: float = 0.8,
          shared_table: bool = False, features: str = "compact", shaping: float = 0.0,
          absorbing: bool = True, alpha_min: float = 0.02, alpha_decay: float = 0.0,
          mask_shoot: bool = True, hint_prior: float = 0.0,
          eval_every: int = 200, eval_seeds=CURVE_EVAL, snapshot_at: int | None = None,
          on_snapshot=None, verbose: bool = True, agent: QLearningAgent | None = None,
          eval_cfg=None, ep_offset: int = 0, steps_offset: int = 0):
    """Train two independent Q-learners together. Returns (agent, curve_rows, env_steps).

    Pass `agent` to continue training an existing learner (curriculum stages). `eval_cfg`
    (default: cfg) is the game the learning curve is measured on; `ep_offset` and
    `steps_offset` keep episode / env-step counts cumulative across stages."""
    env = OutbreakEnv(cfg)
    eval_cfg = eval_cfg or cfg
    if agent is None:
        agent = QLearningAgent(alpha=alpha, gamma=gamma, shared_table=shared_table,
                               features=features, seed=run_seed, alpha_min=alpha_min,
                               alpha_decay=alpha_decay, mask_shoot=mask_shoot,
                               hint_prior=hint_prior)
    rng = np.random.default_rng(1000 + run_seed)
    order = rng.permutation(TRAIN)
    curve, recent_ret, recent_succ = [], [], []
    env_steps, t0 = steps_offset, time.perf_counter()

    def log_eval(ep: int) -> None:
        agent.training = False
        agent.reset_coverage()
        s = summarize(evaluate(agent, eval_cfg, eval_seeds))
        row = {
            "episode": ep + ep_offset, "env_steps": env_steps, "epsilon": round(agent.epsilon, 4),
            "eval_return": s["return_mean"], "eval_return_std": s["return_std"],
            "eval_success": s["success_mean"], "eval_partial": s["partial_mean"],
            "eval_lost": s["lost_mean"], "eval_steps": s["steps_mean"],
            "eval_coverage": agent.coverage(),
            "train_return": float(np.mean(recent_ret)) if recent_ret else float("nan"),
            "train_success": float(np.mean(recent_succ)) if recent_succ else float("nan"),
            "states_s1": agent.table_sizes[0], "states_s2": agent.table_sizes[-1],
            "wall_time_s": time.perf_counter() - t0,
        }
        curve.append(row)
        recent_ret.clear()
        recent_succ.clear()
        agent.training = True
        if verbose:
            print(f"ep {ep + ep_offset:6d} | steps {env_steps:8d} | eps {row['epsilon']:.3f} | "
                  f"eval success {row['eval_success']:5.1%} partial {row['eval_partial']:5.1%} "
                  f"return {row['eval_return']:7.2f} | train ret {row['train_return']:7.2f} | "
                  f"states {row['states_s1']}/{row['states_s2']} | cov {row['eval_coverage']:.2f} | "
                  f"{row['wall_time_s']:.0f}s", flush=True)

    agent.training = True
    if ep_offset == 0:
        log_eval(0)
    for ep in range(1, episodes + 1):
        agent.epsilon = epsilon_at(ep, episodes, eps_start, eps_end, eps_decay_frac)
        if (ep - 1) % len(order) == 0 and ep > 1:
            order = rng.permutation(TRAIN)
        seed = int(order[(ep - 1) % len(order)])
        obs, _ = env.reset(seed=seed)
        ep_ret, done = 0.0, False
        feats = [agent.feats(obs, env, i) for i in (0, 1)]
        while not done:
            active_before = env.state.active.copy()
            phi = [potential(env, i) for i in (0, 1)] if shaping else None
            action = agent.act(obs, env)
            obs2, r, term, trunc, info = env.step(action)
            env_steps += 1
            ep_ret += r
            active_after = env.state.active
            feats2 = [agent.feats(obs2, env, i) for i in (0, 1)]
            for i in (0, 1):
                if not active_before[i] and not absorbing:
                    continue
                # With `absorbing`, a survivor that has left the game (evacuated or dead) keeps
                # learning in its absorbing state until the team episode ends, so the shared
                # win/partial reward reaches the first survivor to evacuate. Without it, the
                # learner's episode ends the moment it becomes inactive (original behaviour).
                done_i = term or (not absorbing and not active_after[i])
                r_i = r
                if shaping and active_before[i]:
                    # potential of an inactive survivor: 1 if it evacuated, 0 if it died
                    phi_next = potential(env, i) if active_after[i] else (0.0 if env.state.health[i] <= 0 else 1.0)
                    r_i += shaping * (gamma * (0.0 if done_i else phi_next) - phi[i])
                agent.update(i, feats[i], action[i], r_i, feats2[i], done_i,
                             next_forced=None if active_after[i] else WAIT)
            obs, feats = obs2, feats2
            done = term or trunc
        recent_ret.append(ep_ret)
        recent_succ.append(int(info["won"]))
        if snapshot_at and ep == snapshot_at and on_snapshot:
            on_snapshot(agent, ep, env_steps)
        if eval_every and (ep % eval_every == 0 or ep == episodes):
            log_eval(ep)          # also at the end of a stage, so every curriculum stage has a point
    agent.training = False
    return agent, curve, env_steps


def curriculum_stages(cfg, scale: float = 1.0):
    """Easy-to-hard training stages for the full game `cfg` (hazards switched on one at a time).

    (name, config, episodes, epsilon at the start of the stage). The learners keep their
    Q-tables between stages, so what is learned about reaching the exit and waiting at the
    gate carries over when the infection and then the zombies are switched on."""
    no_hazards = cfg.with_(n_zombies=0, infection_prob=0.0, n_initial_infection=0)
    stages = [("no_hazards", no_hazards, 2000, 1.0),
              ("infection", cfg.with_(n_zombies=0), 2500, 0.2),
              ("zombies", cfg.with_(infection_prob=0.0, n_initial_infection=0), 4000, 0.3)]
    return [(name, c, max(1, int(n * scale)), e0) for name, c, n, e0 in stages]


def train_curriculum(cfg, *, episodes: int, run_seed: int = 0, final_eps_start: float = 0.2,
                     scale: float = 1.0, verbose: bool = True, **kw):
    """Run the curriculum stages, then `episodes` episodes on the full game `cfg`.

    The learning curve is always measured on the full game (eval_cfg=cfg) on the fixed
    CURVE_EVAL maps, with cumulative episode and env-step counts, so it is directly comparable
    with a run that trains on the full game only."""
    agent, curve, steps, ep_off = None, [], 0, 0
    eval_every = kw.pop("eval_every", 200)
    for name, stage_cfg, n, e0 in curriculum_stages(cfg, scale) + [("full", cfg, episodes, final_eps_start)]:
        if verbose:
            print(f"== stage {name}: {n} episodes, epsilon {e0} -> {kw.get('eps_end', 0.05)}", flush=True)
        agent, rows, steps = train(stage_cfg, episodes=n, run_seed=run_seed, eps_start=e0,
                                   eval_every=eval_every, agent=agent, eval_cfg=cfg,
                                   ep_offset=ep_off, steps_offset=steps, verbose=verbose, **kw)
        for r in rows:
            r["stage"] = name
        curve += rows
        ep_off += n
    return agent, curve, steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--run-seed", type=int, default=0)
    ap.add_argument("--config", default="default", choices=sorted(CONFIGS))
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay-frac", type=float, default=0.5,
                    help="epsilon decays linearly over this fraction of each training stage")
    ap.add_argument("--shared-table", action="store_true", help="parameter sharing fallback")
    ap.add_argument("--features", default="compact_t", choices=["compact_t", "compact", "spec"],
                    help="state features: compact_t (compact + hp bucket + time bucket, default), "
                         "compact (section 14 fallback) or the section 3 spec set")
    ap.add_argument("--shaping", type=float, default=10.0,
                    help="potential-based shaping weight on exit distance, training only (0 = off)")
    ap.add_argument("--alpha-min", type=float, default=0.02)
    ap.add_argument("--alpha-decay", type=float, default=0.02,
                    help="per-(state, action) learning-rate decay: alpha / (1 + decay * visits); 0 = constant")
    ap.add_argument("--hint-prior", type=float, default=0.5,
                    help="head start for the exit-hint action in new Q rows (0 = off)")
    ap.add_argument("--no-mask-shoot", action="store_true", help="allow SHOOT with no target in sight")
    ap.add_argument("--no-absorbing", action="store_true",
                    help="old behaviour: stop updating a survivor once it evacuates or dies")
    ap.add_argument("--curriculum-scale", type=float, default=1.0,
                    help="multiply the episode count of the easy curriculum stages (quick smoke tests)")
    ap.add_argument("--curriculum", action="store_true",
                    help="train on easier games first (no hazards -> infection -> zombies -> full game)")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--snapshot-at", type=int, default=1000, help="save an early checkpoint here")
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. _shared")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    name = f"{args.config}{args.tag}_seed{args.run_seed}"
    ckpt_dir = Path("checkpoints/qlearning")
    curve_path = Path(f"results/exp2_qlearning_curve_{name}.csv")
    curve_path.parent.mkdir(parents=True, exist_ok=True)

    def snapshot(agent, ep, steps):
        agent.save(ckpt_dir / f"{name}_ep{ep}.pkl", episodes=ep, env_steps=steps, args=vars(args))

    kw = dict(alpha=args.alpha, gamma=args.gamma, eps_end=args.eps_end,
              eps_decay_frac=args.eps_decay_frac, shared_table=args.shared_table,
              features=args.features, shaping=args.shaping, alpha_min=args.alpha_min,
              alpha_decay=args.alpha_decay, hint_prior=args.hint_prior,
              mask_shoot=not args.no_mask_shoot, absorbing=not args.no_absorbing,
              eval_every=args.eval_every)
    if args.curriculum:
        agent, curve, env_steps = train_curriculum(cfg, episodes=args.episodes,
                                                   run_seed=args.run_seed,
                                                   scale=args.curriculum_scale, **kw)
    else:
        agent, curve, env_steps = train(
            cfg, episodes=args.episodes, run_seed=args.run_seed, eps_start=args.eps_start,
            snapshot_at=args.snapshot_at, on_snapshot=snapshot, **kw)

    agent.save(ckpt_dir / f"{name}.pkl", episodes=args.episodes, env_steps=env_steps, args=vars(args))
    with curve_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in curve for k in r)))
        w.writeheader()
        w.writerows(curve)

    # final greedy check on the full validation split
    agent.reset_coverage()
    s = summarize(evaluate(agent, cfg, VAL))
    print(f"\nfinal greedy policy on VAL ({s['n']} maps): success {s['success_mean']:.1%}, "
          f"partial {s['partial_mean']:.1%}, lost {s['lost_mean']:.1%}, "
          f"return {s['return_mean']:.2f} +/- {s['return_std']:.2f}, coverage {agent.coverage():.2f}")
    print(f"curve -> {curve_path}\ncheckpoint -> {ckpt_dir / (name + '.pkl')}")


if __name__ == "__main__":
    main()
