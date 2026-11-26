"""Sampling-based MPC over joint actions (agents 3, 4 and 5 in Requirements.md §4).

One planner serves Exp 3, 4 and 5; only the dynamics backend changes:

    Exp 3: TrueSimDynamics          (a clone of the real simulator)
    Exp 4: a learned world model    (same Dynamics protocol, filled in later)
    Exp 5: the learned ensemble     (also fills RolloutBatch.disagreement; lam > 0)

Every step the planner samples N joint-action sequences of horizon H from a
per-(step, survivor) categorical, rolls each one out with the backend, scores the
resulting RolloutBatch, executes the first joint action of the best sequence and
replans at the next step. iters > 1 refits the categoricals to the elite sequences
between rounds (the CEM option).

All sequences of one decision are evaluated under the same n_futures dynamics
seeds (common random numbers), so differences in score come from the actions, not
from luckier futures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from env.config import N_ACTIONS, SHOOT, WAIT
from env.state import EnvState
from env.state_encoding import exit_hint

from .base import Agent


# ------------------------------------------------------------ rollout data ---
@dataclass
class RolloutBatch:
    """What a dynamics backend returns for N sequences of horizon H.

    rewards[n, t] is the team reward of step t (0 after the rollout ended), and
    dones[n, t] is True from the step on which the rollout ended (terminated or
    truncated) onwards. The final_* fields describe the last state reached.
    """
    rewards: np.ndarray           # [N, H] float
    dones: np.ndarray             # [N, H] bool
    final_pos: np.ndarray         # [N, 2, 2] int (row, col) per survivor
    final_hp: np.ndarray          # [N, 2] int
    final_evacuated: np.ndarray   # [N, 2] bool
    final_done: np.ndarray        # [N] bool: rollout ended (terminated or truncated) within H
    disagreement: np.ndarray      # [N] float: ensemble disagreement (zeros for the true simulator)
    #: optional [N, H, 2, 2] survivor positions after each step (for drawing the plan)
    pos: np.ndarray | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.rewards)


class Dynamics(Protocol):
    """A dynamics backend. Must not mutate the live env."""

    def rollout(self, env, sequences: np.ndarray, seeds: np.ndarray) -> RolloutBatch:
        """Roll out sequences[N, H, 2] (joint actions) from the env's current state.

        seeds[N] picks the sampled future of each row; rows with the same seed and
        the same actions must give the same result (common random numbers)."""
        ...


class TrueSimDynamics:
    """Rollouts in a clone of the real simulator (Exp 3). Only this backend may use
    env.clone(), get_state() and reseed_dynamics()."""

    name = "true"

    def rollout(self, env, sequences: np.ndarray, seeds: np.ndarray) -> RolloutBatch:
        n, h = sequences.shape[:2]
        sim = env.clone()                       # the live env is never touched
        snap = sim.get_state()
        rewards = np.zeros((n, h), np.float64)
        dones = np.zeros((n, h), bool)
        pos = np.zeros((n, h, 2, 2), np.int64)
        final_pos = np.zeros((n, 2, 2), np.int64)
        final_hp = np.zeros((n, 2), np.int64)
        final_evac = np.zeros((n, 2), bool)
        final_done = np.zeros(n, bool)
        seqs = sequences.tolist()
        for k in range(n):
            sim.set_state(snap)
            sim.reseed_dynamics(int(seeds[k]))
            s = sim.state
            for t in range(h):
                a1, a2 = seqs[k][t]
                _, r, term, trunc, _ = sim.step((a1, a2), build_obs=False)
                s = sim.state
                rewards[k, t] = r
                pos[k, t] = s.surv_pos
                if term or trunc:
                    dones[k, t:] = True
                    pos[k, t + 1:] = s.surv_pos
                    final_done[k] = True
                    break
            final_pos[k] = s.surv_pos
            final_hp[k] = s.health
            final_evac[k] = s.evacuated
        return RolloutBatch(rewards, dones, final_pos, final_hp, final_evac, final_done,
                            np.zeros(n), pos)


# ----------------------------------------------------------------- scoring ---
def leaf_value(batch: RolloutBatch, exit_dist: np.ndarray, dist_weight: float = 0.5) -> np.ndarray:
    """-dist_weight * (d1 + d2) + 2 * (hp1 + hp2) at the final state (spec: dist_weight = 0.5). d is the BFS distance to
    the exit, 0 once evacuated; dead survivors contribute no distance (and hp 0)."""
    r, c = batch.final_pos[..., 0], batch.final_pos[..., 1]
    d = exit_dist[r, c].astype(np.float64)
    dead = batch.final_hp <= 0
    d[batch.final_evacuated | dead] = 0.0
    hp = np.clip(batch.final_hp, 0, None)
    return -dist_weight * d.sum(1) + 2.0 * hp.sum(1)


def score_rollouts(batch: RolloutBatch, exit_dist: np.ndarray, gamma: float = 0.98,
                   leaf: bool = True, lam: float = 0.0, leaf_dist_weight: float = 0.5) -> np.ndarray:
    """[N] scores: sum_t gamma^t r_t, + leaf value if the rollout did not end, - lam * disagreement."""
    h = batch.rewards.shape[1]
    score = (batch.rewards * gamma ** np.arange(h)).sum(1)
    if leaf:
        score = score + np.where(batch.final_done, 0.0, leaf_value(batch, exit_dist, leaf_dist_weight))
    if lam:
        score = score - lam * batch.disagreement
    return score


# ----------------------------------------------------------------- sampler ---
def action_mask(s: EnvState) -> np.ndarray:
    """[2, 6] bool: actions worth sampling at the root. Inactive survivors only WAIT;
    SHOOT needs ammo (with 0 ammo it is just a WAIT)."""
    mask = np.ones((2, N_ACTIONS), bool)
    for i in (0, 1):
        if not s.active[i]:
            mask[i] = False
            mask[i, WAIT] = True
        elif s.ammo[i] <= 0:
            mask[i, SHOOT] = False
    return mask


def initial_probs(s: EnvState, horizon: int, p_hint: float) -> np.ndarray:
    """[H, 2, 6] categoricals: p_hint on the root exit-hint direction, the rest uniform
    over the other allowed actions (uniform over all allowed ones if the hint is WAIT)."""
    mask = action_mask(s)
    probs = np.zeros((2, N_ACTIONS))
    for i in (0, 1):
        allowed = mask[i].astype(float)
        hint = exit_hint(s, i)
        if s.active[i] and hint != WAIT and allowed.sum() > 1:
            others = allowed.copy()
            others[hint] = 0.0
            probs[i] = (1.0 - p_hint) * others / others.sum()
            probs[i, hint] = p_hint
        else:
            probs[i] = allowed / allowed.sum()
    return np.broadcast_to(probs, (horizon, 2, N_ACTIONS)).copy()


def sample_sequences(probs: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n sequences [n, H, 2] from per-(step, survivor) categoricals probs[H, 2, 6]."""
    cdf = np.cumsum(probs, axis=-1)
    cdf[..., -1] = 1.0                                  # guard against rounding
    u = rng.random((n, *probs.shape[:2]))
    a = (cdf[None] <= u[..., None]).sum(-1)             # a zero-probability action is never picked
    return np.minimum(a, N_ACTIONS - 1).astype(np.int64)


def refit_probs(probs: np.ndarray, elites: np.ndarray, mask: np.ndarray, smoothing: float) -> np.ndarray:
    """CEM update: mix the elite action frequencies with the old categoricals.

    new = smoothing * old + (1 - smoothing) * freq(elites), then re-mask and renormalize."""
    freq = np.zeros_like(probs)
    h = probs.shape[0]
    for i in (0, 1):
        for t in range(h):
            freq[t, i] = np.bincount(elites[:, t, i], minlength=N_ACTIONS)
    freq /= max(len(elites), 1)
    new = smoothing * probs + (1.0 - smoothing) * freq
    new = new * mask[None]
    return new / new.sum(-1, keepdims=True)


# ------------------------------------------------------------------- agent ---
class MPCAgent(Agent):
    """Centralized sampling-based MPC for both survivors (Exp 3; Exp 4/5 swap `dynamics`)."""

    name = "mpc"
    observes = "full state + dynamics model"

    def __init__(self, n_samples: int = 200, horizon: int = 8, iters: int = 1, elite_frac: float = 0.1,
                 p_hint: float = 0.5, gamma: float = 0.98, n_futures: int = 1, leaf: bool = True,
                 lam: float = 0.0, smoothing: float = 0.3, leaf_dist_weight: float = 0.5, seed: int = 0, dynamics: Dynamics | None = None):
        self.n_samples, self.horizon, self.iters = int(n_samples), int(horizon), int(iters)
        self.elite_frac, self.p_hint, self.gamma = float(elite_frac), float(p_hint), float(gamma)
        self.n_futures, self.leaf, self.lam = int(n_futures), bool(leaf), float(lam)
        self.smoothing = float(smoothing)
        self.leaf_dist_weight = float(leaf_dist_weight)   # spec value 0.5; tuned in Exp 3
        self.seed = int(seed)
        self.dynamics: Dynamics = dynamics or TrueSimDynamics()
        self.rng = np.random.default_rng(self.seed)
        self.paths: list[list[tuple[int, int]] | None] = [None, None]  # best plan, for scripts/watch.py
        self.last_score: float = 0.0

    @property
    def csv_extra(self) -> dict:
        """Planner settings recorded as extra CSV columns."""
        return {"N": self.n_samples, "H": self.horizon, "iters": self.iters, "lam": self.lam,
                "p_hint": self.p_hint, "n_futures": self.n_futures, "leaf": int(self.leaf), "leaf_dist_weight": self.leaf_dist_weight,
                "dynamics": getattr(self.dynamics, "name", type(self.dynamics).__name__)}

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.seed if seed is None else (self.seed, int(seed)))
        self.paths = [None, None]

    def _evaluate(self, env, seqs: np.ndarray, future_seeds: np.ndarray, exit_dist):
        """Average score of each sequence over the shared futures. Identical sequences are
        rolled out once (under common random numbers they give identical results)."""
        flat = seqs.reshape(len(seqs), -1)
        uniq, inv = np.unique(flat, axis=0, return_inverse=True)
        uniq = uniq.reshape(-1, *seqs.shape[1:])
        total = np.zeros(len(uniq))
        best_batch = None
        for fs in future_seeds:
            batch = self.dynamics.rollout(env, uniq, np.full(len(uniq), fs))
            total += score_rollouts(batch, exit_dist, self.gamma, self.leaf, self.lam, self.leaf_dist_weight)
            if best_batch is None:
                best_batch = batch
        return (total / len(future_seeds))[inv.reshape(-1)], best_batch, inv.reshape(-1)

    def act(self, obs, env) -> tuple[int, int]:
        s = env.state                                    # read only
        mask = action_mask(s)
        probs = initial_probs(s, self.horizon, self.p_hint)
        future_seeds = self.rng.integers(2**31 - 1, size=self.n_futures)
        n_elite = max(1, int(round(self.elite_frac * self.n_samples)))
        best_seq, best_score, best_pos = None, -np.inf, None
        for it in range(self.iters):
            seqs = sample_sequences(probs, self.n_samples, self.rng)
            scores, batch, inv = self._evaluate(env, seqs, future_seeds, s.exit_dist)
            k = int(np.argmax(scores))
            if scores[k] > best_score:
                best_seq, best_score = seqs[k], float(scores[k])
                best_pos = None if batch.pos is None else batch.pos[inv[k]]
            if it + 1 < self.iters:
                elites = seqs[np.argsort(-scores, kind="stable")[:n_elite]]
                probs = refit_probs(probs, elites, mask, self.smoothing)
        self.last_score = best_score
        self._set_paths(s, best_seq, best_pos)
        return int(best_seq[0, 0]), int(best_seq[0, 1])

    def _set_paths(self, s: EnvState, seq, pos) -> None:
        self.paths = [None, None]
        if pos is None:
            return
        for i in (0, 1):
            if s.active[i]:
                pts = [tuple(int(x) for x in s.surv_pos[i])]
                for p in pos[:, i]:
                    q = (int(p[0]), int(p[1]))
                    if q != pts[-1]:
                        pts.append(q)
                self.paths[i] = pts
