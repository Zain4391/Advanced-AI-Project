"""Agent 2 (Exp 2): two independent tabular Q-learners (cooperative MARL).

Requirements.md section 4: epsilon-greedy tabular Q-learning with decay, trained on random
training maps, one independent learner per survivor, both trained together on the
shared team reward.

Each survivor i has its own Q-table (dict: feature tuple -> 6 action values) and sees only
its own egocentric features obs["agent_i"] (env/state_encoding.q_features). Neither
learner sees the other's action or Q-values. From each learner's point of view the
partner is part of the environment, so its world is non-stationary while the partner
is still learning.

    agent = QLearningAgent(seed=0)            # empty tables: acts randomly ("before")
    agent = QLearningAgent(checkpoint="checkpoints/qlearning/default_seed0.pkl")   # trained

Training is driven by experiments/train_qlearning.py, which calls `update()`. During
evaluation (`training=False`) the agent acts greedily and never updates.

Per-learner episode end: survivor i's episode terminates when *it* becomes inactive
(evacuated or dead) or when the team episode terminates. Truncation at the step limit
is not terminal (we bootstrap), because time is not part of the features.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from env import state_encoding as enc
from env.config import N_ACTIONS, SHOOT, WAIT

from .base import Agent


_ALL_ACTIONS = np.arange(N_ACTIONS)
_NO_SHOOT = np.array([a for a in range(N_ACTIONS) if a != SHOOT])


class QLearningAgent(Agent):
    name = "qlearning"
    observes = "own egocentric features only (obs['agent_i'])"

    FEATURE_SETS = ("spec", "compact", "compact_t")

    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.0,
                 shared_table: bool = False, features: str = "compact", seed: int = 0,
                 checkpoint: str | None = None, alpha_min: float = 0.02,
                 alpha_decay: float = 0.0, mask_shoot: bool = True,
                 hint_prior: float = 0.0):
        if features not in self.FEATURE_SETS:
            raise ValueError(f"features must be one of {self.FEATURE_SETS}")
        self.features = features
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.shared_table = shared_table
        # Per-(state, action) learning rate: alpha_eff = max(alpha_min, alpha / (1 + alpha_decay * n)),
        # n = visits of that pair. alpha_decay = 0 keeps a constant alpha. A decaying rate averages
        # out the noise of the stochastic game (zombie moves, infection), so the greedy policy
        # stops flipping between near-equal actions.
        self.alpha_min = alpha_min
        # Informed initialization: a new Q row gives the exit-hint action a small head start, and a
        # never-visited state is played with the hint action instead of at random. Breaks the ties
        # that make greedy tabular policies loop in rarely visited states (compact features only).
        self.hint_prior = hint_prior if features != "spec" else 0.0
        self.alpha_decay = alpha_decay
        # Action masking: SHOOT is only a real action when the survivor has ammo and a target in
        # its sights (feature `can_shoot`, compact set only). Otherwise it is a wasted turn, and
        # greedy tabular policies can get stuck repeating it.
        self.mask_shoot = mask_shoot and features != "spec"
        self.visits: list[dict[tuple, np.ndarray]] = [{}] if shared_table else [{}, {}]
        self._seed = seed
        self.training = False
        self.tables: list[dict[tuple, np.ndarray]] = [{}] if shared_table else [{}, {}]
        self.rng = np.random.default_rng(seed)
        self.unseen = 0      # greedy decisions taken in a never-visited state (coverage stat)
        self.decisions = 0
        if checkpoint is not None:
            self.load(checkpoint)

    # ------------------------------------------------------------- Agent API ---
    def reset(self, seed: int | None = None) -> None:
        # In evaluation, re-seed per map so results are reproducible. In training, keep
        # one exploration stream for the whole run.
        if not self.training:
            self.rng = np.random.default_rng((self._seed, seed or 0))

    def act(self, obs, env=None) -> tuple[int, int]:
        active = env.state.active if env is not None else (True, True)
        return tuple(self._choose(i, self.feats(obs, env, i)) if active[i] else WAIT for i in (0, 1))

    def feats(self, obs, env, i: int) -> tuple:
        """Survivor i's state key: spec features from obs, or the compact fallback set
        (env/state_encoding.q_features_compact) built from the same snapshot."""
        if self.features == "spec":
            return obs[f"agent_{i}"]
        return enc.q_features_compact(env.state, i, env.config, extended=self.features == "compact_t")

    # ------------------------------------------------------------- learning ---
    def update(self, i: int, s: tuple, a: int, r: float, s2: tuple, done: bool,
               next_forced: int | None = None) -> float:
        """One Q-learning backup for survivor i. Returns the TD error.

        `next_forced`: when survivor i is inactive in s2 (evacuated or dead) its only legal
        action is WAIT, so the bootstrap value is Q(s2, WAIT) rather than max_a Q(s2, a).
        Otherwise the never-tried actions (Q = 0) would look better than the real, negative
        values and inflate the target."""
        q = self._q(i, s, create=True)
        target = r
        if not done:
            q2 = self._q(i, s2)
            if q2 is not None:
                if next_forced is not None:
                    target += self.gamma * float(q2[next_forced])
                else:
                    target += self.gamma * float(q2[self._legal(s2)].max())
        td = target - q[a]
        n = self._visits(i, s)
        alpha = max(self.alpha_min, self.alpha / (1.0 + self.alpha_decay * n[a])) if self.alpha_decay else self.alpha
        n[a] += 1
        q[a] += alpha * td
        return float(td)

    # --------------------------------------------------------------- helpers ---
    def _table(self, i: int) -> dict:
        return self.tables[0 if self.shared_table else i]

    def _visits(self, i: int, s: tuple) -> np.ndarray:
        t = self.visits[0 if self.shared_table else i]
        v = t.get(s)
        if v is None:
            v = t[s] = np.zeros(N_ACTIONS)
        return v

    def _legal(self, s: tuple) -> np.ndarray:
        """Indices of the actions the learner may pick in feature key s."""
        if self.mask_shoot and len(s) > 6 and s[6] == 0:      # compact key: slot 6 = can_shoot
            return _NO_SHOOT
        return _ALL_ACTIONS

    def _q(self, i: int, s: tuple, create: bool = False) -> np.ndarray | None:
        t = self._table(i)
        q = t.get(s)
        if q is None and create:
            q = t[s] = np.zeros(N_ACTIONS)
            if self.hint_prior and s[0] < 4:        # compact key: slot 0 = exit_hint (0..3 = a direction)
                q[s[0]] = self.hint_prior
        return q

    def _choose(self, i: int, s: tuple) -> int:
        eps = self.epsilon if self.training else 0.0
        legal = self._legal(s)
        if eps > 0 and self.rng.random() < eps:
            return int(self.rng.choice(legal))
        q = self._q(i, s)
        self.decisions += 1
        if q is None:
            self.unseen += 1
            if self.hint_prior and s[0] < 4:
                return int(s[0])
            return int(self.rng.choice(legal))
        qm = q[legal]
        best = legal[np.flatnonzero(qm == qm.max())]
        return int(best[0] if len(best) == 1 else self.rng.choice(best))

    @property
    def table_sizes(self) -> list[int]:
        return [len(t) for t in self.tables]

    def coverage(self) -> float:
        """Fraction of greedy decisions made in a state seen during training."""
        return 1.0 - self.unseen / self.decisions if self.decisions else float("nan")

    def reset_coverage(self) -> None:
        self.unseen = self.decisions = 0

    # ------------------------------------------------------------ save/load ---
    def save(self, path: str | Path, **meta) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tables": [{k: v.astype(np.float32) for k, v in t.items()} for t in self.tables],
            "shared_table": self.shared_table, "features": self.features,
            "alpha": self.alpha, "gamma": self.gamma, "seed": self._seed, "meta": meta,
            "visits": [{k: v.astype(np.float32) for k, v in t.items()} for t in self.visits],
            # behaviour flags: without them a loaded agent would act differently from the trained one
            "hint_prior": self.hint_prior, "mask_shoot": self.mask_shoot,
            "alpha_min": self.alpha_min, "alpha_decay": self.alpha_decay,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> None:
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        self.shared_table = payload["shared_table"]
        self.features = payload.get("features", "spec")
        self.tables = [{k: v.astype(np.float64) for k, v in t.items()} for t in payload["tables"]]
        self.alpha, self.gamma = payload["alpha"], payload["gamma"]
        # Checkpoints from before these options existed keep the old behaviour.
        self.hint_prior = payload.get("hint_prior", 0.0)
        self.mask_shoot = bool(payload.get("mask_shoot", False)) and self.features != "spec"
        self.alpha_min = payload.get("alpha_min", self.alpha_min)
        self.alpha_decay = payload.get("alpha_decay", 0.0)
        self.visits = [{k: v.astype(np.float64) for k, v in t.items()}
                       for t in payload.get("visits", [{} for _ in self.tables])]
        self.meta = payload.get("meta", {})
