"""OutbreakEnv: two cooperating survivors, scripted zombies, spreading infection.

Pure NumPy game logic -- never import pygame here. The order in which a step is
resolved is part of the frozen spec (docs/interfaces.md, "Step order"):

    1. shots        (from positions at the start of the step)
    2. movement     (simultaneous; conflicts -> stay)
    3. pickups
    4. gate / evacuation
    5. zombie moves
    6. infection spread
    7. damage
    8. t += 1, supply drop
    9. deaths, win / lose / truncation
"""
from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import state_encoding as enc
from .config import (
    MOVE_DELTAS, N_ACTIONS, N_CHANNELS, N_SCALARS, N_ZOMBIE_MOVES, SHOOT, WAIT,
    DEFAULT_CONFIG, EnvConfig, split_joint_action,
)
from .map_gen import generate_map
from .state import EnvState

_NEIGH4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


class OutbreakEnv(gym.Env):
    """Gymnasium env with a joint action (a1, a2) and a shared team reward.

    obs = {
        "grid":    (N_CHANNELS, H, W) float32,
        "scalars": (N_SCALARS,) float32,
        "agent_0": tuple  (Q-learning features for survivor 1),
        "agent_1": tuple  (Q-learning features for survivor 2),
    }
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG
        n = self.config.grid_size
        self.action_space = spaces.MultiDiscrete([N_ACTIONS, N_ACTIONS])
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, (N_CHANNELS, n, n), np.float32),
            "scalars": spaces.Box(0.0, np.inf, (N_SCALARS,), np.float32),
        })
        self.state: EnvState | None = None
        self.episode_seed: int | None = None
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------ gym API ---
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(np.random.default_rng().integers(2**31 - 1))
        self.episode_seed = int(seed)
        self._rng = np.random.default_rng(self.episode_seed)
        self.state = generate_map(self.config, self._rng)
        return self._obs(), self._info(events=None)

    def step(self, action):
        s, cfg = self.state, self.config
        if s is None:
            raise RuntimeError("call reset() before step()")
        if s.done:
            raise RuntimeError("episode is over; call reset()")
        a = self._parse_action(action)
        rng = self._rng
        active = s.active.copy()
        ev = {"kills": 0, "pickups": 0, "damage": [0, 0], "shots": [False, False], "shot_targets": [-1, -1], "deaths": [False, False],
              "zombie_moves": np.full((cfg.zombie_moves_per_step, cfg.max_zombies), WAIT, dtype=np.int8)}

        # 1. shots -------------------------------------------------------------
        for i in (0, 1):
            if active[i] and a[i] == SHOOT and s.ammo[i] > 0:
                z = shot_target(s, cfg, i)
                s.ammo[i] -= 1
                ev["shots"][i] = True
                if z is not None:
                    s.zombie_alive[z] = False
                    ev["kills"] += 1
                    ev["shot_targets"][i] = z

        # 2. movement ----------------------------------------------------------
        start = s.surv_pos.copy()
        target = start.copy()
        zombie_cells = {tuple(p) for p in s.zombie_pos[s.zombie_alive]}
        for i in (0, 1):
            if active[i] and a[i] < 4:
                t = start[i] + MOVE_DELTAS[a[i]]
                if self._walkable(s, t) and tuple(t) not in zombie_cells:
                    j = 1 - i
                    if not (active[j] and (t == start[j]).all()):  # can't step onto partner
                        target[i] = t
        if active.all() and (target[0] == target[1]).all():
            both_exit = s.gate_open and (target[0] == s.exit_pos).all()
            if not both_exit:  # both evacuating together is the only legal overlap
                target = start.copy()
        s.surv_pos = target

        # 3. pickups -----------------------------------------------------------
        for i in (0, 1):
            if not active[i]:
                continue
            r, c = s.surv_pos[i]
            if s.medkits[r, c]:
                s.medkits[r, c] = False
                ev["pickups"] += 1
                j = 1 - i
                heal = [i]
                if cfg.coop_medkit and active[j] and np.abs(s.surv_pos[j] - s.surv_pos[i]).sum() <= 1:
                    heal.append(j)
                for k in heal:
                    s.health[k] = min(cfg.max_health, s.health[k] + cfg.medkit_heal)
            if s.ammo_crates[r, c]:
                s.ammo_crates[r, c] = False
                ev["pickups"] += 1
                s.ammo[i] = min(cfg.max_ammo, s.ammo[i] + 1)

        # 4. gate / evacuation -------------------------------------------------
        if not s.gate_open:
            # dead partners count as accounted for, so a lone survivor can still escape
            near = [bool(s.evacuated[i]) or bool(s.health[i] <= 0)
                    or np.abs(s.surv_pos[i] - s.exit_pos).sum() <= 1 for i in (0, 1)]
            if all(near):
                s.gate_open = True
        if s.gate_open:
            for i in (0, 1):
                if active[i] and (s.surv_pos[i] == s.exit_pos).all():
                    s.evacuated[i] = True
        active = s.active.copy()

        # 5. zombies -----------------------------------------------------------
        for sub in range(cfg.zombie_moves_per_step):
            self._move_zombies(s, rng, active, ev["zombie_moves"][sub])

        # 6. infection ---------------------------------------------------------
        self._spread_infection(s, rng)

        # 7. damage ------------------------------------------------------------
        zombie_alive_pos = s.zombie_pos[s.zombie_alive]
        for i in (0, 1):
            if not active[i]:
                continue
            r, c = s.surv_pos[i]
            dmg = int(s.infection[r, c])
            if len(zombie_alive_pos) and (np.abs(zombie_alive_pos - s.surv_pos[i]).sum(axis=1) == 1).any():
                dmg += 1
            dmg = min(dmg, int(s.health[i]))
            s.health[i] -= dmg
            ev["damage"][i] = dmg

        # 8. time + supply drop -----------------------------------------------
        s.t += 1
        if cfg.supply_interval > 0 and s.t % cfg.supply_interval == 0:
            self._supply_drop(s, rng)

        # 9. termination + reward ---------------------------------------------
        died = [bool(active[i] and s.health[i] <= 0) for i in (0, 1)]
        ev["deaths"] = died
        reward = (cfg.r_step + cfg.r_damage * sum(ev["damage"])
                  + cfg.r_kill * ev["kills"] + cfg.r_pickup * ev["pickups"])
        won = bool(s.evacuated.all())
        partial = False
        if cfg.end_on_first_death:
            lost = any(died)
            terminated = won or lost
            if lost:
                reward += cfg.r_lose
        else:
            reward += cfg.r_death * sum(died)
            nobody_left = not s.active.any()
            terminated = nobody_left
            lost = nobody_left and not s.evacuated.any()          # everyone died
            partial = nobody_left and not won and not lost        # one out, one dead
            if partial:
                reward += cfg.r_partial_win
        if won:
            reward += cfg.r_win
        truncated = (not terminated) and s.t >= cfg.max_steps
        s.done = terminated or truncated

        return (self._obs(), float(reward), terminated, truncated,
                self._info(ev, won=won, lost=lost, partial=partial))

    # ------------------------------------------------- snapshot for planners ---
    def get_state(self) -> EnvState:
        """Deep snapshot including RNG state. Restoring it replays identically."""
        snap = self.state.copy()
        snap.rng_state = copy.deepcopy(self._rng.bit_generator.state)
        return snap

    def set_state(self, snap: EnvState) -> None:
        self.state = snap.copy()
        self._rng = np.random.default_rng()
        if snap.rng_state is not None:
            self._rng.bit_generator.state = copy.deepcopy(snap.rng_state)

    def clone(self) -> "OutbreakEnv":
        """Independent env copy (for MPC rollouts in the true simulator)."""
        e = OutbreakEnv(self.config)
        e.episode_seed = self.episode_seed
        e.set_state(self.get_state())
        return e

    def reseed_dynamics(self, seed: int) -> None:
        """Replace the RNG (e.g. so MPC rollouts sample different futures)."""
        self._rng = np.random.default_rng(seed)

    # --------------------------------------------------------------- helpers ---
    def _obs(self) -> dict:
        s = self.state
        return {
            "grid": enc.grid_tensor(s),
            "scalars": enc.scalar_vector(s, self.config),
            "agent_0": enc.q_features(s, 0),
            "agent_1": enc.q_features(s, 1),
        }

    def _info(self, events, won=False, lost=False, partial=False) -> dict:
        return {"seed": self.episode_seed, "t": self.state.t, "won": won, "lost": lost, "partial": partial,
                "dead": [bool(d) for d in self.state.dead],
                "gate_open": self.state.gate_open, "events": events}

    @staticmethod
    def _parse_action(action) -> tuple[int, int]:
        if np.isscalar(action):
            return split_joint_action(int(action))
        a1, a2 = (int(x) for x in action)
        if not (0 <= a1 < N_ACTIONS and 0 <= a2 < N_ACTIONS):
            raise ValueError(f"invalid action {action}")
        return a1, a2

    @staticmethod
    def _walkable(s: EnvState, p) -> bool:
        h, w = s.walls.shape
        return 0 <= p[0] < h and 0 <= p[1] < w and not s.walls[p[0], p[1]]

    @staticmethod
    def _shot_target(s: EnvState, i: int, max_range: int) -> int | None:
        """Nearest alive zombie in a straight line within range, walls block.
        Ties: closer first, then direction order UP, DOWN, LEFT, RIGHT."""
        h, w = s.walls.shape
        best, best_d = None, max_range + 1
        for dr, dc in _NEIGH4:
            r, c = s.surv_pos[i]
            for d in range(1, max_range + 1):
                r, c = r + dr, c + dc
                if not (0 <= r < h and 0 <= c < w) or s.walls[r, c]:
                    break
                hit = np.flatnonzero(s.zombie_alive & (s.zombie_pos[:, 0] == r) & (s.zombie_pos[:, 1] == c))
                if len(hit):
                    if d < best_d:
                        best, best_d = int(hit[0]), d
                    break
        return best

    def _move_zombies(self, s: EnvState, rng, active, out_moves) -> None:
        """Each alive zombie (slot order): w.p. chase_prob greedy step toward nearest
        active survivor (Manhattan), else uniform over 5 moves. Blocked moves -> stay.
        Two RNG draws per zombie slot always, so RNG use doesn't depend on outcomes."""
        cfg = self.config
        surv = s.surv_pos[active]
        occupied = {tuple(p) for p in s.zombie_pos[s.zombie_alive]}
        surv_cells = {tuple(p) for p in surv}
        for z in range(cfg.max_zombies):
            u, rand_move = rng.random(), int(rng.integers(N_ZOMBIE_MOVES))
            if not s.zombie_alive[z] or len(surv) == 0:
                continue
            pos = s.zombie_pos[z]

            def ok(m):
                t = pos + MOVE_DELTAS[m]
                tt = tuple(t)
                return m == WAIT or (self._walkable(s, t) and tt not in occupied and tt not in surv_cells)

            if u < cfg.zombie_chase_prob:
                d = np.abs(surv - pos).sum(axis=1)
                tgt = surv[int(np.argmin(d))]
                move, best = WAIT, int(np.abs(tgt - pos).sum())
                for m in range(4):
                    if ok(m):
                        nd = int(np.abs(tgt - (pos + MOVE_DELTAS[m])).sum())
                        if nd < best:
                            move, best = m, nd
            else:
                move = rand_move if ok(rand_move) else WAIT

            if move != WAIT:
                occupied.discard(tuple(pos))
                s.zombie_pos[z] = pos + MOVE_DELTAS[move]
                occupied.add(tuple(s.zombie_pos[z]))
            out_moves[z] = move

    def _spread_infection(self, s: EnvState, rng) -> None:
        inf = s.infection
        p = self.config.infection_prob
        padded = np.pad(inf, 1).astype(np.int8)
        n_inf = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        prob = 1.0 - (1.0 - p) ** n_inf
        draw = rng.random(inf.shape)  # always drawn: fixed RNG consumption
        new = (draw < prob) & ~inf & ~s.walls
        new[s.exit_pos[0], s.exit_pos[1]] = False  # exit tile never gets infected
        s.infection = inf | new

    def _supply_drop(self, s: EnvState, rng) -> None:
        free = ~s.walls & ~s.infection & ~s.medkits & ~s.ammo_crates
        free[s.exit_pos[0], s.exit_pos[1]] = False
        for p in s.zombie_pos[s.zombie_alive]:
            free[p[0], p[1]] = False
        for i in (0, 1):
            if not s.evacuated[i]:  # also keeps drops off bodies
                free[s.surv_pos[i, 0], s.surv_pos[i, 1]] = False
        cells = np.argwhere(free)
        if len(cells) == 0:
            return
        r, c = cells[int(rng.integers(len(cells)))]
        if rng.random() < 0.5:
            s.medkits[r, c] = True
        else:
            s.ammo_crates[r, c] = True


def shot_target(s: EnvState, cfg: EnvConfig, i: int) -> int | None:
    """Zombie slot survivor i would hit if it shot now (None = miss). Ignores ammo.

    Used by step() and by the renderer's crosshair, so both always agree."""
    max_range = cfg.shoot_range
    j = 1 - i
    if cfg.cover_fire and s.active[j] and np.abs(s.surv_pos[j] - s.surv_pos[i]).sum() <= 2:
        max_range += 1
    return OutbreakEnv._shot_target(s, i, max_range)
