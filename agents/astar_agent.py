"""Agent 1 (Exp 1): decentralized A* baseline.

Requirements.md section 4: plans on the *current* grid only (no model of the
future), replans every step, infected tiles and tiles next to zombies are
expensive, one planner per survivor, and each survivor treats its partner as a
moving obstacle (no communication, no coordination).

Per survivor, each step:
  1. If a zombie is in its sights (straight line, in range, has ammo) -> SHOOT.
     This is a reactive rule on the current snapshot, not planning.
  2. Otherwise run A* from its tile to the goal set on a cost map:
        wall / zombie / partner tile   impassable
        infected tile                  + infection_cost
        tile next to a zombie          + zombie_adjacent_cost
     The goal is the exit. If the partner is standing on the (closed) exit, the goal
     becomes "any tile next to the exit": standing there opens the gate.
  3. If no path exists, retry without the partner as an obstacle, then WAIT.
  4. Anti-deadlock: if the survivor tried to move but did not move for
     `stuck_patience` steps (e.g. both survivors keep bumping into each other in a
     corridor), it takes one random legal move. This is local, so still no coordination.

Fairness: reads only the current env.state snapshot. Never steps, clones or
peeks at the RNG.
"""
from __future__ import annotations

import heapq
import itertools

import numpy as np

from env.config import MOVE_DELTAS, SHOOT, WAIT
from env.outbreak_env import shot_target
from env.state import EnvState

from .base import Agent

INF = float("inf")
_DIRS = MOVE_DELTAS[:4]  # UP, DOWN, LEFT, RIGHT -> action ids 0..3


def astar(start: tuple[int, int], goals: set[tuple[int, int]], cost: np.ndarray) -> list[tuple[int, int]] | None:
    """Cheapest 4-connected path from start to any goal; cost[r, c] = cost of *entering* (r, c).

    Returns the cells after `start` (empty list if start is a goal), or None if unreachable.
    The heuristic is Manhattan distance to the nearest goal, which is admissible because
    every step costs >= 1. Ties are broken deterministically (f, then h, then insertion order).
    """
    if start in goals:
        return []
    goals = {g for g in goals if np.isfinite(cost[g])}
    if not goals:
        return None
    h, w = cost.shape
    garr = np.array(list(goals))

    def heur(p):
        return int(np.min(np.abs(garr[:, 0] - p[0]) + np.abs(garr[:, 1] - p[1])))

    counter = itertools.count()
    g_best = {start: 0.0}
    parent: dict[tuple, tuple] = {}
    heap = [(heur(start), heur(start), next(counter), start)]
    closed = set()
    while heap:
        _, _, _, node = heapq.heappop(heap)
        if node in closed:
            continue
        if node in goals:
            path = [node]
            while path[-1] in parent and parent[path[-1]] != start:
                path.append(parent[path[-1]])
            return path[::-1]
        closed.add(node)
        for dr, dc in _DIRS:
            nb = (node[0] + dr, node[1] + dc)
            if not (0 <= nb[0] < h and 0 <= nb[1] < w) or nb in closed:
                continue
            step = cost[nb]
            if not np.isfinite(step):
                continue
            g = g_best[node] + step
            if g < g_best.get(nb, INF):
                g_best[nb] = g
                parent[nb] = node
                hn = heur(nb)
                heapq.heappush(heap, (g + hn, hn, next(counter), nb))
    return None


class AStarAgent(Agent):
    name = "astar"
    observes = "current snapshot (full grid), no future/model"

    def __init__(self, infection_cost: float = 6.0, zombie_adjacent_cost: float = 4.0,
                 shoot: bool = True, stuck_patience: int = 2, seed: int = 0):
        self.infection_cost = infection_cost
        self.zombie_adjacent_cost = zombie_adjacent_cost
        self.shoot = shoot
        self.stuck_patience = stuck_patience
        self._seed = seed
        self.paths: list[list[tuple[int, int]] | None] = [None, None]  # for visualisation
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng((self._seed, seed or 0))
        self._last_pos: list[tuple | None] = [None, None]
        self._last_action = [WAIT, WAIT]
        self._stuck = [0, 0]
        self.paths = [None, None]

    # ------------------------------------------------------------------ act ---
    def act(self, obs, env) -> tuple[int, int]:
        s: EnvState = env.state
        return self._act_one(s, env.config, 0), self._act_one(s, env.config, 1)

    def _act_one(self, s: EnvState, cfg, i: int) -> int:
        if not s.active[i]:
            self.paths[i] = None
            return WAIT
        pos = (int(s.surv_pos[i, 0]), int(s.surv_pos[i, 1]))

        # stuck bookkeeping (did my last move actually happen?)
        if self._last_action[i] < 4 and self._last_pos[i] == pos:
            self._stuck[i] += 1
        else:
            self._stuck[i] = 0
        self._last_pos[i] = pos

        if self.shoot and s.ammo[i] > 0 and shot_target(s, cfg, i) is not None:
            self.paths[i] = None
            return self._remember(i, SHOOT)

        if self._stuck[i] >= self.stuck_patience:
            self._stuck[i] = 0
            legal = [a for a, (dr, dc) in enumerate(_DIRS) if self._free(s, i, pos[0] + dr, pos[1] + dc)]
            if legal:
                self.paths[i] = None
                return self._remember(i, int(self.rng.choice(legal)))

        goals = self._goals(s, i)
        path = astar(pos, goals, self._cost_map(s, i, partner_blocks=True))
        if path is None:
            path = astar(pos, goals, self._cost_map(s, i, partner_blocks=False))
        self.paths[i] = path
        if not path:  # None (unreachable) or [] (already at goal)
            return self._remember(i, WAIT)
        nxt = path[0]
        action = _DIRS.index((nxt[0] - pos[0], nxt[1] - pos[1]))
        return self._remember(i, action)

    # ------------------------------------------------------------- helpers ---
    def _remember(self, i, a):
        self._last_action[i] = a
        return a

    def _goals(self, s: EnvState, i: int) -> set[tuple[int, int]]:
        ex = (int(s.exit_pos[0]), int(s.exit_pos[1]))
        j = 1 - i
        partner_on_exit = s.active[j] and tuple(s.surv_pos[j]) == ex
        if s.gate_open or not partner_on_exit:
            return {ex}
        h, w = s.walls.shape
        return {(ex[0] + dr, ex[1] + dc) for dr, dc in _DIRS
                if 0 <= ex[0] + dr < h and 0 <= ex[1] + dc < w and not s.walls[ex[0] + dr, ex[1] + dc]}

    def _cost_map(self, s: EnvState, i: int, partner_blocks: bool) -> np.ndarray:
        cost = np.ones(s.walls.shape, dtype=float)
        cost[s.infection] += self.infection_cost
        h, w = cost.shape
        for zr, zc in s.zombie_pos[s.zombie_alive]:
            for dr, dc in _DIRS:
                r, c = zr + dr, zc + dc
                if 0 <= r < h and 0 <= c < w:
                    cost[r, c] += self.zombie_adjacent_cost
        for zr, zc in s.zombie_pos[s.zombie_alive]:
            cost[zr, zc] = INF
        cost[s.walls] = INF
        j = 1 - i
        if partner_blocks and s.active[j]:
            cost[s.surv_pos[j, 0], s.surv_pos[j, 1]] = INF
        return cost

    @staticmethod
    def _free(s: EnvState, i: int, r: int, c: int) -> bool:
        h, w = s.walls.shape
        if not (0 <= r < h and 0 <= c < w) or s.walls[r, c]:
            return False
        if any((zp == (r, c)).all() for zp in s.zombie_pos[s.zombie_alive]):
            return False
        j = 1 - i
        return not (s.active[j] and tuple(s.surv_pos[j]) == (r, c))
