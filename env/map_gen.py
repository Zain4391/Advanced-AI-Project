"""Seeded procedural map generation. Pure NumPy."""
from __future__ import annotations

from collections import deque

import numpy as np

from .config import EnvConfig
from .state import EnvState

INF = 10_000


def bfs_distances(walls: np.ndarray, src) -> np.ndarray:
    """4-connected BFS distance from src over non-wall tiles (INF where unreachable)."""
    h, w = walls.shape
    dist = np.full((h, w), INF, dtype=np.int32)
    r0, c0 = int(src[0]), int(src[1])
    if walls[r0, c0]:
        return dist
    dist[r0, c0] = 0
    q = deque([(r0, c0)])
    while q:
        r, c = q.popleft()
        d = dist[r, c] + 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not walls[nr, nc] and dist[nr, nc] > d:
                dist[nr, nc] = d
                q.append((nr, nc))
    return dist


def _manhattan_grid(n: int, p) -> np.ndarray:
    rr, cc = np.mgrid[0:n, 0:n]
    return np.abs(rr - p[0]) + np.abs(cc - p[1])


def generate_map(cfg: EnvConfig, rng: np.random.Generator, max_attempts: int = 200) -> EnvState:
    """Build the initial state. All randomness comes from `rng`.

    Guarantees: both survivors can reach the exit at step 0 (walls only), and
    zombies / initial infection start at least `zombie_min_start_dist` from survivors.
    """
    n = cfg.grid_size
    corners = np.array([[0, 0], [0, n - 1], [n - 1, 0], [n - 1, n - 1]])

    for _ in range(max_attempts):
        k = int(rng.integers(4))
        exit_pos = corners[k]
        s1 = corners[3 - k]                      # opposite corner
        s2 = s1 + np.array([0, 1 if s1[1] == 0 else -1])  # next to s1 along the row

        walls = rng.random((n, n)) < cfg.wall_density
        for p in (exit_pos, s1, s2):
            walls[p[0], p[1]] = False

        exit_dist = bfs_distances(walls, exit_pos)
        if exit_dist[s1[0], s1[1]] >= INF or exit_dist[s2[0], s2[1]] >= INF:
            continue

        free = (~walls) & (exit_dist < INF)      # reachable floor only
        for p in (exit_pos, s1, s2):
            free[p[0], p[1]] = False
        far = (_manhattan_grid(n, s1) >= cfg.zombie_min_start_dist) & (
            _manhattan_grid(n, s2) >= cfg.zombie_min_start_dist
        )

        # zombies
        cand = np.argwhere(free & far)
        if len(cand) < cfg.n_zombies + cfg.n_initial_infection:
            continue
        zi = rng.choice(len(cand), size=cfg.n_zombies, replace=False)
        zombie_pos = np.zeros((cfg.max_zombies, 2), dtype=int)
        zombie_alive = np.zeros(cfg.max_zombies, dtype=bool)
        zombie_pos[: cfg.n_zombies] = cand[zi]
        zombie_alive[: cfg.n_zombies] = True
        occupied = np.zeros((n, n), bool)
        occupied[cand[zi, 0], cand[zi, 1]] = True

        # initial infection: far from survivors and not on/next to the exit
        infection = np.zeros((n, n), bool)
        inf_cand = np.argwhere(free & far & ~occupied & (_manhattan_grid(n, exit_pos) >= 3))
        if len(inf_cand) < cfg.n_initial_infection:
            continue
        ii = rng.choice(len(inf_cand), size=cfg.n_initial_infection, replace=False)
        infection[inf_cand[ii, 0], inf_cand[ii, 1]] = True

        # items on clean, unoccupied floor
        medkits = np.zeros((n, n), bool)
        ammo_crates = np.zeros((n, n), bool)
        item_cand = np.argwhere(free & ~occupied & ~infection)
        n_items = cfg.n_medkits + cfg.n_ammo_crates
        if n_items:
            if len(item_cand) < n_items:
                continue
            idx = rng.choice(len(item_cand), size=n_items, replace=False)
            for j, cell in enumerate(item_cand[idx]):
                (medkits if j < cfg.n_medkits else ammo_crates)[cell[0], cell[1]] = True

        return EnvState(
            walls=walls,
            exit_pos=exit_pos.copy(),
            exit_dist=exit_dist,
            infection=infection,
            medkits=medkits,
            ammo_crates=ammo_crates,
            surv_pos=np.stack([s1, s2]).astype(int),
            health=np.full(2, cfg.start_health, dtype=int),
            ammo=np.full(2, cfg.start_ammo, dtype=int),
            evacuated=np.zeros(2, bool),
            zombie_pos=zombie_pos,
            zombie_alive=zombie_alive,
            gate_open=not cfg.coop_gate,
        )

    raise RuntimeError(f"map generation failed after {max_attempts} attempts; check config {cfg}")
