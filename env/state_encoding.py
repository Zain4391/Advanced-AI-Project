"""The three state representations (Requirements.md, Section 3).

1. Full state          -> grid_tensor() + scalar_vector()   (true env, planners)
2. Q-learning features -> q_features(state, i)               (egocentric, per survivor)
3. World-model input   -> world_model_input(state, a1, a2)   (grid + scalars + one-hot joint action)

All agents MUST go through this module so the fairness rule is auditable in one place.
"""
from __future__ import annotations

import numpy as np

from .config import (
    CH_AMMO, CH_EXIT, CH_INFECTION, CH_MEDKIT, CH_SURV1, CH_SURV2, CH_WALL, CH_ZOMBIE,
    MOVE_DELTAS, N_CHANNELS, N_JOINT_ACTIONS, N_SCALARS, WAIT, EnvConfig, joint_action,
)
from .state import EnvState

CLIP = 3
NONE_OFFSET = CLIP + 1  # sentinel offset when the target does not exist (no zombie / partner gone)


# ------------------------------------------------------------- full state ---
def grid_tensor(s: EnvState) -> np.ndarray:
    """(N_CHANNELS, H, W) float32 in {0, 1}. Evacuated and dead survivors are not drawn
    (their status is in the scalars: evac flag / hp == 0)."""
    h, w = s.walls.shape
    g = np.zeros((N_CHANNELS, h, w), dtype=np.float32)
    g[CH_WALL] = s.walls
    g[CH_INFECTION] = s.infection
    zp = s.zombie_pos[s.zombie_alive]
    g[CH_ZOMBIE, zp[:, 0], zp[:, 1]] = 1.0
    g[CH_MEDKIT] = s.medkits
    g[CH_AMMO] = s.ammo_crates
    for i, ch in ((0, CH_SURV1), (1, CH_SURV2)):
        if s.active[i]:
            g[ch, s.surv_pos[i, 0], s.surv_pos[i, 1]] = 1.0
    g[CH_EXIT, s.exit_pos[0], s.exit_pos[1]] = 1.0
    return g


def scalar_vector(s: EnvState, cfg: EnvConfig) -> np.ndarray:
    """(N_SCALARS,) float32, order = config.SCALAR_NAMES. Raw counts (not normalised) except t_frac."""
    v = np.array(
        [s.health[0], s.ammo[0], s.health[1], s.ammo[1],
         s.evacuated[0], s.evacuated[1], s.gate_open, s.t / cfg.max_steps],
        dtype=np.float32,
    )
    assert v.shape == (N_SCALARS,)
    return v


def zombie_array(s: EnvState) -> np.ndarray:
    """(max_zombies, 3) int: row, col, alive. Slot identity is stable across steps."""
    return np.concatenate([s.zombie_pos, s.zombie_alive[:, None].astype(int)], axis=1)


# --------------------------------------------------------- Q-learning view ---
def _exit_hint(s: EnvState, i: int) -> int:
    """Wall-aware direction (0..3) that decreases BFS distance to exit; 4 if on the exit / evacuated."""
    if not s.active[i]:
        return WAIT
    r, c = s.surv_pos[i]
    h, w = s.walls.shape
    best, best_d = WAIT, s.exit_dist[r, c]
    for a, (dr, dc) in enumerate(MOVE_DELTAS[:4]):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and s.exit_dist[nr, nc] < best_d:
            best, best_d = a, s.exit_dist[nr, nc]
    return best


def _clip_offset(d: np.ndarray) -> tuple[int, int]:
    return int(np.clip(d[0], -CLIP, CLIP)), int(np.clip(d[1], -CLIP, CLIP))


def q_features(s: EnvState, i: int) -> tuple:
    """Hashable egocentric features for survivor i (dict Q-table key).

    (exit_hint, inf_U, inf_D, inf_L, inf_R, wall_U, wall_D, wall_L, wall_R,
     zombie_dr, zombie_dc, own_hp, own_ammo, partner_dr, partner_dc, partner_hp)
    """
    h, w = s.walls.shape
    r, c = s.surv_pos[i]
    inf_flags, wall_flags = [], []
    for dr, dc in MOVE_DELTAS[:4]:
        nr, nc = r + dr, c + dc
        inside = 0 <= nr < h and 0 <= nc < w
        wall_flags.append(int((not inside) or bool(s.walls[nr, nc])))
        inf_flags.append(int(inside and bool(s.infection[nr, nc])))

    alive = s.zombie_pos[s.zombie_alive]
    if len(alive):
        d = alive - s.surv_pos[i]
        zdr, zdc = _clip_offset(d[np.argmin(np.abs(d).sum(axis=1))])
    else:
        zdr = zdc = NONE_OFFSET

    j = 1 - i
    if not s.active[j]:
        pdr = pdc = NONE_OFFSET
    else:
        pdr, pdc = _clip_offset(s.surv_pos[j] - s.surv_pos[i])

    return (
        _exit_hint(s, i), *inf_flags, *wall_flags,
        zdr, zdc, int(s.health[i]), int(s.ammo[i]),
        pdr, pdc, int(s.health[j]),
    )


# -------------------------------------------------------- world-model view ---
def one_hot_joint_action(a1: int, a2: int) -> np.ndarray:
    v = np.zeros(N_JOINT_ACTIONS, dtype=np.float32)
    v[joint_action(a1, a2)] = 1.0
    return v


def world_model_input(s: EnvState, cfg: EnvConfig, a1: int, a2: int) -> dict[str, np.ndarray]:
    return {
        "grid": grid_tensor(s),
        "scalars": scalar_vector(s, cfg),
        "zombies": zombie_array(s),
        "action": one_hot_joint_action(a1, a2),
    }
