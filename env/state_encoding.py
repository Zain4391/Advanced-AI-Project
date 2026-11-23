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


# ------------------------------------------------- compact Q-learning view ---
def _time_bucket(s: EnvState, cfg: EnvConfig) -> int:
    """Coarse time pressure (0: first quarter, 1: middle, 2: last 40% of the step limit). The
    infection keeps spreading, so the right trade-off between safety and speed changes with time."""
    f = s.t / max(1, cfg.max_steps)
    return 0 if f < 0.25 else 1 if f < 0.6 else 2


def q_features_compact(s: EnvState, i: int, cfg: EnvConfig, extended: bool = False) -> tuple:
    """Coarse egocentric features for survivor i: the Requirements.md section 14 fallback
    ("Q-learning does not learn -> simplify features").

    The spec features (q_features) have ~7.5e8 possible values, too many for a table to
    revisit. This keeps what drives decisions and drops exact offsets:

    (exit_hint,                         0..4  wall-aware direction to exit (4 = on exit / gone)
     nbr_U, nbr_D, nbr_L, nbr_R,        0 free, 1 hazard (infected or next to a zombie), 2 blocked
     here_hazard,                       0/1   standing on infection or next to a zombie now
     can_shoot,                         0/1   has ammo and a zombie is in its sights
     zombie_dr, zombie_dc,              -1/0/1 sign of offset to nearest zombie if within 3, else 0, 0
     hp_low,                            0/1   own health <= 2
     near_exit,                         0/1   within 1 tile of the exit
     partner,                           0 far from exit, 1 next to/on exit, 2 evacuated or dead
     gate_open)                         0/1
    """
    from .outbreak_env import shot_target  # local import: outbreak_env imports this module

    j = 1 - i
    if not s.active[i]:
        # Absorbing state for a survivor that has left the game. It still receives the shared
        # team reward (so the +20 win reaches the first survivor to evacuate), so the key says
        # why it is inactive (here_hazard slot: 2 = evacuated, 3 = dead) and what its partner is doing.
        if not s.active[j]:
            partner_ = 2
        else:
            partner_ = int(np.abs(s.surv_pos[j] - s.exit_pos).sum() <= 1)
        base_ = (WAIT, 2, 2, 2, 2, 3 if s.health[i] <= 0 else 2, 0, 0, 0, 0, 0, partner_,
                 int(s.gate_open))
        return base_ + (0, _time_bucket(s, cfg)) if extended else base_
    h, w = s.walls.shape
    r, c = (int(v) for v in s.surv_pos[i])
    zp = s.zombie_pos[s.zombie_alive]
    zombie_cells = {(int(a), int(b)) for a, b in zp}
    near_zombie = {(a + dr, b + dc) for a, b in zombie_cells for dr, dc in MOVE_DELTAS[:4]}
    partner_cell = (int(s.surv_pos[j, 0]), int(s.surv_pos[j, 1])) if s.active[j] else None

    nbrs = []
    for dr, dc in MOVE_DELTAS[:4]:
        cell = (r + dr, c + dc)
        if (not (0 <= cell[0] < h and 0 <= cell[1] < w) or s.walls[cell]
                or cell in zombie_cells or cell == partner_cell):
            nbrs.append(2)
        elif s.infection[cell] or cell in near_zombie:
            nbrs.append(1)
        else:
            nbrs.append(0)
    here_hazard = int(bool(s.infection[r, c]) or (r, c) in near_zombie)
    can_shoot = int(s.ammo[i] > 0 and shot_target(s, cfg, i) is not None)

    zdr = zdc = 0
    if len(zp):
        d = zp - s.surv_pos[i]
        k = int(np.argmin(np.abs(d).sum(axis=1)))
        if np.abs(d[k]).sum() <= 3:
            zdr, zdc = int(np.sign(d[k, 0])), int(np.sign(d[k, 1]))

    ex = s.exit_pos
    near_exit = int(abs(r - ex[0]) + abs(c - ex[1]) <= 1)
    if not s.active[j]:
        partner = 2
    else:
        partner = int(np.abs(s.surv_pos[j] - ex).sum() <= 1)
    base = (_exit_hint(s, i), *nbrs, here_hazard, can_shoot, zdr, zdc,
            int(s.health[i] <= 2), near_exit, partner, int(s.gate_open))
    if not extended:
        return base
    hp = int(s.health[i])
    return base + (0 if hp <= 2 else 1 if hp <= 4 else 2, _time_bucket(s, cfg))
