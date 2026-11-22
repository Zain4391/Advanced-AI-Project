"""Frozen constants and the environment configuration.

Everything other modules need to agree on (action ids, channel order, scalar
order) lives here. Changing any of these breaks saved datasets and trained
models -- tell the whole team first (see docs/interfaces.md).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

# ---------------------------------------------------------------- actions ---
UP, DOWN, LEFT, RIGHT, WAIT, SHOOT = range(6)
N_ACTIONS = 6
N_JOINT_ACTIONS = N_ACTIONS * N_ACTIONS  # 36
ACTION_NAMES = ("up", "down", "left", "right", "wait", "shoot")

# (row, col) deltas for UP, DOWN, LEFT, RIGHT, STAY. Index 4 == stay.
# Zombie moves use the same 5-way index (0..4) -> categorical target for the world model.
MOVE_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
N_ZOMBIE_MOVES = 5


def joint_action(a1: int, a2: int) -> int:
    """(a1, a2) -> 0..35."""
    return a1 * N_ACTIONS + a2


def split_joint_action(j: int) -> tuple[int, int]:
    """0..35 -> (a1, a2)."""
    return divmod(int(j), N_ACTIONS)


# ----------------------------------------------------------- grid channels ---
CH_WALL, CH_INFECTION, CH_ZOMBIE, CH_MEDKIT, CH_AMMO, CH_SURV1, CH_SURV2, CH_EXIT = range(8)
CHANNEL_NAMES = ("wall", "infection", "zombie", "medkit", "ammo", "survivor1", "survivor2", "exit")
N_CHANNELS = 8

# ---------------------------------------------------------- scalar vector ---
SCALAR_NAMES = ("hp1", "ammo1", "hp2", "ammo2", "evac1", "evac2", "gate_open", "t_frac")
N_SCALARS = len(SCALAR_NAMES)


# ------------------------------------------------------------------ config ---
@dataclass(frozen=True)
class EnvConfig:
    # map
    grid_size: int = 8
    wall_density: float = 0.15
    # survivors
    max_health: int = 5
    max_ammo: int = 3
    start_health: int = 5
    start_ammo: int = 1
    # zombies (scripted NPCs)
    n_zombies: int = 3
    max_zombies: int = 6            # fixed slot count for the world model
    zombie_chase_prob: float = 0.7
    zombie_moves_per_step: int = 1  # 2 == "faster zombies"
    zombie_min_start_dist: int = 3
    # infection
    infection_prob: float = 0.05    # p: per infected neighbour, per step
    n_initial_infection: int = 1
    # items
    n_medkits: int = 1
    n_ammo_crates: int = 1
    supply_interval: int = 10       # K; 0 disables supply drops
    medkit_heal: int = 2
    # combat
    shoot_range: int = 2
    cover_fire: bool = False        # optional rule: +1 range if partner within 2 tiles
    # cooperation rules (flip off for the no-coordination ablation)
    coop_medkit: bool = True        # medkit heals both if partner adjacent on pickup
    coop_gate: bool = True          # gate opens only when both within 1 tile of the exit
    # episode
    max_steps: int = 100
    # False (default): a dead survivor stays down and the partner plays on; the episode
    #   ends when nobody is left on the board (all evacuated or dead).
    # True: original spec -- the episode is lost as soon as either survivor dies.
    end_on_first_death: bool = False
    # rewards (shared team reward)
    r_win: float = 20.0
    r_lose: float = -20.0           # end_on_first_death=True only: first death
    r_death: float = -10.0          # end_on_first_death=False: per survivor death
    r_partial_win: float = 10.0     # one survivor evacuated, the other died
    r_step: float = -0.1
    r_damage: float = -1.0          # per health point lost, any survivor
    r_kill: float = 2.0
    r_pickup: float = 1.0

    def with_(self, **kw) -> "EnvConfig":
        return replace(self, **kw)


DEFAULT_CONFIG = EnvConfig()

# Exp 6 shifted conditions -- evaluated without retraining, on experiments.seeds.SHIFTED.
SHIFTED_CONFIGS: dict[str, EnvConfig] = {
    "unseen_maps": DEFAULT_CONFIG,
    "infection_2p": DEFAULT_CONFIG.with_(infection_prob=DEFAULT_CONFIG.infection_prob * 2),
    "more_zombies": DEFAULT_CONFIG.with_(n_zombies=5),
    "faster_zombies": DEFAULT_CONFIG.with_(zombie_moves_per_step=2),
    "fewer_supplies": DEFAULT_CONFIG.with_(supply_interval=25, n_medkits=0, n_ammo_crates=0),
    "dense_walls": DEFAULT_CONFIG.with_(wall_density=0.25),
}

NO_COORD_CONFIG = DEFAULT_CONFIG.with_(coop_medkit=False, coop_gate=False)
