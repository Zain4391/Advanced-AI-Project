"""EnvState: the complete, copyable simulator state (pure NumPy)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EnvState:
    # static for an episode (shared, not copied, by EnvState.copy)
    walls: np.ndarray            # (H, W) bool
    exit_pos: np.ndarray         # (2,) int  (row, col)
    exit_dist: np.ndarray        # (H, W) int, BFS distance to exit through non-walls (INF if unreachable)
    # dynamic
    infection: np.ndarray        # (H, W) bool
    medkits: np.ndarray          # (H, W) bool
    ammo_crates: np.ndarray      # (H, W) bool
    surv_pos: np.ndarray         # (2, 2) int
    health: np.ndarray           # (2,) int
    ammo: np.ndarray             # (2,) int
    evacuated: np.ndarray        # (2,) bool -- evacuated survivors leave the board
    zombie_pos: np.ndarray       # (max_zombies, 2) int; dead slots keep their last position
    zombie_alive: np.ndarray     # (max_zombies,) bool
    gate_open: bool = False
    t: int = 0
    done: bool = False
    rng_state: dict | None = field(default=None, repr=False)  # filled by OutbreakEnv.get_state()

    @property
    def active(self) -> np.ndarray:
        """(2,) bool: survivors still playing (not evacuated, not dead)."""
        return ~self.evacuated & (self.health > 0)

    @property
    def dead(self) -> np.ndarray:
        return self.health <= 0

    def copy(self) -> "EnvState":
        return EnvState(
            walls=self.walls,
            exit_pos=self.exit_pos,
            exit_dist=self.exit_dist,
            infection=self.infection.copy(),
            medkits=self.medkits.copy(),
            ammo_crates=self.ammo_crates.copy(),
            surv_pos=self.surv_pos.copy(),
            health=self.health.copy(),
            ammo=self.ammo.copy(),
            evacuated=self.evacuated.copy(),
            zombie_pos=self.zombie_pos.copy(),
            zombie_alive=self.zombie_alive.copy(),
            gate_open=self.gate_open,
            t=self.t,
            done=self.done,
            rng_state=copy.deepcopy(self.rng_state),
        )

    @classmethod
    def empty(cls, size: int, max_zombies: int) -> "EnvState":
        """Blank open map, exit top-right, survivors bottom-left. Handy for rule tests."""
        from .map_gen import bfs_distances

        walls = np.zeros((size, size), bool)
        exit_pos = np.array([0, size - 1])
        return cls(
            walls=walls,
            exit_pos=exit_pos,
            exit_dist=bfs_distances(walls, exit_pos),
            infection=np.zeros((size, size), bool),
            medkits=np.zeros((size, size), bool),
            ammo_crates=np.zeros((size, size), bool),
            surv_pos=np.array([[size - 1, 0], [size - 1, 1]]),
            health=np.array([5, 5]),
            ammo=np.array([1, 1]),
            evacuated=np.zeros(2, bool),
            zombie_pos=np.zeros((max_zombies, 2), int),
            zombie_alive=np.zeros(max_zombies, bool),
        )
