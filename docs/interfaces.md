# Frozen Interfaces (Day 1)

These contracts are shared by all three team members. **Do not change any of them without telling the team.** Changes break saved datasets, Q-tables and model checkpoints. Constants live in `env/config.py`; this file explains them.

Status: _proposed by the scaffold, team to confirm on Day 1._ Mark each section ✅ once agreed.

---

## 1. Environment API

```python
from env import OutbreakEnv, EnvConfig
env = OutbreakEnv(EnvConfig())                  # frozen dataclass; use cfg.with_(...) for variants
obs, info = env.reset(seed=123)                 # seed fixes map, zombie starts AND all random events
obs, reward, terminated, truncated, info = env.step((a1, a2))   # or env.step(joint_idx)

snap = env.get_state()      # deep copy incl. RNG state
env.set_state(snap)         # restores exactly; replay is identical
sim = env.clone()           # independent env (for true-dynamics MPC)
sim.reseed_dynamics(k)      # different sampled future from the same state
```

- `reward` is a single shared team reward (float).
- `info = {seed, t, won, lost, gate_open, events}`, where `events = {kills, pickups, damage[2], shots[2], shot_targets[2] (zombie slot hit, -1 = miss), zombie_moves[(moves_per_step, max_zombies)]}`. `env.outbreak_env.shot_target(state, cfg, i)` returns what survivor i would hit right now.
- `info["events"]["zombie_moves"]` holds the realized 0..4 move per zombie slot. **This is the world-model training target.**

## 2. Actions

| id | 0 | 1 | 2 | 3 | 4 | 5 |
|----|---|---|---|---|---|---|
| action | UP | DOWN | LEFT | RIGHT | WAIT | SHOOT |

Joint action index: `a1 * 6 + a2` (0..35), via `joint_action()` / `split_joint_action()`.
Zombie moves use 0..4 = UP, DOWN, LEFT, RIGHT, STAY.
Coordinates are `(row, col)` with row 0 at the top.

## 3. Observation / state representations (`env/state_encoding.py`)

| Key | Shape | Content |
|-----|-------|---------|
| `obs["grid"]` | `(8, H, W)` float32 | channels: `wall, infection, zombie, medkit, ammo, survivor1, survivor2, exit` |
| `obs["scalars"]` | `(8,)` float32 | `hp1, ammo1, hp2, ammo2, evac1, evac2, gate_open, t/max_steps` |
| `obs["agent_0"]`, `obs["agent_1"]` | 16-tuple of int | Q-learning features (below) |
| `zombie_array(state)` | `(max_zombies, 3)` int | `row, col, alive`; **slot identity is stable** |

Q-features for survivor i:
`(exit_hint, infU, infD, infL, infR, wallU, wallD, wallL, wallR, zombie_dr, zombie_dc, own_hp, own_ammo, partner_dr, partner_dc, partner_hp)`.
The offsets are clipped to ±3. A value of 4 means none (no zombie, or the partner has evacuated). `exit_hint` is a BFS-based direction 0..3, or 4 when on the exit or evacuated.

**Compact Q-features (`state_encoding.q_features_compact`, the Q-learning default).** This is the Requirements §14 fallback, "Q-learning does not learn → simplify features". The spec set above has about 7.5×10⁸ possible values: after 10k training episodes, 25–30% of greedy decisions landed in never-seen states, and the learners did not improve. The compact set is a 13-tuple:
`(exit_hint, nbr_U, nbr_D, nbr_L, nbr_R, here_hazard, can_shoot, zombie_dr_sign, zombie_dc_sign, hp_low, near_exit, partner, gate_open)`
- `nbr_*`: 0 = free, 1 = hazard (infected or next to a zombie), 2 = blocked (wall, edge, zombie or partner).
- Zombie direction is the sign of the offset to the nearest zombie, only if it is within 3 tiles (otherwise 0, 0).
- `partner`: 0 = far from the exit, 1 = within 1 tile of it, 2 = evacuated or dead.

It still contains no partner *action* or plan, so the learners stay independent. Choose with `QLearningAgent(features="spec"|"compact"|"compact_t")` or `train_qlearning.py --features`.

**Extended compact set (`q_features_compact(..., extended=True)`, feature set `compact_t`, the training default).** Appends two slots to the 13-tuple: `hp_bucket` (0: hp ≤ 2, 1: hp 3-4, 2: hp 5) and `time_bucket` (0: first 25% of `max_steps`, 1: up to 60%, 2: the rest). The infection keeps spreading, so the best trade-off between safety and speed changes over the episode; without a time signal the learners dawdle around hazards until the infection covers the map.

**Absorbing key for inactive survivors.** A survivor that has evacuated or died keeps its own Q-table entries so that the shared team reward (the +20 win, the +10 partial win) reaches it too. For an inactive survivor `q_features_compact` returns `(WAIT, 2, 2, 2, 2, status, 0, 0, 0, 0, 0, partner, gate_open)`, where `status` sits in the `here_hazard` slot (2 = evacuated, 3 = dead). Its only legal action is WAIT, so its bootstrap value is `Q(s', WAIT)`.

**Training-only helpers (not part of the evaluation contract).** Exit-distance potential shaping (`--shaping`), `SHOOT` masking when `can_shoot = 0`, a head start for the exit-hint action in new Q rows (`--hint-prior`), per-(state, action) learning-rate decay (`--alpha-decay`) and the easy-to-hard `--curriculum`. Evaluation always uses the unshaped env reward and the greedy policy, so results stay comparable with A\* and the planners.

**Fairness rule**

| Agent | Sees |
|---|---|
| A\* | current snapshot (full grid, no future) |
| Q-learning | only its own egocentric features: `obs["agent_i"]` (spec set) or `q_features_compact(state, i, cfg)` (compact set, default) |
| MPC (all variants) | full state + a dynamics model (true sim or learned) |

## 4. Step order (resolution within one `step`)

1. **Shots:** fired from start-of-step positions. Each shot needs ammo ≥ 1 and costs 1 ammo. It hits the nearest alive zombie in a straight line within `shoot_range`. Walls block shots. Ties go to the closer zombie, then direction order U, D, L, R. Shooting with 0 ammo acts as WAIT.
2. **Movement:** simultaneous. A move is cancelled if it hits a wall, the edge, a zombie, or the partner's start tile (so swaps are impossible). If both survivors target the same tile, both stay. The one exception: both may step onto an *open* exit together.
3. **Pickups:** a medkit gives +2 hp, capped at 5. With `coop_medkit`, a partner within Manhattan distance 1 is healed too. An ammo crate gives +1 ammo, capped at 3. Each pickup gives +1 reward.
4. **Gate:** with `coop_gate`, the gate opens (and stays open) once both survivors are within Manhattan distance 1 of the exit, or already evacuated. An active survivor on the open exit **evacuates**: they leave the board, stop acting and take no more damage. With `coop_gate=False` the gate is open from the start.
5. **Zombies:** move in slot order. With probability `zombie_chase_prob`, a zombie takes the greedy Manhattan step toward the nearest active survivor. Otherwise it picks uniformly from the 5 moves. Blocked moves become STAY. Zombies never move onto walls, other zombies or survivors. Each zombie slot always uses two RNG draws, so RNG consumption doesn't depend on outcomes. Repeated `zombie_moves_per_step` times per step.
6. **Infection:** each clean, non-wall tile becomes infected with probability `1 - (1-p)^(number of infected 4-neighbours)`. **The exit tile is never infected.**
7. **Damage** (active survivors only): −1 for standing on an infected tile, and −1 if any zombie is 4-adjacent. Each health point lost costs −1 reward.
8. **t += 1.** If `t % K == 0`, one supply drop (medkit or ammo, 50/50) appears on a random clean, empty tile.
9. **Deaths and end of episode** (default, `end_on_first_death=False`): a survivor reaching hp 0 is **dead**. They stay on the map as a body, can't act, take no damage, are ignored by zombies, don't block movement, and count as "accounted for" at the gate. Each death costs −10. The episode terminates when no survivor is active:
   - both evacuated → **win** (+20, `info["won"]`)
   - one evacuated, one dead → **partial** (+10, `info["partial"]`)
   - both dead → **lost** (`info["lost"]`)

   Otherwise `t >= max_steps` → truncated. With `end_on_first_death=True` (the original spec), the first death ends the episode as lost (−20).
   Every step costs −0.1, and each kill gives +2. `info["dead"]` is a 2-list of bools, and `events["deaths"]` marks deaths that happened this step. `state.active` = not evacuated and not dead.

Map generation: the exit is in a random corner and the survivors start in the opposite corner, next to each other. A BFS check guarantees both can reach the exit through walls at step 0. Zombies and the initial infection start at least 3 tiles from the survivors, and the initial infection starts at least 3 tiles from the exit.

## 5. Agent API (`agents/base.py`)

```python
class Agent:
    def reset(self, seed=None): ...
    def act(self, obs, env) -> tuple[int, int]: ...            # always a JOINT action
    def observe(self, obs, action, reward, next_obs, terminated, truncated): ...  # learners only
```
Decentralized agents wrap two per-survivor policies in one `Agent`. Only planners may call `env.get_state()` or `env.clone()`.

## 6. Dataset format (`world_model/dataset.py`, written by `scripts/collect_data.py`)

One compressed `.npz` per collection run, stored in `data/` (gitignored; share via Drive or Kaggle). N = number of transitions.

| array | shape | dtype |
|---|---|---|
| `grid`, `next_grid` | `(N, 8, H, W)` | uint8 |
| `scalars`, `next_scalars` | `(N, 8)` | float32 |
| `zombies`, `next_zombies` | `(N, max_zombies, 3)` | int8 |
| `action` | `(N, 2)` | int8 |
| `zombie_moves` | `(N, moves_per_step, max_zombies)` | int8 |
| `reward` | `(N,)` | float32 |
| `terminated`, `truncated` | `(N,)` | bool (kept separate) |
| `episode_seed`, `t` | `(N,)` | int32 |
| `policy` | `(N,)` | str: `random` / `qlearning` / `astar` / `mpc` |

## 7. World-model I/O

- **Input:** `grid` + `scalars` + `zombies` + one-hot joint action (36).
- **Outputs:** per-cell infection probability; a 5-way categorical per zombie slot (dead slots masked out of the loss); both survivors' positions, health and ammo; reward; done. Walls and the exit are copied through.
- **Supply drops:** these are random and hard to predict, so they are excluded from the loss. The model copies the item maps forward minus picked-up items. This simplification is stated in the report.

## 8. Results CSV (`experiments/harness.py`)

One row per episode with the columns:
`seed, agent, return, success, steps, lost, truncated, kills, damage, pickups, mean_partner_dist, gate_open_step, ms_per_decision`, plus any extra keys passed to `evaluate(...)` (e.g. `run_seed`, `condition`, `lambda`, `H`, `N`).
File naming: `results/exp{N}_{agent}_{condition}.csv`.
