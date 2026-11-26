# Progress Log

Newest entry at the top. At the end of every work session (human or Claude), add an entry: what changed, what's next, and any open problems. Keep entries short.

---

## Status board

| Component | Owner | Status |
|---|---|---|
| Env + rule tests | P1 | ✅ scaffold done; 34 tests pass; rules await team sign-off |
| Pygame play (`scripts/play.py`) | P1 | ✅ basic version |
| A\* agent | P1 | ✅ `agents/astar_agent.py`; val: 46% success / 40% partial / 14% lost |
| Eval harness + seeds | P2 | ✅ `harness.py`, `seeds.py`, `run_benchmark.py` (CLI → CSV) |
| Q-learning (2 independent learners) | P2 | 🟡 learns the full game: 12-18% both escape, 33-44% one escapes (before the fix: 0%); A\* gets 46%. See the second 2026-11-24 entry |
| Plots / tables | P2 | 🟡 `analysis/plots.py curve` (learning curve) |
| Data collection | P3 | ⬜ |
| World model (CNN + ensemble) | P3 | ⬜ |
| MPC: true dynamics / learned / uncertainty | P3 | ✅ true dynamics (Exp 3): test 36% / 34% / 22% (8% truncated), A\* 50% / 45% / 5%. Learned / uncertainty ⬜ |
| Side-by-side demo viewer | P1 | ⬜ |
| Report / slides | All | ⬜ |

## Open decisions
- [ ] Team sign-off on `docs/interfaces.md` (step order, gate/evacuation rule, supply drops excluded from WM loss)
- [ ] Grid 8x8 (current default) vs 10x10
- [ ] Difficulty tuning: the random agent dies in ~12 steps (0% success). Check with A\*; if A\* success is below ~50%, consider `n_zombies=2` or `zombie_chase_prob=0.5`.
- [ ] Confirm the STORM-reproduction framing with the instructor

## Known performance notes
- True simulator: ~3.4k steps/s, so MPC with N=200, H=6 takes about 350 ms per decision on CPU. Optimize later if needed (vectorize zombie moves, or roll out N states as a batch).

---

## 2026-11-27: Exp 3 final test run
**Chosen config:** N=200, H=8, iters=1, p_hint=0.5, gamma=0.98, n_futures=1, leaf on, leaf_dist_weight=0.5 (spec), lam=0. Best val success (44%, tied with p_hint 0.7). Remaining sweep configs (N 400, iters=3) were skipped after the diagnosis below showed the score, not the search, is the bottleneck.

**Test (100 maps, `results/exp3_mpc_default_test.csv`, `--workers 4`):**

| Agent | Success | Partial | Lost | Truncated | Return | Steps | ms/decision |
|---|---|---|---|---|---|---|---|
| MPC (true dynamics) | 36% | 34% | 22% | 8% | -1.27 ± 20.18 | 40.4 | 355 |
| A\* | 50% | 45% | 5% | 0% | 7.72 ± 14.71 | 19.4 | 0.5 |
| Q-learning (seed 0) | 12% | 41% | 47% | 0% | -13.29 | 31.8 | 0.1 |

- Paired over the same maps, MPC trails A\* by 14 ± 6 points (std. error), so on test the gap is real; on val it was 2 points. MPC wins 14 maps A\* does not, A\* wins 28 that MPC does not.
- Same pattern as val: on the 22 maps both win, MPC takes 30.1 steps vs 19.4.
- 355 ms per decision with 4 workers, under the 0.5 s target.

**Next:** Exp 4 (data collection, world model) reuses this planner and config with a learned `Dynamics` backend; compare against this row. Open: the leaf loophole fix (dead survivor counts with a fixed distance) is a team decision; if adopted, rerun Exp 3 val + test so Exp 4/5 compare against the same score. Shifted configs (Exp 6) not run yet. Nothing committed.

---

## 2026-11-25: Exp 3 true-dynamics MPC built; val sweep paused (resume here)
**Code (nothing committed yet):**
- `agents/mpc_planner.py`: `RolloutBatch`, the `Dynamics` protocol, `TrueSimDynamics`, `score_rollouts` / `leaf_value`, the categorical sampler with CEM refit (`iters` > 1), and `MPCAgent` (registered as `"mpc"`). Identical sequences are rolled out once (under common random numbers they give identical results). `agent.paths` shows the best plan in `watch.py`. Contract in `docs/interfaces.md` §7b.
- `env/outbreak_env.py`: `step(action, build_obs=False)` skips obs and info (they took about 38% of step time). No rule change; equivalence test in `tests/test_rules.py`.
- `env/state_encoding.exit_hint` (public alias of `_exit_hint`).
- `experiments/run_benchmark.py`: `--agent-kwargs "k=v,..."` and `--workers K` (spawn processes, rows sorted by seed, identical to K=1). Adds `workers` plus the agent's `csv_extra` columns (`N, H, iters, lam, p_hint, n_futures, leaf, dynamics`).
- `experiments/time_mpc.py` (ms per decision, N × H grid) and `experiments/diagnose_failures.py` (attributes each death to infection or zombies, counts truncations and gate waits). `scripts/watch.py` takes `--agent-kwargs`.
- `tests/test_mpc.py`: 14 tests. The full suite passes (80 tests).

**ms per decision (single process, 12 fixed states):**

| N \ H | 6 | 8 | 12 |
|---|---|---|---|
| 100 | 142 | 170 | 229 |
| 200 | 239 | 321 | 617 |
| 400 | 576 | 774 | 916 |

What's left is the rule code itself (zombie moves, infection spread), not the planner.

**Val results so far (100 maps):**

| Config | Success | Partial | Lost | Truncated | ms/decision |
|---|---|---|---|---|---|
| MPC base (N=200, H=8, iters=1, p_hint=0.5, leaf on) | 44% | 30% | 19% | 7% | 608 (8 workers sharing the CPU) |
| p_hint 0.3 | 37% | 33% | 17% | 13% | 682 |
| p_hint 0.7 | 44% | 30% | 21% | 5% | 563 |
| leaf off | 41% | 24% | 21% | 14% | 546 |
| H 6 | 42% | 33% | 19% | 6% | 422 |
| H 12 | 38% | 37% | 18% | 7% | 698 |
| N 100 | 39% | 29% | 29% | 3% | 257 |
| A\* | 46% | 40% | 14% | 0% | 0.4 |
| Q-learning | 12-18% | 33-44% | 44-52% | – | – |

Base MPC ties A\* (±5 points on 100 maps). It runs out of time on 7% of maps, where A\* never does. That points to dawdling or a gate deadlock near the exit.

A\* failure diagnosis (`diagnose_failures --agent astar`): 68 deaths. The dead took 318 damage from zombies and 34 from infection; 21 episodes had 5 or more gate-wait steps.

**Failure diagnosis, base config (2026-11-26, `diagnose_failures`, records in `results/scratch/exp3_diag/`):** MPC is too slow, not stuck at the gate.
- On the 24 maps both agents win, MPC takes 29.9 steps vs 19.8 for A\*. It takes less damage (1.6 vs 2.5) and picks up more items (1.8 vs 1.1). Gate opens at step 29 vs 19 on average.
- Over all maps total damage is the same (5.2 each). The slower pace hands the time back to the infection and zombies: MPC's dead took 79 infection damage vs 34 for A\*, and they die later (median t 25 vs 15) and farther from the exit (median 9 tiles vs 6).
- Of the 7 time-outs, 6 are a lone survivor (partner dead) still wandering 7-14 tiles from the exit at t = 100, with 75-85% of the map infected. Only 1 (seed 1035) is a gate deadlock: one survivor waited on the exit for 45 steps while the partner stayed 12 tiles away.
- Likely cause: the score. In the leaf, one tile of progress is worth +0.5, while one hit costs 3 (2 in the leaf, 1 in the reward), so the planner avoids any risk instead of advancing. And an 8-step horizon cannot see that waiting lets the infection grow. More samples or CEM optimize the same objective, so they are unlikely to help.
- The two agents fail on different maps: MPC wins 20 maps that A\* doesn't win, and A\* wins 22 that MPC doesn't.

**Leaf distance weight test (val, 100 maps, N=200, H=8; new `leaf_dist_weight` kwarg, default 0.5 = spec):**

| leaf_dist_weight | Success | Partial | Lost | Truncated | Return | Deaths/ep |
|---|---|---|---|---|---|---|
| 0.5 (base) | 44% | 30% | 19% | 7% | 1.65 | 0.73 |
| 1.5 | 37% | 45% | 17% | 1% | 0.26 | 0.80 |
| 3 | 14% | 53% | 32% | 1% | -8.81 | 1.18 |

A stronger pull removes the time-outs, but success drops and partial escapes rise. Likely reason (inferred, not tested): the leaf excludes dead survivors from the distance term, so a far-away survivor dying removes `w * d` of penalty. At w = 3 and d ≈ 10 that is worth more than the -10 death cost, so the planner stops protecting the trailing survivor. Keep 0.5 unless the leaf is changed (e.g. a dead survivor counts with a fixed distance), which is a team decision because Exp 4/5 reuse the score.

**Resume tomorrow:**
1. (2026-11-26: 7 of 9 done, table above. Claude Code stopped the run for low system memory during N 400, and the partial N 400 output was deleted. Consider `--workers 4` in `sweep.sh` to use less memory.) `bash results/scratch/exp3_sweep/sweep.sh` runs the remaining configs (N 400, and iters=3 with N=100) and skips finished ones. About 5.5 min per config with 8 workers. To stop it, kill the bash script as well as the python processes; stopping only the Python workers lets the script start the next config.
2. `python -m experiments.diagnose_failures --agent mpc --agent-kwargs "<best>" --workers 4` to see whether the truncations are gate deadlocks.
3. Pick the final config, time it with `--workers 4` (target < 0.5 s), and run it once on TEST with `--exp exp3`. Then update the README tables.

---

## 2026-11-24 (later): Q-learning fixed, no change to the game difficulty needed
Resolves the "decision needed" in the entry below: the game stays as it is, and the learners now improve on the full default game.

**Why it never learned (found by reproducing it):**
1. **Credit assignment:** a survivor stopped learning the moment it evacuated or died, so the first survivor out never received the +20 win (or +10 partial) that arrives when the partner finishes. Fix: it keeps learning in an *absorbing* key (see `docs/interfaces.md`) until the team episode ends, bootstrapping from `Q(s', WAIT)`.
2. **Loops and no-ops:** `SHOOT` with no target, and unseen states acted on at random, left greedy policies stuck repeating moves or ping-ponging. Fixes: `SHOOT` masking, and a head start for the exit-hint action in new Q rows and in unseen states.
3. **Noisy values and no time pressure:** constant alpha on a stochastic game, and features that cannot tell "early" from "late" while the infection spreads. Fixes: per-(state, action) alpha decay, and the `compact_t` features (hp bucket + time bucket). On the infection-only game this took greedy success from about 40% to 81%.
4. **Hard exploration:** a random policy dies in about 12 steps on the full game. Fix: `--curriculum` (no hazards, then infection, then zombies, then the full game), keeping the Q-tables between stages.

**Sanity check:** a hand-written policy (follow `exit_hint`, shoot when possible, stand on the exit) wins 100% with no hazards and 30% on the default game, so the game and features are solvable. Parking both survivors one tile from the exit and waiting there deadlocks: they block each other and the gate never opens.

**Results (val seeds, greedy policy, unshaped env reward):** before: 0% success, about 95% both die. After a staged run with 7000 full-game episodes: 15-18% both escape, 33-35% one escapes, 47-52% both die (100 maps, so about ±4 points). The scaled-down CLI smoke run (`--curriculum-scale 0.25 --episodes 2000`) gave 12% / 44% / 44%. A\* gets 46% / 40% / 14%.

**Run it:** `python -m experiments.train_qlearning --curriculum --episodes 7000 --run-seed 0` (then seeds 1 and 2). Expect roughly 6-10 minutes per seed on CPU. Checkpoints and `exp2_*` curves from earlier runs are stale (different features and update rule); regenerate them. 64 tests pass.

**Open:** the plateau is the zombies (most deaths). Ideas: a zombie-distance feature, `--shared-table`, more full-game episodes, tuning the stage lengths. In the report, describe shaping, the curriculum and the hint prior as training-time aids; evaluation uses the unshaped reward.

---

## 2026-11-24: Q-learning (Exp 2), built but blocked on a decision
- **Code:**
  - `agents/qlearning_agent.py`: two independent Q-tables, ε-greedy, save/load.
  - `experiments/train_qlearning.py`: `train()` function and CLI; learning curve every N episodes on `CURVE_EVAL`; env-step counts for sample efficiency; early snapshot for the before/after demo; curriculum support (`agent=`, `eval_cfg=`).
  - `analysis/plots.py curve`: mean ± std learning-curve figure.
  - `run_benchmark.py` / `watch.py` take `--checkpoint`.
  - 8 tests, including a hazard-free regression test that the learners must pass.
- **Compact features** (`state_encoding.q_features_compact`, documented in interfaces.md): the §14 fallback. The spec features (7.5×10⁸ values) left 25–30% of decisions in unseen states; the compact set gives 98% coverage.
- **Bug found and fixed:** a negative shaping potential rewarded standing still, so the greedy policy got stuck walking into walls. It now uses Φ = 1 − dist/area. On a hazard-free map the greedy policy goes from 0% to 85% success in 1.5k episodes.
- **Results on the full game (VAL, 20k episodes, seed 0):** every variant gets 0–1% success, 5–10% partial and 80–94% lost. Variants tried: shaping 0/10/30, ε decay 0.3/0.8, α 0.1/0.2, shared table, spec features, and a curriculum (0 → 1 → 2 → 3 zombies). For comparison, A\* gets 46% / 40% / 14%.
- **Easier game (2 zombies, chase 0.5):** Q-learning reaches 7% / 27% / 57% and is still rising, but A\* reaches 91% / 9% / 0%.
- Sweep logs and curves are in `results/scratch/qlearning_sweep/` (gitignored); the sweep checkpoints are in `checkpoints/qlearning/sweep/`.
- **Decision needed (team):** how to proceed with Q-learning and the game difficulty. Options are listed in the chat of 2026-11-24.

## 2026-11-22: A\* baseline (Exp 1) + watch mode
- `agents/astar_agent.py`: one A\* planner per survivor, replanning every step. Costs: infection +6, tile next to a zombie +4. Walls, zombies and the partner's tile are impassable. If the partner is on the closed exit, the goal becomes the exit's neighbours. A reactive rule shoots whenever a zombie is in its sights. Anti-deadlock: after 2 blocked moves it takes one random legal move. Debug routes are stored in `agent.paths`.
- `agents.make_agent(name)` and the `AGENTS` registry. `scripts/watch.py` lets you watch any agent play itself, with its paths overlaid. `experiments/run_benchmark.py` runs any agent, split and config, and writes `results/exp1_<agent>_<config>_<split>.csv`.
- A\* on val (100 maps): **46% both escape, 40% one escapes, 14% both die**, return 4.2 ± 17.6, about 21 steps, 0.4 ms/decision. Random agent: 0% success.
- Next: run A\* on the test split, the `no_coord` ablation and the shifted configs; Q-learning (P2); data collection (P3).

## 2026-11-22: Survivors can die individually
- **Rule change (spec updated):** the episode now ends only when no survivor is left. A dead survivor stays down as an inert body, and the partner can still escape alone (partial result, +10). Each death costs −10. The original rule is available as `EnvConfig(end_on_first_death=True)`.
- New info and events: `info["partial"]`, `info["dead"]`, `events["deaths"]`. New harness CSV columns: `partial`, `deaths`. Dead survivors are excluded from the grid survivor channel and the Q-features (partner offset becomes 4).
- **Team:** "success" still means both survivors evacuated. Decide whether the report also shows partial escapes.

## 2026-11-22: Visual overhaul
- New `env/render_pygame.Renderer`: pixel-art sprites (from `env/sprites.py`, exported to `assets/`), procedural city tiles, tweened movement, shot tracers, blood decals, damage flashes, floating text, infection spread-in, supply drops, an evacuation animation, a side panel HUD, and win/lose banners.
- `Renderer.draw_board(surface, state, origin)` draws just the map; use it twice for the side-by-side world-model demo.
- Added a headless render smoke test (`tests/test_render.py`).

## 2026-11-22: Scaffold
- Created the repo structure, `pyproject.toml`, `.venv` setup, the full env implementation (pure NumPy), state encodings, the Pygame renderer and play script, a random agent, the eval harness, seed splits, and 34 rule and determinism tests.
- Next: team reviews `docs/interfaces.md`; P1 starts A\*, P2 starts Q-learning, P3 starts `scripts/collect_data.py` and true-dynamics MPC.
