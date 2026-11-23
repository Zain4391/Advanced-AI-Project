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
| MPC: true dynamics / learned / uncertainty | P3 | ⬜ |
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
