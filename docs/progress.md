# Progress Log

Newest entry at the top. At the end of every work session (human or Claude), add an entry: what changed, what's next, and any open problems. Keep entries short.

---

## Status board

| Component | Owner | Status |
|---|---|---|
| Env + rule tests | P1 | ✅ scaffold done; 34 tests pass; rules await team sign-off |
| Pygame play (`scripts/play.py`) | P1 | ✅ basic version |
| A\* agent | P1 | ⬜ |
| Eval harness + seeds | P2 | ✅ scaffold (`experiments/harness.py`, `seeds.py`) |
| Q-learning (2 independent learners) | P2 | ⬜ |
| Plots / tables | P2 | ⬜ |
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
