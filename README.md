# Zombie Outbreak: World Models for Intelligent Game Agents

Two cooperating survivors must reach the evacuation gate together on a small grid while infection spreads and zombies chase them. We benchmark A\*, independent Q-learners, and MPC planners that use the true simulator, a learned (STORM-inspired) world model, or an uncertainty-penalized ensemble.

- Spec: [`Requirements.md`](Requirements.md)
- Frozen interfaces: [`docs/interfaces.md`](docs/interfaces.md)
- Status and handoff log: [`docs/progress.md`](docs/progress.md)

## Setup
```bash
python -m venv .venv
.venv/Scripts/activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev,model]"
pytest -q
```

## Play
```bash
python scripts/play.py --seed 0
```
| | Move | Shoot | Wait |
|---|---|---|---|
| Survivor 1 (blue) | arrow keys | Space | Enter |
| Survivor 2 (orange) | W A S D | F | Q |

The game advances one step once both survivors have chosen an action. R restarts, N loads the next seed, Esc quits.
The gate opens only when both survivors are next to the exit. A survivor who steps onto the open exit evacuates, and the team wins once both have evacuated.

Options: `--cell 64` for a smaller window, `--size 10` for a 10x10 map.

## Assets
The pixel-art sprites live in `assets/*.png` (16x16; HUD icons 8x8). Their source definitions are in `env/sprites.py`, and the tiles are generated in `env/render_pygame.py`.
To reskin the game, edit or replace any PNG and keep its file name. The renderer prefers the PNG over the built-in version. Delete a PNG to restore the default, or run `python scripts/make_assets.py --force` to regenerate all of them.

## Layout
```
env/          config, state, map_gen, outbreak_env (pure NumPy), state_encoding,
              sprites (pixel-art data), render_pygame (Renderer: sprites, tweens, effects, HUD)
assets/       PNG sprites (reskinnable)
agents/       base, random_agent, (astar, qlearning, mpc_planner)
world_model/  (dataset, model, train, eval_prediction, rollout)
experiments/  seeds, harness, (run_benchmark, run_generalization, accuracy_vs_decision), configs/
analysis/     (plots, tables)
scripts/      play, make_assets, (collect_data, demo_side_by_side)
tests/        rule + determinism tests
results/      CSVs and figures
report/       report and slides
docs/         interfaces, progress log
```
