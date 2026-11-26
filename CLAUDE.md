# CLAUDE.md

Course project: a two-survivor zombie-outbreak grid game used to benchmark A\*, Q-learning, and MPC with true and learned world models (STORM-inspired, with an ensemble-disagreement penalty). Team of 3, one-week timeline.

## Status (details in `docs/progress.md`)
- Done: environment, A\* (about 46% both escape on val), Q-learning (about 12-18%; recipe and results in the 2026-11-24 progress entry).
- Now: Exp 3, the true-dynamics MPC planner. Next: data collection and the learned world model (Exp 4), the ensemble-disagreement penalty (Exp 5), generalization (Exp 6).

## Read first, every session
1. `docs/progress.md`: current status, open decisions, last session's handoff. **Start here.**
2. `docs/interfaces.md`: frozen contracts (env API, actions, tensors, step order, dataset, CSV).
3. `Requirements.md`: the full spec. Read only the section relevant to the task (§4 agents, §5 world model, §7 metrics); it's long.

## Commands
Write each command on one line. Windows paths shown; use `.venv/bin/python` elsewhere.
```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"   # torch is installed globally or via requirements.txt
.venv/Scripts/python -m pytest -q                 # all rule, determinism, agent and render tests
.venv/Scripts/python -m experiments.harness       # smoke test: random agent on 50 val seeds
.venv/Scripts/python scripts/play.py --seed 0     # keyboard play
.venv/Scripts/python scripts/watch.py --agent astar --seed 2000 --auto    # watch an agent play itself
.venv/Scripts/python scripts/watch.py --agent qlearning --checkpoint checkpoints/qlearning/default_seed0.pkl --seed 2000
.venv/Scripts/python -m experiments.run_benchmark --agent astar --split val             # headless eval -> results/*.csv
.venv/Scripts/python -m experiments.run_benchmark --agent qlearning --checkpoint checkpoints/qlearning/default_seed0.pkl --agent-seed 0 --split test --exp exp2
.venv/Scripts/python -m experiments.train_qlearning --curriculum --episodes 7000 --run-seed 0    # then seeds 1 and 2
```
`--agent-seed` must differ per checkpoint, or the result CSVs overwrite each other.

## Hard rules
- `env/` game logic is **pure NumPy**. Never import pygame outside `env/render_pygame.py` and `scripts/`.
- Never change constants in `env/config.py` (actions, channels, scalars), the step order, or the dataset format without updating `docs/interfaces.md` and flagging it to the user. Teammates depend on them.
- All seeds come from `experiments/seeds.py`. Never hard-code seed lists.
- All agents return a joint action `(a1, a2)` through `agents/base.Agent`. Only planners may use `env.get_state()` or `env.clone()` (fairness rule).
- Every evaluation goes through `experiments/harness.evaluate()` so the CSVs stay comparable.
- All feature and tensor encoding goes through `env/state_encoding.py`.
- Any change to env rules needs a test in `tests/test_rules.py`, and `pytest` must pass before committing.
- Don't name a virtualenv `env/`; that name is our package. Use `.venv/`.
- Out of scope (Requirements §13): MCTS, transformers/VAE, fog-of-war, and so on. DQN only if Q-learning regresses. CEM is allowed only as an option (`iters`) inside the one sampling-based planner. Push back on scope creep.

## Q-learning (done: don't regress it)
- Training-only aids (potential shaping, curriculum, hint prior, `SHOOT` mask, alpha decay) must never touch evaluation. Evaluation is the greedy policy with the unshaped env reward, through the harness.
- The compact feature key layout is relied on by the agent: slot 0 = exit hint, slot 6 = can_shoot. Inactive survivors (evacuated or dead) have an absorbing key so the shared win reward reaches them. Changing the layout means updating `_legal`, the docs and the tests, and retraining. Checkpoints from before 2026-11-24 are stale.
- Checkpoints store their own flags. Load them with `make_agent("qlearning", checkpoint=...)`.

## Planners (Exp 3, 4, 5)
- **One planner implementation** in `agents/mpc_planner.py` serves Exp 3, 4 and 5. Only the dynamics backend changes (true simulator, then learned model), and then the score gets a disagreement penalty. Never fork the planner per experiment: the comparison must isolate the effect of the model.
- Method: sampling-based MPC. Sample N joint-action sequences of horizon H (36 joint actions per step), roll each out, score it, execute the first action, replan every step. Optional elite refit (`iters` > 1) is the CEM variant.
- Score = discounted env reward over the rollout, plus a leaf heuristic if the rollout did not end: `-0.5 * (d1 + d2) + 2 * (hp1 + hp2)`, where `d` is the BFS distance from `state.exit_dist` (0 once evacuated) and dead survivors contribute no distance.
- Only the true-simulator backend may call `env.get_state()`, `env.clone()` and `reseed_dynamics()`. Planners never mutate the live env. A learned backend sees only what a model could know (no RNG state, no future).
- Evaluate sequences under the same random futures within one decision (common random numbers) so the comparison between sequences is not noise.
- Tune on `val`. Use `test` only for final numbers. Record N, H, iters and lambda as extra CSV columns.
- Measure ms per decision. Use `--workers` (one process per map group) so 100-map evaluations finish in reasonable time.

## Files and formatting
- `docs/*.md`, `Requirements.md` and this file use CRLF line endings; `README.md` uses LF. Preserve the line endings when editing, and check that `git diff --stat` stays small (no whole-file rewrites).
- Result files: `results/exp{N}_{agent}_{condition}_{split}[_seed{k}].csv`. `checkpoints/`, `data/` and `*.pt` are gitignored.
- Windows multiprocessing uses spawn: guard entry points with `if __name__ == "__main__":` and keep module-level code free of work.

## End of every session
Update `docs/progress.md`: the status board, plus a dated entry covering what changed, what's next, and any open problems. Update the README results table when a new agent has numbers.
