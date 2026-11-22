# CLAUDE.md

Course project: a two-survivor zombie-outbreak grid game used to benchmark A\*, Q-learning, and MPC with true and learned world models (STORM-inspired, with an ensemble-disagreement penalty). Team of 3, one-week timeline.

## Read first, every session
1. `docs/progress.md`: current status, open decisions, last session's handoff. **Start here.**
2. `docs/interfaces.md`: frozen contracts (env API, actions, tensors, step order, dataset, CSV).
3. `Requirements.md`: the full spec. Read only the section relevant to the task; it's long.

## Commands
```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"   # Windows paths; torch already installed globally or via requirements.txt
.venv/Scripts/python -m pytest -q                 # all rule + determinism tests
.venv/Scripts/python -m experiments.harness       # smoke test: random agent on 50 val seeds
.venv/Scripts/python scripts/play.py --seed 0     # keyboard play
.venv/Scripts/python scripts/watch.py --agent astar --auto     # watch an agent play itself
.venv/Scripts/python -m experiments.run_benchmark --agent astar --split val   # headless eval -> results/*.csv
```

## Hard rules
- `env/` game logic is **pure NumPy**. Never import pygame outside `env/render_pygame.py` and `scripts/`.
- Never change constants in `env/config.py` (actions, channels, scalars), the step order, or the dataset format without updating `docs/interfaces.md` and flagging it to the user. Teammates depend on them.
- All seeds come from `experiments/seeds.py`. Never hard-code seed lists.
- All agents return a joint action `(a1, a2)` through `agents/base.Agent`. Only planners may use `env.get_state()` or `env.clone()` (fairness rule).
- Every evaluation goes through `experiments/harness.evaluate()` so the CSVs stay comparable.
- All feature and tensor encoding goes through `env/state_encoding.py`.
- Any change to env rules needs a test in `tests/test_rules.py`, and `pytest` must pass before committing.
- Don't name a virtualenv `env/`; that name is our package. Use `.venv/`.
- Out of scope (Requirements §13): DQN unless Q-learning fails, MCTS/CEM, transformers/VAE, fog-of-war, and so on. Push back on scope creep.

## End of every session
Update `docs/progress.md`: the status board, plus a dated entry covering what changed, what's next, and any open problems.
