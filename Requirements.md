# Zombie Outbreak: Project Requirements

Course project: _Learning to Play by Imagining: Building and Benchmarking World Models for Intelligent Game Agents_
Team: 3 students | Timeline: 1 week | Stack: Python + Pygame (NumPy, PyTorch, Gymnasium, Matplotlib)

---

## 1. Project Summary

A survivor (optionally two cooperating survivors) on a small city grid must reach an evacuation point before a spreading infection and chasing zombies kill them. The infection spread and zombie movement are the dynamics a learned world model must predict, and a planner uses that model to imagine futures before acting.

**Research question:** Can an agent learn an internal model of the game world, imagine future states, and use them to decide better than conventional agents? Does a more accurate world model necessarily lead to better decisions?

**Base paper:** STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning (Zhang et al., NeurIPS 2023), https://arxiv.org/abs/2310.09615. Implemented as a simplified, STORM-inspired adaptation (see Section 6).

**Supporting papers:** PETS (Chua et al., NeurIPS 2018), MOPO (Yu et al., NeurIPS 2020), DreamerV3 (context only). Verify exact citations before use.

---

## 2. Game Environment (Course Component 1)

| Item                  | Requirement                                                                                                                                                                                                             |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Grid                  | 8x8 (10x10 if time allows), walls, procedurally generated from a seed; a path from start to exit must exist at step 0                                                                                                   |
| Agents                | 1 survivor (core). Second cooperative survivor is a stretch goal, decided at the end of Day 2                                                                                                                           |
| Survivor stats        | Health 0-5, ammo 0-3                                                                                                                                                                                                    |
| Objects (>= 3)        | Zombie, medkit, ammo crate, wall, evacuation point                                                                                                                                                                      |
| Dynamic events (>= 2) | (1) Infection spreads to adjacent clean tiles with probability p per step. (2) Zombies move toward the nearest survivor with probability 0.7, otherwise randomly. (3) Supply drops appear every K steps on a clean tile |
| Actions               | Up, down, left, right, wait, shoot (kills nearest zombie within 2 tiles in a straight line, costs 1 ammo)                                                                                                               |
| Damage                | 1 health per step on an infected tile; 1 health per step while adjacent to a zombie                                                                                                                                     |
| Pickups               | Medkit +2 health, ammo crate +1 ammo, applied on entering the tile                                                                                                                                                      |
| Win                   | Reach the evacuation tile (never infected)                                                                                                                                                                              |
| Lose                  | Health reaches 0, or the 100-step limit is reached                                                                                                                                                                      |
| Reward                | +20 exit, -20 death, -0.1 per step, -1 per damage taken, +2 per zombie killed, +1 per pickup (shared reward if two survivors)                                                                                           |
| Reproducibility       | `reset(seed=...)` fixes the map, zombie starts and all random events; experiments use fixed seed lists                                                                                                                  |
| Observability         | Full observation in the base game                                                                                                                                                                                       |
| Playable              | Keyboard control in Pygame (arrow keys, space to shoot, key for wait)                                                                                                                                                   |

**Seed splits:** train 0-999, validation 1000-1099, test 2000-2099, plus a separate "shifted" set for generalization.

**Acceptance:** rule tests pass (damage, pickups, win/lose, infection spread, determinism under the same seed).

---

## 3. State and Representations (Course Component 2)

Keep and document three representations. This is also the "actual dynamics vs learned model" distinction required by the rubric.

| Use              | Representation                                                                                                                                                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| True environment | Full grid channels (walls, infection, zombies, medkits, ammo, survivor(s), exit) plus health, ammo, time step                                                                                                       |
| Q-learning       | Compact egocentric features: direction hint toward exit (wall-aware), infected flags for the 4 neighbours, wall flags for the 4 neighbours, nearest-zombie offset clipped to +/-3, health, ammo. Dictionary Q-table |
| World model      | Full grid channels plus one-hot action                                                                                                                                                                              |

**Fairness rule:** document what each agent knows. A\* uses the current snapshot; Q-learning sees only local features; both planners get the full state.

---

## 4. Agents and Experiments

| #   | Agent                               | Description                                                                                                                                                    | Course experiment |
| --- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| 1   | A\*                                 | Plans on the current grid; infected tiles and tiles next to zombies have high cost; replans every step; no model of the future                                 | Exp 1             |
| 2   | Q-learning (compulsory)             | Epsilon-greedy tabular Q-learning with decay; trained on random training maps; independent Q-tables per survivor if two survivors; DQN only if the table fails | Exp 2             |
| 3   | True-dynamics MPC                   | Random-shooting MPC (N sequences, horizon H) using a copy of the real simulator                                                                                | Exp 3             |
| 4   | Learned-model MPC                   | Same planner as agent 3, with imagined rollouts from the learned world model; logs predicted vs actual next state                                              | Exp 4             |
| 5   | Uncertainty-aware MPC (improvement) | Same as agent 4, scoring each plan as `mean_return - lambda * ensemble_disagreement`                                                                           | Exp 5             |
| -   | Generalization and robustness       | All agents evaluated on shifted conditions without retraining                                                                                                  | Exp 6             |

**Planner defaults (tune):** N = 200-300 sequences, H = 5-8, replan every step. Shorten H before dropping to one-step lookahead if time is short.

**Shifted conditions (Exp 6):** unseen maps, infection rate 2p, more zombies, faster zombies, fewer supplies, different wall layout.

---

## 5. World Model (Course Component 2 and 4)

- **Formulation:** s_hat(t+1) = f_theta(s_t, a_t), plus reward and done predictions.
- **Inputs:** state channels plus one-hot action.
- **Outputs:** next infection map (per-cell probability), zombie moves (categorical over 5 moves per zombie, to avoid creating or losing zombies), survivor position, health, ammo, reward, done flag. Static parts (walls, exit) are copied through.
- **Architecture:** small CNN with MLP heads for scalars. Ensemble of 3 models with different seeds and data shuffles (5 if time allows).
- **Stochastic prediction (STORM-inspired):** imagined rollouts sample from predicted distributions.
- **Loss:** cross-entropy for maps and categorical outputs, MSE for health, ammo and reward.
- **Data:** about 50k transitions from random play, Q-agent and A\*; stored as compressed `.npz` with episode seeds, terminal and truncated flags kept separate; collected headless. Optional second round collected with the planner, followed by retraining.
- **Evaluation:** one-step error (infection log-loss or F1, position accuracy, health and reward error) and multi-step error (open-loop replay for 1, 3, 5, 10 steps against the real environment) on held-out test seeds.

---

## 6. Research Reproduction and Improvement (Course Component 5)

**What is kept from STORM:** learned dynamics, reward and termination prediction; stochastic prediction with sampled imagined rollouts; planning in imagination.

**What is dropped or changed (must be stated in the report):** transformer sequence model, categorical VAE over pixels, actor-critic trained in imagination (replaced by MPC), pixel observations (replaced by a grid-state model).

**Improvement (Section 7 of the course description):** ensemble-disagreement penalty on candidate plans (from PETS/MOPO), motivated by STORM's observation that model-reality mismatch makes agents pursue virtual goals. Evaluated against the STORM-inspired planner (agent 4) with a lambda sweep, in-distribution and shifted.

**Report must explain:** the selected paper's problem formulation, architecture, methodology, experimental setup and limitations; and how the modification differs from the original method.

---

## 7. Metrics

- Average cumulative reward
- Task completion (success) rate
- Survival time
- One-step state prediction error
- Multi-step prediction error
- Learning curve (reward and success rate every 200 episodes on fixed evaluation seeds)
- Sample efficiency (environment steps to a target success rate; for the model-based agent, transitions used to train the model)
- Planning performance (success vs horizon H and vs number of samples)
- Generalization gap (shifted vs training conditions)
- Computational cost (training time, ms per decision, parameter count; log from Day 2)

**Accuracy vs decision study:** train world models on 3 data sizes (for example 1k, 10k, 50k transitions), plot prediction error against planner success, and report the correlation.

**Evidence:** at least 3 seeds per experiment (5 if time allows), 100 evaluation episodes per condition where possible, mean and standard deviation, fixed seed lists, tables and graphs.

---

## 8. Technical Stack

- Python 3.10+
- Pygame (playable view and demo viewer)
- NumPy (game logic)
- PyTorch (world model)
- Gymnasium (environment API)
- Matplotlib (plots)
- TensorBoard (optional)
- Training and data collection run on CPU, with Colab or Kaggle for the final multi-seed runs

**Rule:** game logic is pure NumPy with no Pygame imports. Rendering only draws a state. Experiments run headless (set `SDL_VIDEODRIVER=dummy` if Pygame is ever imported on a server).

---

## 9. Repository Layout

```
zombie-outbreak/
  env/          outbreak_env.py, map_gen.py, render_pygame.py, config.py
  agents/       astar_agent.py, qlearning_agent.py, mpc_planner.py
  world_model/  dataset.py, model.py, train.py, eval_prediction.py
  experiments/  run_benchmark.py, run_generalization.py,
                accuracy_vs_decision.py, configs/
  analysis/     plots.py, tables.py
  tests/        determinism and rule tests
  results/      csv and figures
  report/
  requirements.md
```

---

## 10. Deliverables

1. Playable game (Pygame, keyboard control)
2. Well-organized, documented source code
3. World model implementation (representation and prediction)
4. Learning-based agent (Q-learning) with training curves
5. Planning agents (true-dynamics and learned-model MPC)
6. Research-inspired approach: code and explanation (STORM-inspired model and ensemble penalty)
7. Experimental results: tables, graphs, comparative analysis
8. Technical report, 6-10 pages, following the 13-section research format (abstract, introduction, related work, game and formulation, state and world model, agents and planning, reproduction and baselines, experimental setup, results, discussion and limitations, improvement and future work, conclusion, references)
9. Short proposal and literature review (STORM plus PETS/MOPO)
10. 10-15 minute presentation and live demo, plus a recorded backup video
11. Disclosure of generative-AI use per course policy; citation of all papers, libraries and resources

---

## 11. Demo Requirements

1. Game running; a person plays briefly and the rules are shown
2. Q-learning agent acting autonomously, with the learning curve and before/after behaviour
3. World model predicting future states (side-by-side window: real game on the left, predicted next state on the right)
4. Planner scoring several candidate actions and picking one
5. Actual outcome compared with the predicted outcome
6. Comparison table: A\*, Q-learning, true-dynamics MPC, learned-model MPC, uncertainty-aware MPC
7. Generalization results and the accuracy-vs-decision plot
8. The proposed improvement and its results

---

## 12. One-Week Plan and Team Split

| Person   | Responsibilities                                                        |
| -------- | ----------------------------------------------------------------------- |
| Person 1 | Environment, Pygame renderer and demo viewer, A\*                       |
| Person 2 | Q-learning, evaluation harness (fixed seeds, logging), plots and tables |
| Person 3 | Data collection, world model, MPC planners, improvement                 |
| All      | Literature, report sections, demo rehearsal                             |

| Day | Goals                                                                                                                                         |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Freeze spec and interfaces (env API, state tensor shape, dataset format); skim STORM and PETS; env skeleton; repo and harness; short proposal |
| 2   | Env complete with rule tests; A\*; Q-learning training starts; true-dynamics MPC; decide on second survivor                                   |
| 3   | Collect about 50k transitions; train CNN; first Q-learning curve; one-step error                                                              |
| 4   | Learned-model MPC; multi-step error; predicted-vs-actual logging; side-by-side viewer                                                         |
| 5   | Ensemble penalty and lambda sweep; generalization runs                                                                                        |
| 6   | Final multi-seed runs; tables and plots; accuracy-vs-decision study                                                                           |
| 7   | Report, slides, demo rehearsal, backup video                                                                                                  |

**Working rules:** one Git repo; merge daily; do not change the shared state format without telling everyone; write report sections as results appear; run heavy jobs overnight on Colab or Kaggle.

**Fallbacks if behind on Day 4:** drop the lambda sweep breadth or generalization breadth; keep the core pipeline (env, A\*, Q-learning, both MPC agents, comparison).

---

## 13. Out of Scope (Explicit Cuts)

DQN (unless Q-learning fails), MCTS and CEM planners, Three.js viewer, day/night cycle, fog-of-war, barricade building, zombie variants, transformer or latent-VAE world model, actor-critic in imagination, 5-model ensembles and 5+ seeds (stretch only).

---

## 14. Risks and Mitigations

| Risk                                         | Mitigation                                                                     |
| -------------------------------------------- | ------------------------------------------------------------------------------ |
| Q-learning does not learn                    | Simplify features, shape the reward, shrink the grid, fall back to DQN         |
| Model errors compound over long horizons     | Short horizon, replan every step, ensemble, disagreement penalty               |
| Planner exploits model errors                | Uncertainty penalty (the improvement)                                          |
| Sampled zombie positions become inconsistent | Predict each zombie's move as a categorical over 5 moves                       |
| Planning too slow on CPU                     | Batch rollouts, smaller N and H, run final jobs on Colab or Kaggle             |
| Scope creep                                  | Freeze the spec on Day 1; add stretch items only after the full pipeline works |
| Noisy results                                | Fixed seed lists, 3+ seeds, 100 evaluation episodes per condition              |

---

## 15. Open Decisions

1. Second cooperative survivor: include or not (decision gate: end of Day 2). Not required by the course description.
2. Base-paper framing: STORM-inspired stochastic model as the "reproduction" with the ensemble penalty as the "small improvement" (current plan); confirm with the instructor if concerned.
3. Grid size: 8x8 vs 10x10.
4. Number of seeds: 3 minimum, 5 if time allows.

---

## 16. Final Checklist (maps to the course description)

- [ ] Playable game, >= 3 object types, >= 2 dynamic events, reset and reproducibility
- [ ] State, action, observation, reward and transition documented
- [ ] Q-learning trained, learning curve and multiple evaluation episodes reported
- [ ] World model predicts next state and reward; one-step and multi-step error reported
- [ ] Planning with imagined futures; predicted vs actual outcome shown
- [ ] A\* baseline (Exp 1), Q-learning (Exp 2), true-dynamics planner (Exp 3), learned-model planner (Exp 4)
- [ ] Research-inspired improvement evaluated against its baseline (Exp 5)
- [ ] Generalization and robustness tests (Exp 6)
- [ ] All metrics in Section 7 reported, including sample efficiency and computational cost
- [ ] Accuracy-vs-decision analysis
- [ ] Compared against at least two baselines under comparable conditions
- [ ] STORM discussed in the report; simplifications stated explicitly
- [ ] All sources cited; generative-AI use disclosed
- [ ] Report (6-10 pages), presentation (10-15 min), live demo, backup video
