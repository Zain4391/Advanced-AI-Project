"""Single source of truth for seed splits (Requirements.md, Section 2). Never hard-code seeds elsewhere."""

TRAIN = list(range(0, 1000))
VAL = list(range(1000, 1100))
TEST = list(range(2000, 2100))
SHIFTED = list(range(3000, 3100))

# Seeds for the *agent/model* randomness (training runs), not for maps. >= 3 per experiment.
RUN_SEEDS = [0, 1, 2]
RUN_SEEDS_STRETCH = [0, 1, 2, 3, 4]

# Fixed evaluation subset for learning curves (every 200 training episodes).
CURVE_EVAL = VAL[:20]

SPLITS = {"train": TRAIN, "val": VAL, "test": TEST, "shifted": SHIFTED}
