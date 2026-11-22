import numpy as np
import pytest

from agents import RandomAgent
from env import DEFAULT_CONFIG, SHIFTED_CONFIGS, OutbreakEnv
from env.map_gen import INF
from experiments.harness import run_episode


def _rollout(seed, config=DEFAULT_CONFIG):
    env = OutbreakEnv(config)
    row = run_episode(env, RandomAgent(seed=7), seed, record=True)
    return row["trajectory"], env.state


def test_same_seed_same_trajectory():
    for seed in (0, 1, 42, 999):
        tr_a, s_a = _rollout(seed)
        tr_b, s_b = _rollout(seed)
        assert tr_a == tr_b
        assert (s_a.infection == s_b.infection).all()
        assert (s_a.zombie_pos == s_b.zombie_pos).all()


def test_reset_fixes_map():
    env = OutbreakEnv()
    o1, _ = env.reset(seed=5)
    env.step((0, 0))
    o2, _ = env.reset(seed=5)
    assert np.array_equal(o1["grid"], o2["grid"])


def test_different_seeds_differ():
    env = OutbreakEnv()
    grids = {env.reset(seed=s)[0]["grid"].tobytes() for s in range(20)}
    assert len(grids) > 15


@pytest.mark.parametrize("config", [DEFAULT_CONFIG, *SHIFTED_CONFIGS.values()])
def test_path_to_exit_exists_at_start(config):
    env = OutbreakEnv(config)
    for seed in range(100):
        env.reset(seed=seed)
        s = env.state
        for i in (0, 1):
            assert s.exit_dist[s.surv_pos[i, 0], s.surv_pos[i, 1]] < INF


def test_get_set_state_replays_identically():
    env = OutbreakEnv()
    env.reset(seed=3)
    env.step((4, 4))
    snap = env.get_state()
    actions = np.random.default_rng(0).integers(6, size=(30, 2))

    def play():
        out = []
        for a in actions:
            _, r, term, trunc, _ = env.step(tuple(a))
            out.append(r)
            if term or trunc:
                break
        return out

    out_a = play()
    s_a = env.get_state()
    env.set_state(snap)
    out_b = play()
    s_b = env.get_state()
    assert len(out_a) > 1 and out_a == out_b
    assert (s_a.infection == s_b.infection).all() and (s_a.zombie_pos == s_b.zombie_pos).all()


def test_clone_is_independent():
    env = OutbreakEnv()
    env.reset(seed=11)
    c = env.clone()
    c.step((0, 0))
    assert env.state.t == 0 and c.state.t == 1
