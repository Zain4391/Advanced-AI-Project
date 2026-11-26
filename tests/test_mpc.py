"""Tests for the sampling-based MPC planner (Exp 3) and the parallel benchmark runner."""
import numpy as np
import pytest

from agents import MPCAgent, RandomAgent, make_agent
from agents.mpc_planner import (
    RolloutBatch, TrueSimDynamics, action_mask, initial_probs, leaf_value, refit_probs,
    sample_sequences, score_rollouts,
)
from env import DEFAULT_CONFIG, SHOOT, WAIT, EnvState, OutbreakEnv
from env.state_encoding import exit_hint
from experiments.harness import evaluate
from experiments.run_benchmark import parse_kwargs, run
from experiments.seeds import VAL

SMALL = dict(n_samples=24, horizon=4)


def _states_equal(a: EnvState, b: EnvState) -> bool:
    arrays = ("infection", "medkits", "ammo_crates", "surv_pos", "health", "ammo", "evacuated",
              "zombie_pos", "zombie_alive")
    return (all(np.array_equal(getattr(a, k), getattr(b, k)) for k in arrays)
            and (a.gate_open, a.t, a.done) == (b.gate_open, b.t, b.done))


# ------------------------------------------------------------ determinism ---
def test_same_seed_same_actions():
    def play(seed):
        env = OutbreakEnv(DEFAULT_CONFIG)
        obs, _ = env.reset(seed=VAL[0])
        agent = MPCAgent(seed=seed, **SMALL)
        agent.reset(VAL[0])
        actions, done = [], False
        while not done and len(actions) < 30:
            a = agent.act(obs, env)
            actions.append(a)
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
        return actions

    assert play(0) == play(0)


def test_act_leaves_live_env_unchanged():
    env = OutbreakEnv(DEFAULT_CONFIG)
    obs, _ = env.reset(seed=VAL[1])
    for _ in range(3):
        obs, *_ = env.step((WAIT, WAIT))
    before = env.get_state()
    agent = MPCAgent(iters=2, n_futures=2, **SMALL)
    agent.reset(VAL[1])
    agent.act(obs, env)
    after = env.get_state()
    assert _states_equal(before, after)
    assert before.rng_state == after.rng_state
    # the live env still continues exactly as an untouched copy would
    twin = OutbreakEnv(DEFAULT_CONFIG)
    twin.reset(seed=VAL[1])
    twin.set_state(before)
    r1 = env.step((WAIT, WAIT))[1]
    r2 = twin.step((WAIT, WAIT))[1]
    assert r1 == r2 and _states_equal(env.state, twin.state)


# ---------------------------------------------------------------- sampler ---
def _quiet_state(**kw):
    s = EnvState.empty(DEFAULT_CONFIG.grid_size, DEFAULT_CONFIG.max_zombies)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_sampler_no_shoot_without_ammo_and_inactive_waits():
    s = _quiet_state(ammo=np.array([0, 2]), evacuated=np.array([False, True]))
    mask = action_mask(s)
    assert not mask[0, SHOOT] and mask[0, :5].all()
    assert mask[1].tolist() == [False, False, False, False, True, False]
    probs = initial_probs(s, horizon=5, p_hint=0.5)
    assert probs.shape == (5, 2, 6)
    np.testing.assert_allclose(probs.sum(-1), 1.0)
    seqs = sample_sequences(probs, 500, np.random.default_rng(0))
    assert seqs.shape == (500, 5, 2) and seqs.dtype == np.int64
    assert not (seqs[:, :, 0] == SHOOT).any()
    assert (seqs[:, :, 1] == WAIT).all()


def test_sampler_hint_gets_p_hint():
    s = _quiet_state()
    probs = initial_probs(s, horizon=3, p_hint=0.7)
    for i in (0, 1):
        h = exit_hint(s, i)
        assert h < 4 and np.allclose(probs[:, i, h], 0.7)
        others = np.delete(probs[0, i], h)
        np.testing.assert_allclose(others, 0.3 / 5)


def test_dead_survivor_waits():
    s = _quiet_state(health=np.array([0, 5]))
    probs = initial_probs(s, horizon=2, p_hint=0.5)
    assert (probs[:, 0, WAIT] == 1.0).all()


def test_refit_moves_toward_elites():
    """Toy problem: score = number of RIGHT actions by survivor 0. CEM should shift mass to RIGHT."""
    rng = np.random.default_rng(0)
    mask = np.ones((2, 6), bool)
    probs = np.full((4, 2, 6), 1 / 6)
    for _ in range(3):
        seqs = sample_sequences(probs, 200, rng)
        score = (seqs[:, :, 0] == 3).sum(1)
        elites = seqs[np.argsort(-score, kind="stable")[:20]]
        new = refit_probs(probs, elites, mask, smoothing=0.3)
        assert (new[:, 0, 3] > probs[:, 0, 3]).all()
        probs = new
    assert (probs[:, 0, 3] > 0.8).all()
    np.testing.assert_allclose(probs.sum(-1), 1.0)


def test_refit_keeps_mask():
    mask = np.ones((2, 6), bool)
    mask[0, SHOOT] = False
    probs = np.full((2, 2, 6), 1 / 6)
    probs[:, 0, SHOOT] = 0
    probs[:, 0] /= probs[:, 0].sum(-1, keepdims=True)
    elites = np.full((5, 2, 2), SHOOT)  # nonsense elites can't revive a masked action
    elites[:, :, 0] = 1
    new = refit_probs(probs, elites, mask, 0.3)
    assert (new[:, 0, SHOOT] == 0).all()


# ---------------------------------------------------------------- scoring ---
def _batch(rewards, final_pos, hp, evac, done, dis=None):
    rewards = np.asarray(rewards, float)
    n = len(rewards)
    return RolloutBatch(rewards, np.zeros_like(rewards, bool), np.asarray(final_pos), np.asarray(hp),
                        np.asarray(evac, bool), np.asarray(done, bool),
                        np.zeros(n) if dis is None else np.asarray(dis, float))


def test_leaf_value_evacuated_and_dead():
    exit_dist = np.arange(64).reshape(8, 8)  # distance of (r, c) = 8r + c
    b = _batch([[0.0]] * 3,
               final_pos=[[[0, 2], [0, 3]], [[0, 2], [0, 3]], [[0, 2], [0, 3]]],
               hp=[[5, 4], [5, 4], [0, 4]],
               evac=[[False, False], [True, False], [False, False]],
               done=[False] * 3)
    v = leaf_value(b, exit_dist)
    assert v[0] == pytest.approx(-0.5 * (2 + 3) + 2 * 9)
    assert v[1] == pytest.approx(-0.5 * 3 + 2 * 9)   # evacuated: d = 0
    assert v[2] == pytest.approx(-0.5 * 3 + 2 * 4)   # dead: no distance, hp 0
    w = leaf_value(b, exit_dist, dist_weight=3.0)
    assert w[0] == pytest.approx(-3.0 * (2 + 3) + 2 * 9)


def test_score_discount_leaf_and_penalty():
    exit_dist = np.ones((8, 8), int)
    b = _batch([[1.0, 1.0], [1.0, 1.0]], final_pos=[[[0, 0], [0, 0]]] * 2, hp=[[1, 1]] * 2,
               evac=[[False, False]] * 2, done=[False, True], dis=[0.5, 0.5])
    s = score_rollouts(b, exit_dist, gamma=0.5, leaf=True, lam=2.0)
    leaf = -0.5 * 2 + 2 * 2
    assert s[0] == pytest.approx(1.5 + leaf - 1.0)
    assert s[1] == pytest.approx(1.5 - 1.0)          # finished rollouts get no leaf
    s = score_rollouts(b, exit_dist, gamma=0.5, leaf=False, lam=0.0)
    np.testing.assert_allclose(s, [1.5, 1.5])


def test_true_sim_rollout_shapes_and_crn():
    env = OutbreakEnv(DEFAULT_CONFIG)
    env.reset(seed=VAL[3])
    seqs = sample_sequences(np.full((5, 2, 6), 1 / 6), 6, np.random.default_rng(1))
    seqs[3] = seqs[0]
    b = TrueSimDynamics().rollout(env, seqs, np.full(6, 7))
    assert b.rewards.shape == (6, 5) and b.final_pos.shape == (6, 2, 2) and b.final_hp.shape == (6, 2)
    assert b.final_evacuated.shape == (6, 2) and b.final_done.shape == (6,) and b.disagreement.shape == (6,)
    assert np.array_equal(b.rewards[0], b.rewards[3])  # same actions + same seed -> same future
    assert (b.disagreement == 0).all()


# ------------------------------------------------------------ smoke + runner ---
def test_mpc_beats_random_quickly():
    seeds = VAL[:5]
    mpc = evaluate(MPCAgent(n_samples=30, horizon=5), DEFAULT_CONFIG, seeds)
    rnd = evaluate(RandomAgent(seed=0), DEFAULT_CONFIG, seeds)
    assert np.mean([r["return"] for r in mpc]) > np.mean([r["return"] for r in rnd])
    assert sum(r["success"] + r["partial"] for r in mpc) > sum(r["success"] + r["partial"] for r in rnd)


def test_parse_kwargs():
    assert parse_kwargs("n_samples=200,leaf=false,p_hint=0.5,name=x") == \
        {"n_samples": 200, "leaf": False, "p_hint": 0.5, "name": "x"}
    assert parse_kwargs("") == {}


def test_workers_match_serial():
    kw = {"seed": 0, "n_samples": 16, "horizon": 3}
    seeds = VAL[:6]
    one = run("mpc", kw, "default", seeds, workers=1)
    two = run("mpc", kw, "default", seeds, workers=2)
    skip = {"ms_per_decision", "workers"}
    strip = lambda rows: [{k: v for k, v in r.items() if k not in skip} for r in rows]  # noqa: E731
    assert [r["seed"] for r in two] == seeds
    assert strip(one) == strip(two)
    assert one[0]["N"] == 16 and one[0]["H"] == 3 and two[0]["workers"] == 2


def test_make_agent_mpc():
    assert isinstance(make_agent("mpc", n_samples=10), MPCAgent)
