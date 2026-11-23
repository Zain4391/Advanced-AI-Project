import numpy as np

from agents import QLearningAgent, make_agent
from env import WAIT, OutbreakEnv
from experiments.train_qlearning import epsilon_at


def test_update_is_q_learning_backup():
    ag = QLearningAgent(alpha=0.5, gamma=0.9)
    s, s2 = (1,), (2,)
    ag._q(0, s2, create=True)[3] = 10.0
    ag.update(0, s, 1, 1.0, s2, done=False)
    assert np.isclose(ag._q(0, s)[1], 0.5 * (1.0 + 0.9 * 10.0))
    ag.update(0, s, 2, 4.0, s2, done=True)      # terminal: no bootstrap
    assert np.isclose(ag._q(0, s)[2], 2.0)


def test_learners_are_independent_unless_shared():
    ag = QLearningAgent()
    ag.update(0, (1,), 0, 5.0, (1,), done=True)
    assert ag._q(1, (1,)) is None               # survivor 2 never saw that state
    shared = QLearningAgent(shared_table=True)
    shared.update(0, (1,), 0, 5.0, (1,), done=True)
    assert shared._q(1, (1,)) is not None


def test_greedy_in_eval_and_deterministic():
    env = OutbreakEnv()
    obs, _ = env.reset(seed=3)
    ag = QLearningAgent()
    for i in (0, 1):
        ag._q(i, ag.feats(obs, env, i), create=True)[2] = 1.0
    ag.reset(3)
    assert ag.act(obs, env) == (2, 2)
    fresh_a, fresh_b = QLearningAgent(seed=1), QLearningAgent(seed=1)
    fresh_a.reset(7)
    fresh_b.reset(7)
    assert fresh_a.act(obs, env) == fresh_b.act(obs, env)   # random tie-break is seeded


def test_inactive_survivor_waits():
    env = OutbreakEnv()
    obs, _ = env.reset(seed=0)
    env.state.health[1] = 0
    assert QLearningAgent().act(obs, env)[1] == WAIT


def test_save_load_roundtrip(tmp_path):
    ag = QLearningAgent(seed=4)
    ag.update(0, (1, 2), 3, 7.0, (1, 2), done=True)
    p = tmp_path / "q.pkl"
    ag.save(p, episodes=1)
    loaded = make_agent("qlearning", checkpoint=str(p))
    assert np.allclose(loaded._q(0, (1, 2)), ag._q(0, (1, 2)))
    assert loaded.meta["episodes"] == 1


def test_epsilon_schedule():
    assert epsilon_at(0, 100, 1.0, 0.05, 0.8) == 1.0
    assert np.isclose(epsilon_at(80, 100, 1.0, 0.05, 0.8), 0.05)
    assert np.isclose(epsilon_at(100, 100, 1.0, 0.05, 0.8), 0.05)


def test_compact_features():
    from env import EnvConfig
    from env.state_encoding import q_features_compact
    from tests.test_rules import QUIET, add_zombie, make_env
    env = make_env()                              # S1 (7,0), S2 (7,1), exit (0,7)
    env.state.infection[6, 0] = True
    f = q_features_compact(env.state, 0, env.config)
    assert len(f) == 13
    assert f[1:5] == (1, 2, 2, 2)                 # up: infected; down/left: edge; right: partner
    add_zombie(env, 5, 1)                         # 2 tiles above S2
    f2 = q_features_compact(env.state, 1, env.config)
    assert f2[6] == 1 and (f2[7], f2[8]) == (-1, 0)   # can shoot, zombie straight up
    env.state.health[0] = 0
    assert q_features_compact(env.state, 1, env.config)[11] == 2   # partner gone
    assert len({q_features_compact(env.state, 1, env.config)}) == 1  # hashable


def test_learns_to_escape_without_hazards():
    """Regression test for the learning loop + shaping: with no zombies or infection, two
    independent learners must learn to escape together (greedy policy, unseen val maps)."""
    from env import EnvConfig
    from experiments.seeds import VAL
    from experiments.train_qlearning import potential, train

    cfg = EnvConfig(n_zombies=0, infection_prob=0.0, n_initial_infection=0)
    _, curve, _ = train(cfg, episodes=1000, eps_decay_frac=0.5, eval_every=1000,
                        eval_seeds=VAL[:10], shaping=10, verbose=False)
    assert curve[-1]["eval_success"] >= 0.3

    env = OutbreakEnv(cfg)
    env.reset(seed=0)
    assert 0.0 <= potential(env, 0) <= 1.0     # non-negative potential (no reward for standing still)


# ---------------------------------------------------------------- fixes (Nov 24) ---
def test_inactive_survivor_keeps_a_distinct_absorbing_key():
    """An evacuated or dead survivor must keep learning from the shared reward, so its key says
    why it is inactive and what the partner is doing (otherwise the +20 win never reaches it)."""
    from env.state_encoding import q_features_compact
    from tests.test_rules import make_env
    env = make_env()
    env.state.evacuated[0] = True
    evac = q_features_compact(env.state, 0, env.config)
    assert evac[0] == WAIT and evac[5] == 2 and evac[11] == 0     # evacuated, partner far from exit
    env.state.evacuated[0] = False
    env.state.health[0] = 0
    dead = q_features_compact(env.state, 0, env.config)
    assert dead[5] == 3 and dead != evac
    env.state.evacuated[1] = True
    assert q_features_compact(env.state, 0, env.config)[11] == 2   # partner gone too


def test_update_bootstraps_from_wait_for_inactive_next_state():
    ag = QLearningAgent(alpha=1.0, gamma=0.5)
    s, s2 = (1,), (2,)
    q2 = ag._q(0, s2, create=True)
    q2[WAIT] = -1.0                       # the only action the survivor can take there
    ag.update(0, s, 0, 0.0, s2, done=False, next_forced=WAIT)
    assert np.isclose(ag._q(0, s)[0], -0.5)       # gamma * Q(s2, WAIT), not gamma * max(...) = 0


def test_shoot_is_masked_without_a_target():
    from env.config import SHOOT
    ag = QLearningAgent(features="compact", seed=0)
    ag.training, ag.epsilon = True, 1.0
    no_target = (2,) + (0,) * 12                  # can_shoot slot (6) = 0
    assert all(ag._choose(0, no_target) != SHOOT for _ in range(300))
    ag._q(0, no_target, create=True)[SHOOT] = 5.0   # even a (bogus) high value cannot be picked
    ag.epsilon = 0.0
    assert ag._choose(0, no_target) != SHOOT
    can = (2, 0, 0, 0, 0, 0, 1) + (0,) * 6
    ag.epsilon = 1.0
    assert any(ag._choose(0, can) == SHOOT for _ in range(300))


def test_hint_prior_breaks_ties_toward_the_exit():
    ag = QLearningAgent(hint_prior=0.5, seed=0)
    key = (2,) + (0,) * 12
    assert np.isclose(ag._q(0, key, create=True)[2], 0.5)
    assert ag._choose(0, (1,) + (0,) * 12) == 1   # never-visited state: play the hint direction
    assert QLearningAgent(hint_prior=0.5)._q(0, (4,) + (0,) * 12, create=True).max() == 0.0


def test_learning_rate_decays_per_state_action():
    ag = QLearningAgent(alpha=0.5, alpha_decay=1.0, alpha_min=0.1)
    ag.update(0, (1,), 0, 1.0, (1,), done=True)   # alpha = 0.5
    assert np.isclose(ag._q(0, (1,))[0], 0.5)
    ag.update(0, (1,), 0, 1.0, (1,), done=True)   # alpha = 0.5 / 2
    assert np.isclose(ag._q(0, (1,))[0], 0.625)


def test_compact_t_features_and_time_bucket():
    from env.state_encoding import q_features_compact
    from tests.test_rules import make_env
    env = make_env()
    f = q_features_compact(env.state, 0, env.config, extended=True)
    assert len(f) == 15 and f[:13] == q_features_compact(env.state, 0, env.config)
    assert f[14] == 0
    env.state.t = 50
    assert q_features_compact(env.state, 0, env.config, extended=True)[14] == 1
    env.state.t = 90
    assert q_features_compact(env.state, 0, env.config, extended=True)[14] == 2


def test_curriculum_smoke():
    from env import EnvConfig
    from experiments.train_qlearning import train_curriculum
    agent, curve, _ = train_curriculum(
        EnvConfig(), episodes=5, scale=0.002, eval_every=1000, features="compact_t",
        hint_prior=0.5, alpha_decay=0.02, shaping=10, verbose=False)
    assert {r["stage"] for r in curve} == {"no_hazards", "infection", "zombies", "full"}
    assert agent.features == "compact_t" and sum(agent.table_sizes) > 0


def test_checkpoint_keeps_behaviour_flags(tmp_path):
    """A loaded agent must act like the trained one: features, hint prior, SHOOT mask, alpha decay."""
    from env.config import SHOOT
    ag = QLearningAgent(features="compact_t", hint_prior=0.5, alpha_decay=0.02, seed=1)
    ag.update(0, (2,) + (0,) * 14, 2, 1.0, (2,) + (0,) * 14, done=True)
    p = tmp_path / "q.pkl"
    ag.save(p)
    loaded = QLearningAgent(checkpoint=str(p))
    assert (loaded.features, loaded.hint_prior, loaded.alpha_decay) == ("compact_t", 0.5, 0.02)
    assert loaded._choose(0, (3,) + (0,) * 14) == 3           # unseen state: the hint direction
    assert loaded._legal((3,) + (0,) * 14).tolist() == [0, 1, 2, 3, 4]    # SHOOT masked (no target)
    old = tmp_path / "old.pkl"                                 # a checkpoint without the new keys
    import pickle
    payload = pickle.load(open(p, "rb"))
    for k in ("hint_prior", "mask_shoot", "alpha_min", "alpha_decay", "visits"):
        payload.pop(k)
    pickle.dump(payload, open(old, "wb"))
    legacy = QLearningAgent(checkpoint=str(old))
    assert legacy.hint_prior == 0.0 and not legacy.mask_shoot
