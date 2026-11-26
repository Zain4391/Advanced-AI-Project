"""Rule tests on hand-built states (Requirements.md, Section 2 acceptance)."""
import numpy as np

from env import DOWN, LEFT, RIGHT, SHOOT, UP, WAIT, EnvConfig, EnvState, OutbreakEnv

# Quiet config: no random dynamics unless a test turns them on.
QUIET = EnvConfig(infection_prob=0.0, supply_interval=0, n_zombies=0)


def make_env(cfg=QUIET, **state_kw):
    env = OutbreakEnv(cfg)
    env.reset(seed=0)
    s = EnvState.empty(cfg.grid_size, cfg.max_zombies)
    s.gate_open = not cfg.coop_gate
    for k, v in state_kw.items():
        setattr(s, k, v)
    env.set_state(s)
    env.reseed_dynamics(0)
    return env


def add_zombie(env, r, c, slot=0):
    env.state.zombie_pos[slot] = (r, c)
    env.state.zombie_alive[slot] = True


def test_step_penalty_only():
    env = make_env()
    _, r, term, trunc, _ = env.step((WAIT, WAIT))
    assert r == -0.1 and not term and not trunc


def test_walls_block_movement():
    env = make_env()
    env.state.walls[6, 0] = True  # above survivor 1 at (7, 0)
    env.step((UP, WAIT))
    assert tuple(env.state.surv_pos[0]) == (7, 0)


def test_survivors_cannot_overlap_or_swap():
    env = make_env()
    env.step((RIGHT, LEFT))  # swap attempt
    assert tuple(env.state.surv_pos[0]) == (7, 0) and tuple(env.state.surv_pos[1]) == (7, 1)
    env.state.surv_pos[:] = [[7, 0], [7, 2]]
    env.step((RIGHT, LEFT))  # both into (7, 1)
    assert tuple(env.state.surv_pos[0]) == (7, 0) and tuple(env.state.surv_pos[1]) == (7, 2)


def test_infection_damage():
    env = make_env()
    env.state.infection[6, 0] = True
    _, r, *_ = env.step((UP, WAIT))
    assert env.state.health[0] == 4
    assert np.isclose(r, -0.1 - 1.0)


def test_zombie_adjacency_damage():
    env = make_env(EnvConfig(infection_prob=0.0, supply_interval=0, n_zombies=0, zombie_chase_prob=1.0))
    add_zombie(env, 5, 0)  # moves to (6, 0), adjacent to S1 at (7, 0)
    env.step((WAIT, WAIT))
    assert tuple(env.state.zombie_pos[0]) == (6, 0)
    assert env.state.health[0] == 4


def test_medkit_and_ammo_pickup():
    env = make_env()
    env.state.health[:] = [2, 2]
    env.state.surv_pos[1] = (3, 3)  # partner far away -> no shared heal
    env.state.medkits[6, 0] = True
    _, r, *_ = env.step((UP, WAIT))
    assert env.state.health[0] == 4 and env.state.health[1] == 2
    assert not env.state.medkits[6, 0]
    assert np.isclose(r, -0.1 + 1.0)
    env.state.ammo_crates[5, 0] = True
    env.step((UP, WAIT))
    assert env.state.ammo[0] == 2


def test_medkit_heals_both_when_adjacent():
    env = make_env()
    env.state.health[:] = [2, 2]
    env.state.medkits[6, 0] = True
    env.state.surv_pos[1] = (6, 1)
    env.step((UP, WAIT))
    assert list(env.state.health) == [4, 4]


def test_medkit_no_coop_when_rule_disabled():
    env = make_env(QUIET.with_(coop_medkit=False))
    env.state.health[:] = [2, 2]
    env.state.medkits[6, 0] = True
    env.state.surv_pos[1] = (6, 1)
    env.step((UP, WAIT))
    assert list(env.state.health) == [4, 2]


def test_health_capped():
    env = make_env()
    env.state.medkits[6, 0] = True
    env.step((UP, WAIT))
    assert env.state.health[0] == 5


def test_shoot_kills_and_costs_ammo():
    env = make_env()
    add_zombie(env, 5, 0)  # 2 tiles above S1
    _, r, *_ = env.step((SHOOT, WAIT))
    assert not env.state.zombie_alive[0]
    assert env.state.ammo[0] == 0
    assert np.isclose(r, -0.1 + 2.0)


def test_shoot_out_of_range_or_blocked():
    env = make_env()
    add_zombie(env, 4, 0)  # 3 tiles away
    env.step((SHOOT, WAIT))
    assert env.state.zombie_alive[0] and env.state.ammo[0] == 0
    env = make_env()
    add_zombie(env, 5, 0)
    env.state.walls[6, 0] = True
    env.step((SHOOT, WAIT))
    assert env.state.zombie_alive[0]


def test_shoot_without_ammo_does_nothing():
    env = make_env()
    env.state.ammo[0] = 0
    add_zombie(env, 6, 0)
    env.step((SHOOT, WAIT))
    assert env.state.zombie_alive[0]


def test_gate_needs_both_survivors():
    env = make_env()
    env.state.surv_pos[:] = [[0, 6], [5, 5]]  # S1 next to exit (0, 7), S2 far
    env.step((RIGHT, WAIT))
    s = env.state
    assert not s.gate_open and not s.evacuated[0]
    assert tuple(s.surv_pos[0]) == (0, 7)  # standing on the closed exit


def test_win_when_both_evacuate():
    env = make_env()
    env.state.surv_pos[:] = [[0, 6], [1, 7]]  # both adjacent to exit
    _, r, term, _, info = env.step((RIGHT, WAIT))  # S1 steps on; gate opens; S1 evacuates
    assert env.state.gate_open and env.state.evacuated[0] and not term
    _, r, term, _, info = env.step((WAIT, UP))
    assert term and info["won"]
    assert np.isclose(r, -0.1 + 20.0)


def test_gate_open_from_start_when_rule_disabled():
    env = make_env(QUIET.with_(coop_gate=False))
    env.state.surv_pos[:] = [[0, 6], [5, 5]]
    env.step((RIGHT, WAIT))
    assert env.state.evacuated[0]


def test_first_death_ends_episode_when_flag_set():
    env = make_env(QUIET.with_(end_on_first_death=True))
    env.state.health[0] = 1
    env.state.infection[6, 0] = True
    _, r, term, _, info = env.step((UP, WAIT))
    assert term and info["lost"] and env.state.health[0] == 0
    assert np.isclose(r, -0.1 - 1.0 - 20.0)


def test_one_death_does_not_end_episode():
    env = make_env()
    env.state.health[0] = 1
    env.state.infection[6, 0] = True
    _, r, term, trunc, info = env.step((UP, WAIT))
    assert not term and not trunc and not info["lost"]
    assert info["dead"] == [True, False] and info["events"]["deaths"] == [True, False]
    assert np.isclose(r, -0.1 - 1.0 - 10.0)
    assert not env.state.active[0] and env.state.active[1]


def test_dead_survivor_is_inert():
    env = make_env()
    env.state.health[0] = 0
    env.state.infection[7, 0] = True
    _, r, *_ = env.step((UP, WAIT))           # dead: can't move, takes no damage
    assert tuple(env.state.surv_pos[0]) == (7, 0) and env.state.health[0] == 0
    assert np.isclose(r, -0.1)
    env.state.surv_pos[1] = (7, 1)
    env.step((WAIT, LEFT))                    # bodies don't block the partner
    assert tuple(env.state.surv_pos[1]) == (7, 0)


def test_zombies_ignore_dead_survivor():
    env = make_env(QUIET.with_(zombie_chase_prob=1.0))
    env.state.health[0] = 0                   # body at (7, 0)
    env.state.surv_pos[1] = (0, 0)            # living partner far away, top-left
    add_zombie(env, 5, 0)
    env.step((WAIT, WAIT))
    assert tuple(env.state.zombie_pos[0]) == (4, 0)   # chases the living one (up), not the body


def test_both_dead_ends_episode():
    env = make_env()
    env.state.health[:] = [0, 1]
    env.state.surv_pos[1] = (7, 2)
    env.state.infection[6, 2] = True
    _, r, term, _, info = env.step((WAIT, UP))
    assert term and info["lost"] and not info["won"] and not info["partial"]
    assert np.isclose(r, -0.1 - 1.0 - 10.0)


def test_lone_survivor_can_escape_for_partial_win():
    env = make_env()
    env.state.health[0] = 0                   # partner dead: gate counts them as accounted for
    env.state.surv_pos[1] = (0, 6)
    _, r, term, _, info = env.step((WAIT, RIGHT))
    assert env.state.gate_open and env.state.evacuated[1]
    assert term and info["partial"] and not info["won"] and not info["lost"]
    assert np.isclose(r, -0.1 + 10.0)


def test_truncation_at_step_limit():
    env = make_env(QUIET.with_(max_steps=3))
    for _ in range(2):
        assert not env.step((WAIT, WAIT))[3]
    _, _, term, trunc, _ = env.step((WAIT, WAIT))
    assert trunc and not term


def test_infection_spreads_p1_and_not_p0():
    env = make_env(QUIET.with_(infection_prob=1.0))
    env.state.infection[3, 3] = True
    env.step((WAIT, WAIT))
    inf = env.state.infection
    assert inf[2, 3] and inf[4, 3] and inf[3, 2] and inf[3, 4] and inf.sum() == 5
    env = make_env()
    env.state.infection[3, 3] = True
    env.step((WAIT, WAIT))
    assert env.state.infection.sum() == 1


def test_infection_never_reaches_exit_or_walls():
    env = make_env(QUIET.with_(infection_prob=1.0))
    env.state.infection[0, 6] = True
    env.state.walls[1, 6] = True
    env.step((WAIT, WAIT))
    assert not env.state.infection[0, 7] and not env.state.infection[1, 6]


def test_supply_drop_every_k_steps():
    env = make_env(QUIET.with_(supply_interval=2))
    env.step((WAIT, WAIT))
    assert env.state.medkits.sum() + env.state.ammo_crates.sum() == 0
    env.step((WAIT, WAIT))
    assert env.state.medkits.sum() + env.state.ammo_crates.sum() == 1


def test_zombies_never_overlap():
    env = OutbreakEnv(EnvConfig(n_zombies=5))
    rng = np.random.default_rng(0)
    for seed in range(20):
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, t, tr, _ = env.step(tuple(rng.integers(6, size=2)))
            done = t or tr
            zp = env.state.zombie_pos[env.state.zombie_alive]
            assert len({tuple(p) for p in zp}) == len(zp)


def test_q_features_hashable_and_sized():
    env = OutbreakEnv()
    obs, _ = env.reset(seed=0)
    assert len(obs["agent_0"]) == 16
    {obs["agent_0"]: 0, obs["agent_1"]: 1}


def test_survivor2_shot_reports_target():
    env = make_env()
    add_zombie(env, 5, 1, slot=2)  # 2 tiles above S2 at (7, 1)
    _, _, _, _, info = env.step((WAIT, SHOOT))
    assert info["events"]["shot_targets"] == [-1, 2]
    assert not env.state.zombie_alive[2] and env.state.ammo[1] == 0


def test_miss_still_spends_ammo():
    env = make_env()
    _, _, _, _, info = env.step((WAIT, SHOOT))
    assert info["events"]["shots"] == [False, True] and info["events"]["shot_targets"] == [-1, -1]
    assert env.state.ammo[1] == 0

def test_build_obs_false_gives_identical_transitions():
    """step(..., build_obs=False) (used by planner rollouts) must not change the dynamics."""
    from env import DEFAULT_CONFIG

    a, b = OutbreakEnv(DEFAULT_CONFIG), OutbreakEnv(DEFAULT_CONFIG)
    a.reset(seed=1000)
    b.reset(seed=1000)
    rng = np.random.default_rng(0)
    for _ in range(60):
        act = (int(rng.integers(6)), int(rng.integers(6)))
        _, ra, ta, ua, _ = a.step(act)
        obs, rb, tb, ub, info = b.step(act, build_obs=False)
        assert obs is None and info is None and (ra, ta, ua) == (rb, tb, ub)
        for k in ("infection", "medkits", "ammo_crates", "surv_pos", "health", "ammo", "evacuated",
                  "zombie_pos", "zombie_alive"):
            assert np.array_equal(getattr(a.state, k), getattr(b.state, k)), k
        assert (a.state.t, a.state.gate_open, a.state.done) == (b.state.t, b.state.gate_open, b.state.done)
        assert a._rng.bit_generator.state == b._rng.bit_generator.state
        if ta or ua:
            break
