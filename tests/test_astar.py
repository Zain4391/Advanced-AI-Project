import numpy as np

from agents import AStarAgent
from agents.astar_agent import astar
from env import SHOOT, UP, WAIT, EnvConfig, OutbreakEnv
from experiments.harness import evaluate
from tests.test_rules import QUIET, add_zombie, make_env


def test_astar_shortest_open_path():
    cost = np.ones((5, 5))
    path = astar((4, 0), {(0, 4)}, cost)
    assert len(path) == 8 and path[-1] == (0, 4)


def test_astar_respects_walls_and_unreachable():
    cost = np.ones((3, 3))
    cost[1, :] = np.inf
    assert astar((2, 0), {(0, 0)}, cost) is None
    cost[1, 2] = 1
    path = astar((2, 0), {(0, 0)}, cost)
    assert (1, 2) in path and (1, 0) not in path


def test_astar_detours_around_expensive_tile():
    cost = np.ones((3, 3))
    cost[1, 1] = 10  # straight route through the middle is expensive
    path = astar((1, 0), {(1, 2)}, cost)
    assert (1, 1) not in path and len(path) == 4


def test_agent_avoids_infection_when_detour_is_cheap():
    env = make_env()
    env.state.surv_pos[:] = [[2, 0], [7, 7]]
    env.state.infection[1, 0] = True     # straight up is infected
    agent = AStarAgent()
    a1, _ = agent.act(None, env)
    assert a1 != UP


def test_agent_shoots_zombie_in_sights():
    env = make_env()
    add_zombie(env, 5, 0)
    a1, _ = AStarAgent().act(None, env)
    assert a1 == SHOOT


def test_waits_on_closed_exit_and_partner_targets_neighbour():
    env = make_env()
    env.state.surv_pos[:] = [[0, 7], [3, 7]]   # S1 on closed exit
    agent = AStarAgent()
    a1, a2 = agent.act(None, env)
    assert a1 == WAIT and agent.paths[1][-1] in {(1, 7), (0, 6)}


def test_astar_always_wins_without_hazards():
    cfg = EnvConfig(n_zombies=0, infection_prob=0.0, n_initial_infection=0)
    rows = evaluate(AStarAgent(), cfg, range(50))
    assert all(r["success"] for r in rows), [r["seed"] for r in rows if not r["success"]]
