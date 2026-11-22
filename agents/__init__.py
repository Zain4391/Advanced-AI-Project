from .astar_agent import AStarAgent
from .base import Agent
from .random_agent import RandomAgent

# name -> class; used by experiments/run_benchmark.py and scripts/watch.py
AGENTS: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "astar": AStarAgent,
}


def make_agent(name: str, **kwargs) -> Agent:
    if name not in AGENTS:
        raise KeyError(f"unknown agent {name!r}; choose from {sorted(AGENTS)}")
    return AGENTS[name](**kwargs)


__all__ = ["Agent", "RandomAgent", "AStarAgent", "AGENTS", "make_agent"]
