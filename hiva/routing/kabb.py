from __future__ import annotations
from typing import List
import math, random
from ..graph.model import AgentGraph
from ..orchestrator.state import Task, RunConfig
from .distance import KnowledgeDistance
from .synergy import synergy_score

class KABBRouter:
    def __init__(self, G: AgentGraph, dist: KnowledgeDistance, cfg: RunConfig) -> None:
        self.G = G; self.dist = dist; self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    def choose_successors(self, u: str, task: Task) -> List[str]:
        succs = self.G.successors(u)
        if not succs:
            return []
        scored = []
        for v in succs:
            node = self.G.agents[v]
            a = getattr(node, "alpha", 1.0); b = getattr(node, "beta", 1.0)
            s = self.rng.betavariate(a, b)
            d = self.dist(node.meta, task.requirements)
            score = s * math.exp(-self.cfg.bandit_lambda * d)
            scored.append((score, v))
        scored.sort(reverse=True)
        return [v for _, v in scored[: self.cfg.max_successors]]

    def update_beliefs(self, trace, task: Task, loss: float, env_feedback: str) -> None:
        """Simple belief update: reward = 1 - normalized loss in [0,1]."""
        reward = max(0.0, min(1.0, 1.0 - loss))
        used = set([rec["agent"] for rec in trace.records])
        for aid, agent in self.G.agents.items():
            if not hasattr(agent, "alpha"): agent.alpha = 1.0
            if not hasattr(agent, "beta"): agent.beta = 1.0
            if aid in used:
                agent.alpha = 0.9 * agent.alpha + (0.1 * (reward + 1e-6))
                agent.beta  = 0.9 * agent.beta  + (0.1 * (1.0 - reward + 1e-6))
            else:
                # slight decay
                agent.alpha *= math.exp(-self.cfg.decay_kappa)
                agent.beta  *= math.exp(-self.cfg.decay_kappa)
