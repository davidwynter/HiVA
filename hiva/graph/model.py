from __future__ import annotations
from typing import Dict, Iterable, List
import networkx as nx
from hiva.agents.base import AgentBase, AggregatorAgent

class AgentGraph:
    """Directed acyclic graph of agents with attributes and edge synergy memory."""
    def __init__(self) -> None:
        self.G = nx.DiGraph()
        self.agents: Dict[str, AgentBase] = {}

    def add_agent(self, agent: AgentBase) -> None:
        self.agents[agent.id] = agent
        self.G.add_node(agent.id, kind=agent.kind)

    def add_edge(self, u: str, v: str) -> None:
        if u == v:
            return
        self.G.add_edge(u, v, c_syn=0.0, r_contrib=0.0, alpha=1.0, beta=1.0)

    def remove_edge(self, u: str, v: str) -> None:
        if self.G.has_edge(u, v):
            self.G.remove_edge(u, v)

    def predecessors(self, v: str) -> List[str]:
        return list(self.G.predecessors(v))

    def successors(self, v: str) -> List[str]:
        return list(self.G.successors(v))

    def nodes(self) -> Iterable[str]:
        return self.G.nodes()

    def ensure_dag(self) -> None:
        if not nx.is_directed_acyclic_graph(self.G):
            # simple repair: remove a back-edge found in any cycle
            for cycle in nx.simple_cycles(self.G.to_directed()):
                u, v = cycle[-1], cycle[0]
                if self.G.has_edge(u, v):
                    self.G.remove_edge(u, v)

    def get_aggregator(self) -> AggregatorAgent:
        aggs = [a for a in self.agents.values() if isinstance(a, AggregatorAgent)]
        if not aggs:
            raise RuntimeError("No aggregator agent in graph")
        return aggs[0]
