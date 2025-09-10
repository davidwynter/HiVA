from __future__ import annotations
from ..graph.model import AgentGraph
import networkx as nx

def is_dag(G: AgentGraph) -> bool:
    return nx.is_directed_acyclic_graph(G.G)
