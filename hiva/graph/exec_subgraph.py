from __future__ import annotations
from typing import List
from dataclasses import dataclass, field
from ..orchestrator.state import Task, RunConfig
from .model import AgentGraph
import networkx as nx

@dataclass
class ExecutionTrace:
    task_id: str
    records: list = field(default_factory=list)
    final: str | None = None

def build_exec_order(G: AgentGraph, task: Task, k: int) -> List[str]:
    """Return a topological order of nodes excluding isolated ones except aggregator.
    For MVP we just use a global topo order; routing determines actual message flow.
    """
    G.ensure_dag()
    topo = list(nx.topological_sort(G.G))
    # keep nodes that have either incoming or outgoing edges, plus aggregator
    agg_id = G.get_aggregator().id
    filtered = [n for n in topo if (G.G.in_degree(n) + G.G.out_degree(n) > 0) or n == agg_id]
    return filtered
