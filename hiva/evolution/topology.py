from __future__ import annotations
from typing import Dict, Any
from ..graph.model import AgentGraph

def apply_topology_update(G: AgentGraph, aid: str, grad: Dict[str, Any]) -> bool:
    sugg = grad.get("suggestions", {})
    action = sugg.get("action")
    changed = False
    if action == "connect_to_aggregator":
        agg = G.get_aggregator()
        if aid != agg.id and not G.G.has_edge(aid, agg.id):
            G.add_edge(aid, agg.id)
            changed = True
    elif action == "remove_all_successors":
        for v in list(G.successors(aid)):
            G.remove_edge(aid, v)
            changed = True
    return changed

def repair_topology(G: AgentGraph) -> None:
    G.ensure_dag()
