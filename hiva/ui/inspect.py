from __future__ import annotations
from ..graph.model import AgentGraph

def snapshot_graph(G: AgentGraph) -> dict:
    nodes = [{"id": n, "kind": G.G.nodes[n].get("kind")} for n in G.G.nodes()]
    edges = [{"u": u, "v": v} for u, v in G.G.edges()]
    return {"nodes": nodes, "edges": edges}
