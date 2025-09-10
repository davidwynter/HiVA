from hiva.evolution.semantic import apply_semantic_update
from hiva.evolution.topology import apply_topology_update
from hiva.graph.model import AgentGraph
from hiva.agents.base import AgentBase, AggregatorAgent

def test_semantic_and_topology_updates():
    G = AgentGraph()
    a = AgentBase(id="a", role_prompt="You are helpful.")
    agg = AggregatorAgent(id="aggregator")
    G.add_agent(a); G.add_agent(agg)
    changed = apply_semantic_update(G, "a", {"suggestions":{"add_phrase":"Be explicit."}, "scope":"semantic"})
    assert changed and "explicit" in G.agents["a"].role_prompt
    changed2 = apply_topology_update(G, "a", {"scope":"topology","suggestions":{"action":"connect_to_aggregator"}})
    assert changed2 and ("a","aggregator") in G.G.edges()
