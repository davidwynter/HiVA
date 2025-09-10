from hiva.graph.model import AgentGraph
from hiva.agents.base import AgentBase, AggregatorAgent

def test_graph_add_agents_and_edges():
    G = AgentGraph()
    a = AgentBase(id="a")
    b = AgentBase(id="b")
    agg = AggregatorAgent(id="aggregator")
    for n in (a,b,agg):
        G.add_agent(n)
    G.add_edge("a","b")
    G.add_edge("b","aggregator")
    G.ensure_dag()
    assert "a" in G.nodes()
    assert "b" in G.nodes()
    assert "aggregator" in G.nodes()
    assert G.successors("a") == ["b"]
    assert G.predecessors("b") == ["a"]
    assert G.get_aggregator().id == "aggregator"
