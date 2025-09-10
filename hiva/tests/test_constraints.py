from hiva.evolution.constraints import is_dag
from hiva.graph.model import AgentGraph
from hiva.agents.base import AgentBase, AggregatorAgent

def test_is_dag():
    G = AgentGraph()
    a = AgentBase(id="a")
    b = AgentBase(id="b")
    agg = AggregatorAgent(id="aggregator")
    for n in (a,b,agg):
        G.add_agent(n)
    G.add_edge("a","b")
    G.add_edge("b","aggregator")
    assert is_dag(G)
