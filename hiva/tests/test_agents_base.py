from hiva.agents.base import AgentBase, AggregatorAgent
from hiva.tools.registry import ToolRegistry

def test_agent_forward_and_aggregator():
    calc = AgentBase(id="calc", tools=[ToolRegistry.get("calc")])
    out, logs = calc.forward({"task":"2*3"})
    assert out.strip() == "6"
    fmt = AgentBase(id="fmt", tools=[ToolRegistry.get("uppercase")])
    out2, _ = fmt.forward({"upstream":"hello"})
    assert out2 == "HELLO"
    agg = AggregatorAgent(id="ag")
    best, meta = agg.aggregate(["a","abc","ab"])
    assert best == "abc"
