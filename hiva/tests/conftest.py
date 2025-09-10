import pytest
from hiva.graph.model import AgentGraph
from hiva.agents.base import AgentBase, AggregatorAgent
from hiva.tools.registry import ToolRegistry
from hiva.orchestrator.state import Task, RunConfig

@pytest.fixture
def simple_graph():
    G = AgentGraph()
    a1 = AgentBase(id="calc_agent", role_prompt="Calculator agent.",
                   tools=[ToolRegistry.get("calc")],
                   meta={"capabilities":["math","calc"],
                         "capabilities_uri":["http://example.org/Math","http://example.org/Calc"]})
    a2 = AgentBase(id="format_agent", role_prompt="Formatter agent.",
                   tools=[ToolRegistry.get("uppercase")],
                   meta={"capabilities":["format"],
                         "capabilities_uri":["http://example.org/Format"]})
    agg = AggregatorAgent(id="aggregator", role_prompt="Aggregator.")
    for a in (a1, a2, agg):
        G.add_agent(a)
    G.add_edge("calc_agent","format_agent")
    G.add_edge("calc_agent","aggregator")
    G.add_edge("format_agent","aggregator")
    G.ensure_dag()
    return G

@pytest.fixture
def run_cfg():
    return RunConfig()

@pytest.fixture
def math_tasks():
    from hiva.eval.datasets import python_expr_dataset
    return python_expr_dataset()
