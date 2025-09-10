# Create a full pytest suite for the HiVA project.
# Tests are conservative (no external services required). KG mode is tested via mocks.
# A helper conftest builds a minimal graph and tasks.

import os, textwrap, pathlib, json, sys

BASE = pathlib.Path("/home/david/devtools/HiVA")
PKG = BASE / "hiva"
TESTS = BASE / "tests"
TESTS.mkdir(exist_ok=True)

def w(relpath: str, content: str):
    p = TESTS / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")

# -------- conftest --------
w("conftest.py", r"""
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
""")

# -------- orchestrator tests --------
w("test_orchestrator_loop.py", r"""
from hiva.orchestrator.loop import step, run
from hiva.orchestrator.state import Task, RunConfig, RunState
from hiva.envs.programmatic import PythonExprEnv

def test_step_and_run(simple_graph, run_cfg):
    env = PythonExprEnv()
    task = Task.from_text("1+1", requirements={"capabilities":["math","calc"]})
    rs = RunState()
    ans, loss, info = step(task, simple_graph, env, run_cfg, rs)
    assert isinstance(ans, str)
    assert loss in (0.0, 1.0)
    out = run([task], simple_graph, env, run_cfg)
    assert "summary" in out and "results" in out
    assert len(out["results"]) == 1
""")

w("test_state.py", r"""
from hiva.orchestrator.state import Task, RunConfig, RunState

def test_task_and_config_flags():
    t = Task.from_text("2+2", requirements={"capabilities":["math"]})
    assert t.instruction == "2+2"
    cfg = RunConfig(distance_mode="tag")
    assert cfg.distance_mode in ("tag","kg")
    rs = RunState()
    assert rs.iteration == 0
""")

# -------- graph tests --------
w("test_graph_model.py", r"""
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
""")

w("test_exec_subgraph.py", r"""
from hiva.graph.exec_subgraph import build_exec_order
from hiva.orchestrator.state import Task

def test_exec_order(simple_graph):
    order = build_exec_order(simple_graph, Task.from_text("1+1"), k=2)
    assert "aggregator" in order
    # Should include nodes with edges
    assert "calc_agent" in order and "format_agent" in order
""")

# -------- agents tests --------
w("test_agents_base.py", r"""
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
""")

w("test_semantics.py", r"""
from hiva.agents.semantics import rewrite_prompt

def test_rewrite_prompt_add_remove():
    p = "You are helpful."
    g = {"suggestions":{"add_phrase":"Be explicit.", "remove_phrase":"unrelated"}}
    p2 = rewrite_prompt(p, g)
    assert "Be explicit." in p2
    g2 = {"suggestions":{"remove_phrase":"helpful"}}
    p3 = rewrite_prompt(p2, g2)
    assert "helpful" not in p3
""")

# -------- routing tests --------
w("test_routing_distance.py", r"""
import pytest
from hiva.routing.distance import KnowledgeDistanceAblation

def test_tag_distance():
    dist = KnowledgeDistanceAblation(mode="tag")
    d = dist({"capabilities":["math","calc"]}, {"capabilities":["math"]})
    assert 0.0 <= d <= 1.0
    assert d == 0.0  # perfect overlap

def test_kg_distance_mock(monkeypatch):
    # mock KG client query to pretend there is always a match (count=1)
    class DummyClient:
        def query(self, sparql: str):
            return [{"c":"1"}]
    # Inject dummy into ablation object
    dist = KnowledgeDistanceAblation.__new__(KnowledgeDistanceAblation)
    from hiva.routing.distance import KnowledgeDistanceKG
    kg = KnowledgeDistanceKG.__new__(KnowledgeDistanceKG)
    kg.client = DummyClient()
    dist.impl = kg
    d = dist({"capabilities_uri":["http://example.org/Calc"]},
             {"capabilities_uri":["http://example.org/Math"]})
    assert d == 0.0  # overlap=1.0 -> distance 0.0

def test_kg_requires_endpoint():
    with pytest.raises(ValueError):
        KnowledgeDistanceAblation(mode="kg")
""")

w("test_routing_kabb.py", r"""
from hiva.routing.kabb import KABBRouter
from hiva.routing.distance import KnowledgeDistanceAblation
from hiva.orchestrator.state import Task, RunConfig

def test_kabb_choose_and_update(simple_graph):
    dist = KnowledgeDistanceAblation(mode="tag")
    cfg = RunConfig()
    router = KABBRouter(simple_graph, dist, cfg)
    succs = router.choose_successors("calc_agent", Task.from_text("1+1", requirements={"capabilities":["math"]}))
    assert isinstance(succs, list)
    # Update beliefs after a fake trace
    class Trace: records=[{"agent":"calc_agent"}]
    router.update_beliefs(Trace(), Task.from_text("1+1"), loss=0.0, env_feedback="Correct.")
    assert hasattr(simple_graph.agents["calc_agent"], "alpha")
""")

# -------- gradients tests --------
w("test_gradients_parser.py", r"""
from hiva.gradients.textgrad_adapter import TextGradEngine
from hiva.gradients.parser import SystemFeedbackParser, AgentFeedbackParser

def test_parsers():
    tge = TextGradEngine()
    gsys = SystemFeedbackParser(tge).parse("ans", "Incorrect. Expected 2, got 3.", 1.0)
    assert gsys["scope"] in ("semantic","none")
    glocal = AgentFeedbackParser(tge).parse("", None, gsys)
    assert glocal["scope"] in ("semantic","topology")
""")

w("test_credit_assign.py", r"""
from hiva.gradients.credit_assign import backpropagate_gradients
from hiva.gradients.parser import AgentFeedbackParser
from hiva.gradients.textgrad_adapter import TextGradEngine

def test_backprop(simple_graph):
    order = ["calc_agent","format_agent","aggregator"]
    g_global = {"scope":"semantic","suggestions":{"add_phrase":"X"}}
    produced = {"calc_agent":"3","format_agent":"THREE","aggregator":"THREE"}
    local = backpropagate_gradients(simple_graph, order[::-1], g_global, produced, AgentFeedbackParser(TextGradEngine()))
    assert "calc_agent" in local and "format_agent" in local
""")

# -------- evolution tests --------
w("test_evolution.py", r"""
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
""")

w("test_constraints.py", r"""
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
""")

# -------- tools tests --------
w("test_tools.py", r"""
from hiva.tools.registry import ToolRegistry, ToolSpec
from hiva.tools.runtime import ToolRuntime

def test_registry_and_runtime():
    assert "calc" in ToolRegistry.list()
    out = ToolRuntime.run(ToolRegistry.get("calc"), {"task":"pow(2,5)"})
    assert out.strip() == "32"
    ToolRegistry.register(ToolSpec(name="echo", purpose="Echo upstream", func=lambda m: m.get("upstream","")))
    assert "echo" in ToolRegistry.list()
    eout = ToolRuntime.run(ToolRegistry.get("echo"), {"upstream":"x"})
    assert eout == "x"
""")

# -------- knowledge tests (mocked) --------
w("test_knowledge_matching.py", r"""
from hiva.knowledge.matching import kg_distance

class DummyClient:
    def query(self, sparql: str):
        # Return count=1 to simulate a match
        return [{"c":"1"}]

def test_kg_distance_mock():
    client = DummyClient()
    d = kg_distance(client,
                    {"capabilities_uri":["http://example.org/Calc"]},
                    {"capabilities_uri":["http://example.org/Math"]})
    assert d == 0.0
""")

# -------- envs tests --------
w("test_envs.py", r"""
from hiva.envs.programmatic import PythonExprEnv
from hiva.envs.qa import QAExactMatchEnv
from hiva.envs.math import SimpleMathEnv
from hiva.orchestrator.state import Task

def test_programmatic_env():
    env = PythonExprEnv()
    t = Task.from_text("sqrt(16)")
    loss, fb = env.evaluate(t, "4.0")
    assert loss == 0.0
    assert env.accuracy(t, "4.0") == 1.0

def test_qa_env():
    env = QAExactMatchEnv()
    t = Task(id="x", instruction="Who?", context={"gold":"Alice"}, requirements={})
    loss, fb = env.evaluate(t, "alice")
    assert loss == 0.0

def test_math_env():
    env = SimpleMathEnv()
    t = Task.from_text("2+3")
    loss, _ = env.evaluate(t, "5")
    assert loss == 0.0
""")

# -------- eval & ui tests --------
w("test_eval_and_metrics.py", r"""
from hiva.eval.datasets import python_expr_dataset
from hiva.eval.metrics import CostAndAccuracy

def test_dataset_and_metrics():
    ds = python_expr_dataset()
    assert len(ds) >= 3
    m = CostAndAccuracy()
    m.add(acc=1.0, cost=0.0)
    s = m.summary()
    assert s["avg_acc"] > 0.0 and s["n"] == 1
""")

w("test_ui_inspect.py", r"""
from hiva.ui.inspect import snapshot_graph

def test_snapshot(simple_graph):
    snap = snapshot_graph(simple_graph)
    assert "nodes" in snap and "edges" in snap
    ids = [n["id"] for n in snap["nodes"]]
    assert "calc_agent" in ids and "aggregator" in ids
""")

print("Test suite written to /home/david/devtools/hiva/tests")

