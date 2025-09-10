from hiva.graph.exec_subgraph import build_exec_order
from hiva.orchestrator.state import Task

def test_exec_order(simple_graph):
    order = build_exec_order(simple_graph, Task.from_text("1+1"), k=2)
    assert "aggregator" in order
    # Should include nodes with edges
    assert "calc_agent" in order and "format_agent" in order
