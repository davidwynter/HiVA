from hiva.ui.inspect import snapshot_graph

def test_snapshot(simple_graph):
    snap = snapshot_graph(simple_graph)
    assert "nodes" in snap and "edges" in snap
    ids = [n["id"] for n in snap["nodes"]]
    assert "calc_agent" in ids and "aggregator" in ids
