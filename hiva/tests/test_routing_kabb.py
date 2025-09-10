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
