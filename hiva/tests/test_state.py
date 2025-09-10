from hiva.orchestrator.state import Task, RunConfig, RunState

def test_task_and_config_flags():
    t = Task.from_text("2+2", requirements={"capabilities":["math"]})
    assert t.instruction == "2+2"
    cfg = RunConfig(distance_mode="tag")
    assert cfg.distance_mode in ("tag","kg")
    rs = RunState()
    assert rs.iteration == 0
