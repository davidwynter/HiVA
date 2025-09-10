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
