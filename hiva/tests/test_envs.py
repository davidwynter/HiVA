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
