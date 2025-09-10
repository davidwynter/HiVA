from __future__ import annotations
from typing import Tuple
from ..orchestrator.state import Task
from .base import Environment

class SimpleMathEnv(Environment):
    """Assumes task.instruction is 'a+b' etc. Correctness by integer compare."""
    def evaluate(self, task: Task, answer: str) -> Tuple[float, str]:
        try:
            gold = str(eval(task.instruction, {"__builtins__": {}}, {}))
        except Exception as e:
            return 1.0, f"Task error: {e}"
        if answer.strip() == gold.strip():
            return 0.0, "Correct."
        return 1.0, f"Incorrect. Expected {gold}, got {answer}."
    def accuracy(self, task: Task, answer: str) -> float:
        loss, _ = self.evaluate(task, answer)
        return 1.0 - loss
