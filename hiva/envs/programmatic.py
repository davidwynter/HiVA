from __future__ import annotations
from typing import Tuple
from ..orchestrator.state import Task
from .base import Environment
import math

class PythonExprEnv(Environment):
    """Tasks are Python expressions; correct answer is the eval result.
    Loss = 0 if exact string match, else 1. Feedback indicates mismatch.
    """
    def evaluate(self, task: Task, answer: str) -> Tuple[float, str]:
        try:
            allowed = {k: v for k, v in vars(math).items() if not k.startswith("_")}
            truth = str(eval(task.instruction, {"__builtins__": {}}, allowed))
        except Exception as e:
            return 1.0, f"Task evaluation error: {e}"
        if answer.strip() == truth.strip():
            return 0.0, "Correct."
        return 1.0, f"Incorrect. Expected {truth}, got {answer}."

    def accuracy(self, task: Task, answer: str) -> float:
        loss, _ = self.evaluate(task, answer)
        return 1.0 - loss
