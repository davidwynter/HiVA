from __future__ import annotations
from typing import Tuple
from ..orchestrator.state import Task
from .base import Environment

class AgenticStubEnv(Environment):
    """Agentic environment stub with deterministic scoring based on presence of a keyword."""
    def evaluate(self, task: Task, answer: str) -> Tuple[float, str]:
        key = task.context.get("keyword","")
        if key and key.lower() in answer.lower():
            return 0.0, "Correct keyword present."
        return 1.0, "Keyword missing."
    def accuracy(self, task: Task, answer: str) -> float:
        loss, _ = self.evaluate(task, answer)
        return 1.0 - loss
