from __future__ import annotations
from typing import Tuple
from ..orchestrator.state import Task
from .base import Environment

class QAExactMatchEnv(Environment):
    """Question-answer environment with exact-match target in task.context['gold']"""
    def evaluate(self, task: Task, answer: str) -> Tuple[float, str]:
        gold = str(task.context.get("gold",""))
        if not gold:
            return 1.0, "No gold answer provided."
        if answer.strip().lower() == gold.strip().lower():
            return 0.0, "Correct."
        return 1.0, f"Incorrect. Expected {gold}, got {answer}."

    def accuracy(self, task: Task, answer: str) -> float:
        loss, _ = self.evaluate(task, answer)
        return 1.0 - loss
