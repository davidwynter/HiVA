from __future__ import annotations
from typing import Tuple
from ..orchestrator.state import Task

class Environment:
    def evaluate(self, task: Task, answer: str) -> Tuple[float, str]:
        raise NotImplementedError

    def accuracy(self, task: Task, answer: str) -> float:
        raise NotImplementedError
