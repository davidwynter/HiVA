from __future__ import annotations
from dataclasses import dataclass

@dataclass
class CostAndAccuracy:
    total_cost: float = 0.0
    total_acc: float = 0.0
    n: int = 0
    def add(self, acc: float, cost: float) -> None:
        self.total_acc += acc; self.total_cost += cost; self.n += 1
    def summary(self) -> dict:
        return {"avg_acc": (self.total_acc / self.n) if self.n else 0.0, "total_cost": self.total_cost, "n": self.n}
