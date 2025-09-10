from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time, uuid, random

@dataclass
class Task:
    id: str
    instruction: str
    context: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_text(text: str, requirements: Optional[Dict[str, Any]] = None) -> "Task":
        return Task(id=str(uuid.uuid4()), instruction=text, requirements=requirements or {})

@dataclass
class RunConfig:
    seed: int = 42
    max_successors: int = 2
    temperature: float = 1.0
    bandit_lambda: float = 0.5
    synergy_eta: float = 0.5
    knowledge_delta: float = 0.2
    decay_kappa: float = 0.05
    evolution_cooldown: int = 2
    budget_calls: Optional[int] = None

    # New: routing distance ablation
    distance_mode: str = "tag"  # "tag" or "kg"
    oxi_query_endpoint: Optional[str] = None
    oxi_update_endpoint: Optional[str] = None

@dataclass
class RunState:
    iteration: int = 0
    last_topology_change: int = -999
    token_cost: float = 0.0
    started_at: float = field(default_factory=time.time)

    def rng(self, seed: int) -> random.Random:
        return random.Random(seed)
