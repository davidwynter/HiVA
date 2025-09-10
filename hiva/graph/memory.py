from __future__ import annotations
from dataclasses import dataclass

@dataclass
class EdgeSyn:
    src: str
    dst: str
    c_syn: float = 0.0
    r_contrib: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
