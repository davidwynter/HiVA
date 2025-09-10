from __future__ import annotations
from typing import Any, Dict, Callable, List
from dataclasses import dataclass, field

@dataclass
class ToolSpec:
    name: str
    purpose: str
    func: Callable[[Dict[str, Any]], Any]

class ToolRegistry:
    _tools: Dict[str, ToolSpec] = {}

    @classmethod
    def register(cls, spec: ToolSpec) -> None:
        cls._tools[spec.name] = spec

    @classmethod
    def get(cls, name: str) -> ToolSpec:
        return cls._tools[name]

    @classmethod
    def list(cls) -> List[str]:
        return list(cls._tools.keys())
