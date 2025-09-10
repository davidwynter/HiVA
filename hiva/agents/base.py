from __future__ import annotations
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass, field
from ..tools.runtime import ToolRuntime
from ..tools.registry import ToolSpec, ToolRegistry

@dataclass
class AgentBase:
    id: str
    kind: str = "agent"
    role_prompt: str = "You are a helpful agent."
    tools: List[ToolSpec] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def forward(self, message: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        """Run tools sequentially if provided, else echo upstream or task."""
        logs: List[Dict[str, Any]] = []
        if self.tools:
            output = ""
            for t in self.tools:
                res = ToolRuntime.run(t, message)
                logs.append({"tool": t.name, "result": str(res)[:200]})
                output = str(res)
                # cascade output to next tool as message
                message = {"tool_output": output, **message}
            return output, logs
        # default behavior
        if "upstream" in message:
            return str(message["upstream"]), logs
        return str(message.get("task","")), logs

class AggregatorAgent(AgentBase):
    kind: str = "aggregator"
    def aggregate(self, candidates: List[str]) -> Tuple[str, Dict[str, Any]]:
        # simple heuristic: longest string wins (proxy for richness)
        if not candidates:
            return "", {"policy": "empty"}
        best = max(candidates, key=lambda s: len(s))
        return best, {"policy": "length_max"}
