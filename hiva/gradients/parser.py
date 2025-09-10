from __future__ import annotations
from typing import Dict, Any

class SystemFeedbackParser:
    def __init__(self, tge) -> None:
        self.tge = tge

    def parse(self, final_text: str, env_feedback: str, loss: float) -> Dict[str, Any]:
        """Produce a global gradient dict."""
        scope = "semantic" if loss > 0.0 else "none"
        suggestions = {}
        if "incorrect" in env_feedback.lower() or loss > 0.0:
            suggestions["add_phrase"] = "Be explicit and verify arithmetic."
        return {"scope": scope, "suggestions": suggestions, "confidence": 0.6}

class AgentFeedbackParser:
    def __init__(self, tge) -> None:
        self.tge = tge

    def parse(self, agent_output: str, upstream: str | None, global_grad: Dict[str, Any]) -> Dict[str, Any]:
        """Compute a localized gradient for one agent."""
        scope = global_grad.get("scope", "none")
        sugg = dict(global_grad.get("suggestions", {}))
        # Simple rule: if agent produced empty text, request topology change
        if not agent_output.strip():
            scope = "topology"
            sugg = {"action": "connect_to_aggregator"}
        return {"scope": scope, "suggestions": sugg, "confidence": 0.5}
