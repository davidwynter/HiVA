from __future__ import annotations
from typing import Dict, Any

class KnowledgeDistance:
    """Lightweight Dist(A_i, I_task) based on tag overlap.
    For production, replace with RDF/SPARQL scoring in knowledge.matching.
    """
    def __call__(self, agent_meta: Dict[str, Any], requirements: Dict[str, Any]) -> float:
        a_tags = set(agent_meta.get("capabilities", []))
        t_tags = set(requirements.get("capabilities", []))
        if not t_tags:
            return 0.0
        inter = len(a_tags & t_tags)
        return max(0.0, 1.0 - inter / max(1, len(t_tags)))
