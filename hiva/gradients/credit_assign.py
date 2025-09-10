from __future__ import annotations
from typing import Dict, Any, List
from ..graph.model import AgentGraph
from .parser import AgentFeedbackParser

def backpropagate_gradients(G: AgentGraph, reverse_exec_order: List[str], g_global: Dict[str, Any],
                            produced_by: Dict[str, str], parser: AgentFeedbackParser) -> Dict[str, Dict[str, Any]]:
    local: Dict[str, Dict[str, Any]] = {}
    upstream_text = None
    for aid in reverse_exec_order:
        out = produced_by.get(aid, "")
        g = parser.parse(out, upstream_text, g_global)
        local[aid] = g
        upstream_text = out
    return local
