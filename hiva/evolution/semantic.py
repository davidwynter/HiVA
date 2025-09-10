from __future__ import annotations
from typing import Dict, Any
from ..graph.model import AgentGraph
from ..agents.semantics import rewrite_prompt

def apply_semantic_update(G: AgentGraph, aid: str, grad: Dict[str, Any]) -> bool:
    agent = G.agents.get(aid)
    if not agent:
        return False
    newp = rewrite_prompt(agent.role_prompt, grad)
    changed = (newp != agent.role_prompt)
    agent.role_prompt = newp
    return changed
