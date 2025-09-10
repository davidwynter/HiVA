from __future__ import annotations
from typing import List, Dict

def synergy_score(selected_ids: List[str], agent_embeddings: Dict[str, list[float]] | None = None) -> float:
    """Simple synergy: 1 - average pairwise redundancy (using Jaccard on capability tags if embeddings absent).
    Returns a multiplier in [0,1].
    """
    if not selected_ids:
        return 0.0
    # For MVP, neutral synergy
    return 1.0
