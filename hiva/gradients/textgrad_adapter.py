from __future__ import annotations
from typing import Dict, Any

class TextGradEngine:
    """Lightweight adapter. If external textgrad is installed, you can extend this to call it.
    For now, it provides deterministic parsing helpers with simple heuristics.
    """
    def score(self, text: str) -> float:
        # crude readability/length proxy
        return min(1.0, max(0.0, len(text) / 2000.0))

    def choose(self, candidates: Dict[str, str]) -> str:
        # pick highest score text id
        best_id, _ = max(((cid, self.score(ct)) for cid, ct in candidates.items()), key=lambda x: x[1])
        return best_id
