from __future__ import annotations
from typing import Dict, Any

def rewrite_prompt(current: str, gradient: Dict[str, Any]) -> str:
    """Apply a minimal but concrete textual update suggested by gradient."""
    add = gradient.get("suggestions", {}).get("add_phrase")
    remove = gradient.get("suggestions", {}).get("remove_phrase")
    newp = current
    if remove and remove in newp:
        newp = newp.replace(remove, "").strip()
    if add and add not in newp:
        newp = (newp + " " + add).strip()
    return newp
