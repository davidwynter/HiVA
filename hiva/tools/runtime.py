from __future__ import annotations
from typing import Any, Dict
from .registry import ToolSpec

class ToolRuntime:
    @staticmethod
    def run(spec: ToolSpec, message: Dict[str, Any]) -> Any:
        return spec.func(message)

# Built-in basic tools automatically registered
from .registry import ToolRegistry

def _calc_tool(msg: Dict[str, Any]) -> str:
    expr = str(msg.get("task") or msg.get("upstream") or msg.get("tool_output",""))
    try:
        # extremely restricted eval: only arithmetic
        allowed = {k: v for k, v in vars(__import__("math")).items() if not k.startswith("_")}
        return str(eval(expr, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"ERROR: {e}"

def _uppercase_tool(msg: Dict[str, Any]) -> str:
    text = str(msg.get("upstream") or msg.get("task") or msg.get("tool_output",""))
    return text.upper()

ToolRegistry.register(ToolSpec(name="calc", purpose="Evaluate arithmetic/math expressions.", func=_calc_tool))
ToolRegistry.register(ToolSpec(name="uppercase", purpose="Uppercase transformation.", func=_uppercase_tool))
