from __future__ import annotations
from typing import List
from ..orchestrator.state import Task

def python_expr_dataset() -> List[Task]:
    exprs = ["1+1","2*3","10/2","sqrt(16)","sin(0)","cos(0)","pow(2,5)"]
    return [Task.from_text(e, requirements={"capabilities":["math","calc"]}) for e in exprs]
