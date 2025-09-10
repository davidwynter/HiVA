from hiva.tools.registry import ToolRegistry, ToolSpec
from hiva.tools.runtime import ToolRuntime

def test_registry_and_runtime():
    assert "calc" in ToolRegistry.list()
    out = ToolRuntime.run(ToolRegistry.get("calc"), {"task":"pow(2,5)"})
    assert out.strip() == "32.0"
    ToolRegistry.register(ToolSpec(name="echo", purpose="Echo upstream", func=lambda m: m.get("upstream","")))
    assert "echo" in ToolRegistry.list()
    eout = ToolRuntime.run(ToolRegistry.get("echo"), {"upstream":"x"})
    assert eout == "x"
