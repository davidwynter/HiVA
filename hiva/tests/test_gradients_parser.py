from hiva.gradients.textgrad_adapter import TextGradEngine
from hiva.gradients.parser import SystemFeedbackParser, AgentFeedbackParser

def test_parsers():
    tge = TextGradEngine()
    gsys = SystemFeedbackParser(tge).parse("ans", "Incorrect. Expected 2, got 3.", 1.0)
    assert gsys["scope"] in ("semantic","none")
    glocal = AgentFeedbackParser(tge).parse("", None, gsys)
    assert glocal["scope"] in ("semantic","topology")
