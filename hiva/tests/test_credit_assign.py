from hiva.gradients.credit_assign import backpropagate_gradients
from hiva.gradients.parser import AgentFeedbackParser
from hiva.gradients.textgrad_adapter import TextGradEngine

def test_backprop(simple_graph):
    order = ["calc_agent","format_agent","aggregator"]
    g_global = {"scope":"semantic","suggestions":{"add_phrase":"X"}}
    produced = {"calc_agent":"3","format_agent":"THREE","aggregator":"THREE"}
    local = backpropagate_gradients(simple_graph, order[::-1], g_global, produced, AgentFeedbackParser(TextGradEngine()))
    assert "calc_agent" in local and "format_agent" in local
