from __future__ import annotations
from typing import Dict, Any, List, Tuple
from .state import Task, RunConfig, RunState
from ..graph.model import AgentGraph
from ..graph.exec_subgraph import build_exec_order, ExecutionTrace
from ..agents.base import AgentBase, AggregatorAgent
from ..routing.kabb import KABBRouter
from ..gradients.textgrad_adapter import TextGradEngine
from ..gradients.parser import SystemFeedbackParser, AgentFeedbackParser
from ..gradients.credit_assign import backpropagate_gradients
from ..evolution.semantic import apply_semantic_update
from ..evolution.topology import apply_topology_update, repair_topology
from ..routing.distance import KnowledgeDistanceAblation
from ..envs.base import Environment
from ..eval.metrics import CostAndAccuracy
from ..ui.inspect import snapshot_graph

def _make_distance(cfg: RunConfig):
    if cfg.distance_mode == "kg":
        return KnowledgeDistanceAblation(
            mode="kg",
            query_endpoint=cfg.oxi_query_endpoint,
            update_endpoint=cfg.oxi_update_endpoint
        )
    return KnowledgeDistanceAblation(mode="tag")

def step(task: Task, G: AgentGraph, env: Environment, cfg: RunConfig, rs: RunState) -> Tuple[str, float, Dict[str, Any]]:
    exec_order = build_exec_order(G, task, cfg.max_successors)
    trace = ExecutionTrace(task_id=task.id)

    # Forward pass
    dist = _make_distance(cfg)
    router = KABBRouter(G, dist, cfg)
    produced_by: Dict[str, str] = {}
    messages: Dict[str, Any] = {}
    for aid in exec_order:
        agent: AgentBase = G.agents[aid]
        inp = messages.get(aid, {"task": task.instruction, "context": task.context})
        out_text, tool_logs = agent.forward(inp)
        trace.records.append({"agent": aid, "output": out_text, "tools": tool_logs})
        produced_by[aid] = out_text

        # route to successors
        succs = router.choose_successors(aid, task)
        for sid in succs:
            messages[sid] = {"upstream": out_text, "task": task.instruction, "context": task.context}

    # Aggregate
    aggregator: AggregatorAgent = G.get_aggregator()
    final_answer, agg_logs = aggregator.aggregate([produced_by[aid] for aid in G.predecessors(aggregator.id)])
    trace.final = final_answer

    # Environment evaluation
    loss, env_feedback = env.evaluate(task, final_answer)

    # Global textual gradient (system-level)
    tge = TextGradEngine()
    sys_parser = SystemFeedbackParser(tge)
    g_global = sys_parser.parse(final_answer, env_feedback, loss)

    # Local gradients (per-agent)
    ag_parser = AgentFeedbackParser(tge)
    local_grads = backpropagate_gradients(G, exec_order[::-1], g_global, produced_by, ag_parser)

    # Evolution
    changed = False
    for aid, grad in local_grads.items():
        if grad["scope"] in ("semantic","tool"):
            changed = apply_semantic_update(G, aid, grad) or changed
        if grad["scope"] == "topology":
            changed = apply_topology_update(G, aid, grad) or changed

    if changed:
        repair_topology(G)
        rs.last_topology_change = rs.iteration

    # Router post-update (bandit + synergy)
    router.update_beliefs(trace, task, loss, env_feedback)

    # Metrics
    snapshot = snapshot_graph(G)
    info = {"trace": trace.records, "env_feedback": env_feedback, "graph": snapshot, "changed": changed}
    return final_answer, loss, info

def run(task_list: List[Task], G: AgentGraph, env: Environment, cfg: RunConfig) -> Dict[str, Any]:
    rs = RunState()
    metrics = CostAndAccuracy()
    results = []
    for t in task_list:
        rs.iteration += 1
        ans, loss, info = step(t, G, env, cfg, rs)
        acc = env.accuracy(t, ans)
        metrics.add(acc=acc, cost=0.0)  # cost can be integrated if using external LLMs/tools
        results.append({"task": t.instruction, "answer": ans, "loss": loss, "acc": acc, "info": info})
    summary = metrics.summary()
    return {"results": results, "summary": summary}
