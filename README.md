# HiVA
An implementation of the paper https://arxiv.org/pdf/2509.00189

Below is a complete, implementation-ready **software architecture plan** for building the paper’s system (“HiVA: Self-organized Hierarchical Variable Agent via Goal-driven Semantic-Topological Evolution”) in Python, with precise module boundaries, data models, algorithms, and an MVP→full-stack rollout. Where I reference the paper’s specific mechanisms (STEV loop, KABB routing, semantic/topological updates, prompt templates), I cite the uploaded PDF.&#x20;

---

# 1) High-level system design

**Goal.** Implement *Semantic-Topological Evolution (STEV)*: at each iteration, dynamically route a task through a subgraph of agents, obtain environment feedback, convert it into **textual gradients**, and **co-evolve** (a) agent semantics (prompts + tool loadouts) and (b) the graph topology itself.&#x20;

**Key subsystems (services/libraries):**

1. **Graph Engine** — keeps the multi-agent DAG `G=(V,E)`; constructs execution subgraphs; enforces acyclicity; maintains edge synergy/memory. (Alg. 2 FORWARDPASS; “RepairTopology”, synergy weights).&#x20;
2. **Routing (KABB)** — Knowledge-Aware Bayesian Bandit (Thompson sampling) to pick successors per node using α/β beliefs + task relevance + team synergy and a knowledge-based cost `Dist(·)`; handles bandit updates with decay.&#x20;
3. **Agents** — pluggable LLM agents with (system prompt, tool registry, successor policy hooks); includes an **Aggregator** sink agent. Supports forward messages and backward localized textual gradients. (Alg. 1 & 2; prompt templates in Appendix A).&#x20;
4. **Textual Gradient Feedback** — parses environment outcomes into structured, local update signals, implements a textual “chain rule” to distribute gradients across the execution subgraph; integrates **TextGrad** for LLM-based gradient parsing/optimization.&#x20;
5. **Evolution** — `fP` (semantic evolution: update system prompts + tools), `fG` (topological evolution: add/delete successor, connect to aggregator, or no-op) under constraints to remain a DAG.&#x20;
6. **Knowledge Layer** — capability ontology + task schema in RDF; computes Dist(·) between task requirements and agent capabilities; stores synergy history as graph memory.&#x20;
7. **Environments** — unified interface over programmatic tests, QA datasets, agentic tools (browser/IDE), etc.; returns both scalar losses and **verbal feedback**.&#x20;
8. **Tooling Runtime** — structured tool schemas, governed execution (timeouts, sandbox), versioning, and feedback-driven evolution of tool code. (Appendix A.3)&#x20;
9. **Observability & Reproducibility** — run registry, graph snapshots, seed control, cost/accuracy metrics, MLflow/W\&B logging; replayable traces.
10. **Orchestrator** — the STEV loop controller coordinating (Forward → Environment → Textual gradients → Evolution → Router updates) per iteration `t=1..T`.&#x20;

---

# 2) Technology choices (Python)

* **Agents/LLM access**: OpenAI API (or your local model wrapper), `litellm` or direct SDK.
* **Textual gradients**: **TextGrad** (zou-group/textgrad) for gradient parsing/evaluation and programmatic optimization loops; extend with custom “gradient parsers” reflecting the paper’s prompts.
* **Graph**: `networkx` for DAG + custom attributes; or `retworkx` for speed.
* **Bandits**: `numpy`/`scipy` for Beta sampling; light custom KABB implementation.
* **Knowledge graph**: `rdflib` client + **Oxigraph** or your existing RDF store; SPARQL for Dist(·).
* **Tools**: Pydantic schemas + subprocess sandbox (or uvicorn workers); `tenacity` for retries/timeouts.
* **Eval**: `datasets`, `jsonlines`, `pytest` style harnesses.
* **Logging & runs**: `mlflow` or `wandb`; `structlog` for JSON logs.
* **Persistence**: `sqlite`/`duckdb` + Parquet for runs; or Postgres if multi-user.

---

# 3) Package layout

```
hiva/
  __init__.py
  orchestrator/              # STEV controller
    loop.py                  # Algorithm 1 wrapper
    state.py                 # run cfg, seeds, checkpoints
  graph/
    model.py                 # G=(V,E), DAG ops, RepairTopology
    exec_subgraph.py         # Alg.2 construction & trace
    memory.py                # edge synergy C_syn, R_ij history
  routing/
    kabb.py                  # Thompson sampling, α/β updates, decay
    distance.py              # Dist(Ai, Itask) via KG & task schema
    synergy.py               # ζ(S_t) and pairwise C_syn computation
  agents/
    base.py                  # Agent, Aggregator (sink), Source (vs)
    semantics.py             # prompts, system roles, tool loadouts
    forward_msgs.py          # successor instruction generation (Fig.6)
    backward_msgs.py         # system/agent feedback prompts (Fig.7,8)
  gradients/
    textgrad_adapter.py      # TextGrad wrappers (global→local gradients)
    parser.py                # parse feedback → Δs_t JSON (fP vs fG)
    credit_assign.py         # textual chain rule across subgraph
  evolution/
    semantic.py              # fP: prompt/tool updates (Fig.9)
    topology.py              # fG: add/del/connect→a/no-op (Fig.10)
    constraints.py           # DAG checks, degree limits, cooldowns
  tools/
    registry.py              # schema, versions, perf metrics
    runtime.py               # governed execution, sandbox
    evolution.py             # tool synthesis/refinement (Fig.12,13)
  knowledge/
    ontology/
      capabilities.ttl       # agent capability classes/properties
      tasks.ttl              # task requirement schema
    kg_client.py             # SPARQL queries to Oxigraph
    matching.py              # Ψ_k sub-indicators, KM/KD
  envs/
    base.py                  # Env interface
    programmatic.py          # unit-test/code eval (e.g., MBPP)
    qa.py                    # HotpotQA/2Wiki adapters
    agentic.py               # browser/IDE stubs
    math.py                  # evaluator producing textual feedback
  eval/
    datasets.py              # loaders, split seeds, samplers
    metrics.py               # accuracy, cost-efficiency score (CS)
    runner.py                # multi-iteration benchmarks
  ui/
    inspect.py               # ASCII/graphviz snapshots, diffs
  configs/
    defaults.yaml            # temperatures, top-k, decay κ, λ, η, ρ
```

(Prompt templates referenced above are from Appendix A: forward, backward, semantic update, topology decision, tool genesis/patch.)&#x20;

---

# 4) Core data models

**Agent** (`agents/base.py`)

```python
class AgentSpec(BaseModel):
    id: str
    role_prompt: str              # semantic parameter
    tools: list[ToolSpec]         # semantic parameter
    alpha: float = 1.0            # KABB belief
    beta: float = 1.0             # KABB belief
    last_used: float | None = None
    meta: dict = {}
```

**Edge Memory / Synergy** (`graph/memory.py`)

```python
class EdgeSyn(BaseModel):
    src: str; dst: str
    c_syn: float = 0.0            # C_syn^(t)
    r_contrib: float = 0.0        # R_ij^(t)
    alpha: float = 1.0            # optional per-edge beliefs
    beta: float = 1.0
```

**Task** (`orchestrator/state.py`)

```python
class Task(BaseModel):
    id: str
    instruction: str
    context: dict | None = None
    requirements: dict            # for Dist(·) KG matching
```

**Gradient Payloads** (`gradients/parser.py`)

```python
class TextualGradient(BaseModel):
    scope: Literal["semantic","topology","tool","routing","global"]
    target_id: str
    suggestions: dict             # structured edits / decisions
    confidence: float
```

---

# 5) Algorithms & execution

## 5.1 Forward Pass (Alg. 2)

* Topologically order `G`; for each visited agent:

  * If it has incoming instruction(s), call its toolchain (optional), produce output `y_i`.
  * **Routing**: sample successor scores via Thompson sampling

    ```
    s_j ~ Beta(α_j, β_j)
          * exp(-λ * Dist(A_j, I_task))
          * ζ(S_t)^η
    ```

    choose top-k successors; increment `R_ij`. (ζ: synergy across chosen set).&#x20;
  * Use the **successor instruction prompt** (Fig. 6) to translate `(x_i, y_i)` into each successor’s input.&#x20;
* The **Aggregator** gathers terminal outputs and produces a final answer (may use RAG/tools).&#x20;

## 5.2 Environment, Loss & Global Gradient

* Send `S_out` to the selected environment adapter: compute task-specific loss + **verbal diagnostic** (oracle/adversary dual role).&#x20;
* Use **TextGrad** + the paper’s **system feedback prompt** (Fig. 7) to produce a *global textual gradient* at the Aggregator.&#x20;

## 5.3 Backward “Textual Chain Rule”

* For each agent in reverse execution order, merge successor feedback + local context and parse an **agent-specific gradient** (Fig. 8).&#x20;
* Gradient objects are *typed*: semantic/topology/tool.

## 5.4 Coordinated Update (STEV)

* **Semantic** `fP`: Rewrite system prompt (Fig. 9) and update tool loadouts per gradient. Guardrails: diff-check, temperature caps, regression tests on “agent unit tests.”&#x20;
* **Topological** `fG`: Apply one of {add successor, remove successor, connect to aggregator, no-op} using the **topology decision** policy (Fig. 10). Ensure DAG (call `RepairTopology`).&#x20;
* **KABB updates**: for each selected agent `A_i`,

  ```
  α_i^(t+1) = γ^Δt * α_i^(t) + [r_i^(t) + δ * KM(A_i, I_task)] * I{A_i ∈ S_t}
  β_i^(t+1) = γ^Δt * β_i^(t) + [1 - r_i^(t) + δ * KD(A_i, I_task)] * I{A_i ∈ S_t}
  ```

  Update edge synergy `C_syn` using task contribution `R_ij`.&#x20;

> **Note.** MVP can begin with *simplified* KM/KD (e.g., cosine between task capability tags and agent capabilities) and later replace with KG-backed Dist(·).

---

# 6) Knowledge-Aware distance `Dist(A_i, I_task)`

Define an **RDF vocabulary** for agent capabilities and task requirements (skills, domains, tools, constraints). Implement 4 sub-indicators `Ψ_k` (e.g., skill match, domain overlap, tool availability, safety/constraint fit). Query with SPARQL to compute mismatch and aggregate to Dist(·) with weights `ω_k` as in the paper; include a difficulty scaling via `log(1+d_I)`.&#x20;

---

# 7) Tool subsystem (governed, evolvable)

* **Schema** (Pydantic): name, natural-language purpose, `inputs`, `outputs`, constraints, examples, safety notes, version.
* **Runtime**: detached worker with CPU/memory/time caps; structured I/O; redaction.
* **Evolution**:

  * *De novo synthesis* prompt to generate a tool from capability spec.
  * *Refinement* prompt to patch a tool codebase given `<TOOL_FEEDBACK>`. (Appendix A.3, Figs. 12–13).&#x20;

---

# 8) TextGrad integration

* **Where**: gradients/global (aggregator), gradients/local (per agent), and semantic update scoring (choose among candidate rewrites).
* **How**:

  1. Call TextGrad to **score** candidate feedback parses; choose the gradient JSON with highest “improvement signal.”
  2. Use TextGrad’s compositional objectives to **rank** alternative prompt rewrites from Fig. 9.
  3. For topology decisions, use TextGrad to evaluate *counterfactual subgraphs* on cached tasks (small validation slate) to avoid myopic topology drift.

This mirrors the paper’s “textual gradient” notion but gives us a robust, testable implementation path.&#x20;

---

# 9) Orchestrator loop (pseudocode)

```python
def step(task: Task, G: Graph, registry, env, textgrad):
    exec = build_exec_subgraph(G, task)          # Alg. 2
    trace = forward_run(exec, task, registry)    # messages, R_ij, outputs
    y = aggregate(exec, trace)

    loss, env_fb = env.evaluate(task, y)         # scalar + verbal feedback
    g_global = textgrad.parse_system_feedback(y, env_fb)     # Fig.7

    local_grads = {}
    for vi in reversed(exec.order):
        succ_fb = collect_successor_feedback(vi, g_global, local_grads)
        g_i = textgrad.parse_agent_feedback(vi, succ_fb)     # Fig.8
        local_grads[vi.id] = g_i

    # Evolution
    for vi, g_i in local_grads.items():
        if g_i.scope in {"semantic","tool"}:
            apply_semantic_update(vi, g_i)                   # Fig.9
        if g_i.scope == "topology":
            apply_topology_update(G, vi, g_i)                # Fig.10

    repair_topology(G)                                       # keep DAG
    update_kabb_beliefs(G, trace, task, loss, env_fb)        # α/β, C_syn
    return y, loss
```

---

# 10) Evaluation harness

* **Benchmarks**: MATH, GSM8K, HotpotQA, 2Wiki, HumanEval, MBPP, MMLU, BBH, GAIA adapters, including sampling procedures as described (stratified where relevant). Track *Accuracy* and **Cost-Efficiency Score** (Acc/\$), replicating the paper’s evaluation signals.&#x20;
* **Ablations** toggles: disable TEV (topology), SEV (semantics), KABB, Env feedback, Tool integration—matching Table 2.&#x20;
* **Scalability**: plot accuracy vs. iteration (e.g., MBPP over 10 steps) and cost.&#x20;

---

# 11) MVP → Full system roadmap

**Phase 0 (infra & stubs)**

* Graph + agents + aggregator; plain routing (uniform); programmatic environment (MBPP subset); no evolution.
* Logging, seeds, replay.

**Phase 1 (textual gradients + semantic evolution)**

* Integrate TextGrad for global→local feedback; implement Fig. 9 prompt updates; add unit tests (“agent self-tests”).

**Phase 2 (KABB routing)**

* Implement α/β beliefs, Dist(·) with a *minimal* capability tag matcher; synergy ζ with simple pairwise similarity; decay `γ^Δt`.

**Phase 3 (topological evolution)**

* Implement decisions {add, remove, connect-to-aggregator, no-op}; RepairTopology; cool-downs to prevent oscillation.

**Phase 4 (knowledge graph Dist & tool evolution)**

* RDF capability ontology + SPARQL Dist(·); governed tool runtime; tool synthesis/refinement prompts; safety gates.

**Phase 5 (benchmarks & ablations)**

* Wire HotpotQA, 2Wiki, MATH, MMLU, GAIA; run ablations; cost tracking; plots.

---

# 12) Practical engineering considerations

* **Conflict resolution at Aggregator** (paper’s failure case on MATH): implement *tie-breaker protocols*: (a) verification agents vote with calibrated confidence, (b) aggregator uses programmatic checkers when available, (c) force additional single-path re-checks on contradictions before giving up.&#x20;
* **Guardrails**: max degree per node, per-iteration edge edits cap, and “persistence” penalties to avoid thrashing.
* **Caching**: memoize agent tool calls & Dist(·) SPARQL; keep miniature “validation slate” to score candidate topological edits via TextGrad before committing.
* **Determinism**: per-iteration seed, logged bandit samples, and serialized graph snapshots for reproducibility.
* **Cost**: implement early-exit and budgeted routing; store cost tokens per agent to feed CS metric.&#x20;

---

# 13) Minimal configuration (defaults)

* Temperature=1.0 (paper’s setup) for comparability; consider annealing later.&#x20;
* KABB hyperparams: `k=top2` successors per node; `λ` (Dist weight) = 0.5, `η` (synergy exponent) = 0.5, `δ` (knowledge bonus) = 0.2, decay `κ=0.05` (so `γ^Δt = e^{-κΔt}`), small cooldown=2 iterations after a topology change. (Tune per dataset).&#x20;

---

# 14) What TextGrad buys us (and limits)

* **Strengths**: principled, composable objectives for ranking candidate rewrites and structured gradient JSON; easier to A/B *semantic* updates; can also guide *topology* choices by scoring small counterfactuals.
* **Limits**: it still depends on LLM faithfulness—mitigate via validators (programmatic unit tests, dataset-based checks) and constrain update magnitude (diff size, temperature clamp). These mitigations align with the paper’s emphasis on “feedback-rich environments” and verifiable outcomes when available.&#x20;

---

# 15) Acceptance criteria (MVP)

* End-to-end STEV over MBPP subset (100 tasks): **+≥3%** accuracy after 10 iterations vs. vanilla single-agent baseline; logged α/β updates; at least **one topology change** that survives cooldown; stable DAG; cost tracked per iteration. (Targets inspired by the paper’s trends.)&#x20;

---

If you want, I can generate a **scaffolded repo** (files with function signatures and first-pass implementations of the STEV loop, KABB, and the TextGrad adapters) in the next message so you can drop it into your environment and iterate.

