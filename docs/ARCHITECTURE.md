# Architecture

Grows through Weeks 6–8; this version documents the scaffold's interfaces and the
decisions taken at the start, so the final write-up is an extension of it rather than
an archaeology exercise.

## Layers

The specification describes four layers, and the package mirrors them:

| Layer | Modules | Does |
| --- | --- | --- |
| Input | `models/inputs.py` | Turns a raw CV and JD into `ParsedCV` / `JobDescription` |
| Capability | `tools/`, `memory/`, `mcp_server/` | Web search, scoring, tailoring, tracking — narrow and independently testable |
| Orchestration | `agents/`, `llm/`, `validation/`, `observability/` | Supervisor + workers, model routing, output validation, tracing |
| Presentation | `app/` | Streamlit view over structured outputs |

## Key interfaces (fixed in the scaffold)

- **`ModelClient`** (`llm/base.py`) — `complete(messages, system, tools, max_tokens) -> ChatResult`.
  Agents never import a vendor SDK. Adding a provider is one adapter in `llm/providers.py`.
- **`ModelRouter`** (`llm/router.py`) — maps a *step name* to a tier (`light`/`heavy`).
  This is where the cost strategy lives; agents ask for a step, not a vendor.
- **`Tool`** / **`ToolRegistry`** (`tools/__init__.py`) — name, description, JSON schema,
  callable. The same definition serves both providers' tool-calling APIs and the MCP server.
- **`BaseAgent`** (`agents/base.py`) — traced `call_model(step, ...)` and `call_tool(name, ...)`,
  plus an abstract `run(context)` that must return a Pydantic model, never free text.
- **`RunContext`** (`memory/session.py`) — the single state object passed between agents;
  there is no hidden channel between workers.
- **`ApplicationTracker`** (`memory/tracker.py`) — four methods. In-memory now, SQLite in
  Week 6, exposed over MCP in Week 7 — all against the same interface.
- **`generate_validated`** (`validation/schema_guard.py`) — produce → validate → retry with
  the error fed back → bounded give-up.
- **`TraceCollector` / `traced_tool_call`** (`observability/tracing.py`) — one event per
  tool and model call with latency and token counts; `totals()` is the run-level report.

## Decisions

**D1 — LangGraph for orchestration, not a hand-rolled loop.**
The specification allows LangGraph or the OpenAI Agents SDK. LangGraph wins here because
the pipeline is a *fixed graph with branches* (fit-threshold branch, no-fabrication retry
loop, next-role loop) and the graph reads the same as the flow diagram in the proposal —
which matters for explaining it in the final presentation. It also keeps the supervisor
deterministic: routing costs no LLM call, the same conclusion reached in Week 5.

**D2 — Two tiers, not two vendors, in the code.**
`FR-11` requires ≥2 providers. Rather than scattering provider checks, the code knows only
`light` and `heavy`; `.env` decides that light is Groq, Ollama, or OpenRouter (all
OpenAI-compatible, one adapter) and heavy is Claude. Week 7's multi-model comparison then
means running the same step through two configured tiers, not new plumbing.

**D3 — Validation is two independent checks.**
Schema validity and factual grounding are separate modules because a perfectly
schema-valid tailored CV can still be a fabrication. The schema guard runs first; the
grounding check (Week 7) can never be satisfied by it.

**D4 — Evidence is stored verbatim.**
`CVEvidence.text` keeps the candidate's own wording and `ParsedCV.raw_text` keeps the full
source, because both are the reference the no-fabrication check compares against. Parsing
must not paraphrase.

**D5 — Tracing is a context manager, not a hook subclass.**
Week 5 traced through Week 4's `HookManager`. Here agents call tools directly, so
`traced_tool_call` wraps the call site instead — and it records token counts, which the
Week 5 version could not.

## What the scaffold deliberately does not include

Parsing, research, scoring, writing, SQLite persistence, the MCP server, and the real UI.
Those are Week 6/7 deliverables; the scaffold fixes their interfaces only.
