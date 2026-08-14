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

**D6 — The model judges, the code scores.**
`overall_fit` is never asked of the LLM. The scoring agent asks it one narrow
question per requirement — met or not, and *which* CV evidence shows it — and
`compute_fit_score` turns those answers into a number:

```
overall_fit = weight of met requirements / weight of all requirements
              must-have = 1.0, nice-to-have = 0.5
```

Equal-weight postings reduce to "3 of 5 met = 0.6", which is explainable in one
sentence at the final presentation and testable without a model. Only
`requirements` are scored; `responsibilities` are not sent to the model at all.

**D7 — The model points at evidence, it does not write it.**
Judgements carry an *index* into `ParsedCV.evidence`, and the code looks up the
wording. There is no free-text evidence field in what the model returns, so an
invented claim has nowhere to go; an index pointing at nothing downgrades the
match to unmet and is recorded in the trace. This is the Week 6 form of the
project's trust requirement — the full grounding check on *generated* text is
still Week 7.

## The Week 6 pipeline (complete)

`Supervisor.run(cv_source, jd_source)` executes a LangGraph `StateGraph` whose state
*is* the existing `RunContext`:

```
START → parse_cv → parse_jd → research → score_fit → track_application → END
```

| Node | Does | Writes to state |
| --- | --- | --- |
| `parse_cv` | `parse_cv` tool (light tier) | `parsed_cv` |
| `parse_jd` | `parse_jd` tool (light tier) | `job` |
| `research` | `ResearchAgent` → `web_search` + summarise (light) | `brief`, `warnings` |
| `score_fit` | `ScoringAgent` (heavy tier) | `fit_report` |
| `track_application` | builds `ApplicationRecord`, `tracker.add` | `application` |

The finished `RunContext` is the pipeline result — parsed inputs, brief, fit report,
persisted record, warnings and trace — so no second result hierarchy exists.

**Failure policy.** Research is best-effort: an unnamed company, a failed search, an
empty result set or a summary that never validates each leave a warning and let the
fit analysis finish. Scoring is essential: if it exhausts its retries the run raises
`RetryExhaustedError` and **nothing is written to the tracker** — a failed score must
never be mistaken for a poor one. New applications are always `draft`; nothing is ever
auto-submitted.

**Injection.** The supervisor receives its tool registry, both worker agents and the
tracker. It never constructs a database. `build_supervisor(settings)` is the single
place that knows production means SQLite plus a real search client.

## Week 7 (in progress): grounding and the writing agent

The pipeline gains one optional stage, between scoring and tracking:

```
... → score_fit → write_application → track_application → END
```

`write_application` appears only when a `WritingAgent` is injected. Without one the
graph is the Week 6 pipeline exactly — which is also what the fit-threshold branch will
want when a role is not worth applying to.

**Grounding is not a graph node.** It validates the writer's *output*, so it lives
inside the writer's validate-and-retry loop, which is where the proposal's activity
diagram puts it:

```
prompt → heavy model → WritingDraft → schema validation ──(malformed)──→ retry
                                            ↓
                                   grounding validation ──(unsupported)─→ retry,
                                            ↓                             quoting the
                                    TailoredCV + CoverLetter              violations
                                            ↓
                          out of attempts → RetryExhaustedError
```

### D8 — Grounding is a separate layer from schema validation

A perfectly schema-valid tailored CV can still be a fabrication. Schema validation
answers *"is this the right shape?"*; grounding answers *"is this true of **this**
candidate?"* Neither can substitute for the other, so both must pass and they live in
separate modules (`validation/schema_guard.py`, `validation/grounding.py`).

### D9 — Grounding is deterministic first, model-assisted only if asked

Stage 1 (`check_grounding`) makes no model call. A generated line is rejected when it
contains:

- a **distinctive term** — capitalised mid-sentence, an acronym, or carrying a digit or
  `+`/`#` — that the CV never mentions (invented technology, employer, product,
  institution, certification), or
- a **number** the CV never stated, with spelled-out numbers normalised so "five years"
  is checked exactly as "5 years" (invented durations, team sizes, percentage
  improvements).

This is not substring matching: a line may be reworded, reordered, or combine two real
bullets and still pass, because only the *facts* it introduces are checked.

The authoritative evidence is the CV alone — `raw_text`, the parsed evidence lines, and
the candidate's listed skills. A job posting mentioning Kubernetes never makes
"experience with Kubernetes" supportable. The one narrow allowance is `context_terms`,
the employer's own proper nouns, so a cover letter can name the company it is addressed
to; it is never used for technologies or achievements.

Stage 2 (`build_entailment_prompt` + `apply_verdicts`) is an optional model pass over
the claims stage 1 accepted, for fabrications with no fingerprint ("led the migration").
It is **off by default** — an extra heavy call per attempt — and it can only *demote* a
claim, never rehabilitate one stage 1 rejected. The call is made by the writing agent
through the existing router, so no provider is named in the validation layer and the
call is traced like any other.

### D11 — Two evidence domains, never merged

The proposal wants the cover letter grounded in the CV **and** the company brief. Merging
the two corpora would be a trust hole: "Arbisoft uses Django" would silently license "I
have Django experience". So the checker keeps them apart and decides, per claim, which
one applies:

- first person **plus** an experience word ("I have experience…", "my skills…") →
  **candidate claim**, CV corpus only. This wins even when the company is also named, so
  a mixed sentence is judged strictly.
- otherwise, if the subject is the employer ("your company", "the team", the company or
  role named) → **company statement**, which may additionally cite the verified brief.
- anything else → candidate, i.e. strict. A bare "Built Django services." cannot quietly
  borrow a company fact.

The tailored CV is checked with no company context at all. The employer's proper nouns
are a *naming* allowance for company statements only — a role advertised as "Junior AI
Engineer" must not make "my AI skills" supportable, any more than "Kubernetes Engineer"
would license Kubernetes.

Errors trend towards false rejection rather than fabrication, which is the direction this
project wants to fail in.

### D12 — Research searches for the company, not the job

A live run searched "Arbisoft Junior AI Engineer Python FastAPI" and got a stranger's
LinkedIn profile, an Instagram post and a vacancy at a different company — then
attributed them to the target employer. Company name plus role plus technologies *is* a
job search. The query is now `"<company>" company overview`, role relevance is applied in
the summariser prompt instead, and a relevance guard discards any result that never
mentions the company before the summariser sees it.

### D13 — The MCP server is an adapter, not a second implementation

`mcp_server/server.py` exposes `applications://all` plus five tools, each one a thin
call into `SQLiteApplicationTracker`, `ApplicationRecord` and `WebSearchClient`. No
business rule is restated, so the MCP surface cannot drift from the pipeline: the same
Pydantic model validates a record, the same status enum applies, the same database file
is read. `ApplicationTracker` being a four-method interface is what makes this possible
without the server knowing where rows live (NFR-9).

`score_fit` and `tailor_application` complete FR-8's "research/tailoring tools" half.
Both parse their inputs with the pipeline's own parsers and then hand structured objects
to `ScoringAgent` / `WritingAgent`, so the rubric and the no-fabrication guard are the
agents' own code — the MCP layer adds no judgement of its own. `tailor_application`
returns the tailored CV *and* the cover letter because they are one model call here;
two tools would double the cost or force the server to hold state between calls.

**Model tier is opt-in, not opt-out.** An MCP call is made by a client the operator may
not be watching, so the model-backed tools run on the *light* tier by default — the
substitution is applied to a copy of the settings, leaving pipeline routing untouched.
`MCP_MODEL_TIER=heavy` selects the configured heavy provider, and nothing else does.

**No company context over MCP.** `tailor_application` passes `brief=None`, so the letter
is grounded in the CV alone. Accepting caller-supplied "company facts" would let an MCP
client introduce material this server cannot verify.

Two further constraints:

- **Nothing can auto-submit.** `set_application_status` records a decision a human has
  already made (FR-14); the server never sends an application anywhere.
- **Errors carry no credentials.** A failed search reports its HTTP status only — httpx
  puts the full request URL in its message, and SerpAPI carries the key as a query
  parameter, so the raw message would have leaked the credential into MCP error
  responses and the trace log.

Errors surface as protocol errors carrying a plain message — an unknown application, an
unknown status (which lists the valid ones), an out-of-range score from the shared
model. No message contains a configuration value; a missing search key is reported by
naming `SEARCH_API_KEY`, never its contents.

### D10 — A failed rewrite costs the documents, not the run

If the writer cannot produce grounded material within its attempt budget it raises
`RetryExhaustedError` rather than returning something unverifiable. The supervisor
catches that, records a warning, and lets the run finish: the fit report and the tracked
application survive, and the user keeps their original CV plus the gap list — the
proposal's stated fallback.

## What the scaffold deliberately does not include

Parsing, research, scoring, writing, SQLite persistence, the MCP server, and the real UI.
Those are Week 6/7 deliverables; the scaffold fixes their interfaces only.
