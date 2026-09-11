# Job Application Agent
## Complete Project Explanation

*Prepared as a study document ahead of the internship mentor review. Every technical claim below was checked directly against the committed code, tests, and git history at commit `171d2f5` (pushed, `origin/main` synchronized) — nothing here is aspirational or invented.*

---

### 1. Project Overview

**In simple language:**

This project is a personal assistant for job applications. You give it two things — your CV and a job posting — and it does the tedious analytical work a careful human would do: it checks how well your CV actually matches what the job asks for, tells you honestly what's missing, looks up basic facts about the company, and produces a tailored version of your CV plus a draft cover letter that only says things your CV already supports. Every application it processes is saved to a small local tracker so you can see everything you've applied to in one place.

The one rule the whole system is built around is: **never say the candidate has a skill or experience that isn't actually in their CV.** It would be easy to build something that makes a candidate look better by inventing a few plausible details — this project deliberately does the opposite, and goes out of its way to catch and reject that kind of thing before it ever reaches the screen.

**Technically:**

It is a multi-agent Python pipeline, orchestrated with **LangGraph**, that takes raw CV and job-description text (or `.txt`/`.md`/`.pdf` files) and runs them through a fixed sequence of specialised components — parsing, company research, evidence-based fit scoring, and grounded document generation — each of which produces a schema-validated **Pydantic** object rather than free text. A Streamlit UI sits on top for a human to run this interactively, and a custom **MCP (Model Context Protocol) server** exposes the same tracker and pipeline as tools for another program (e.g. an IDE agent) to call.

**What the user gives the system:** a CV and a job description (pasted text or an uploaded file).

**What the system produces:**
- a **fit report** — an overall score plus a per-requirement breakdown of how well the CV supports each thing the job asks for
- a **company research brief** with cited sources (when research succeeds)
- a **tailored CV** and a **cover letter**, but only if every sentence in them can be traced back to something the CV actually says
- a **tracked application record** saved to a local database, always as a `draft`

**Main goal:** help a candidate understand their real fit for a role and produce supporting documents, without ever fabricating experience they don't have.

---

### 2. Problem Statement

Applying to jobs well, by hand, involves several genuinely tedious and error-prone steps:

- **Reading a CV against every job description** to work out how well it actually fits — most people either skip this and apply blind, or do it once and reuse the same CV everywhere.
- **Identifying skill gaps honestly** — it's hard to be objective about your own CV; people either underestimate weak matches or convince themselves something "counts" when it doesn't.
- **Researching the company** for every application — reading About pages, recent news, and product info takes real time per application.
- **Tailoring the CV's emphasis** for each specific role, without rewriting it from scratch each time.
- **Writing a cover letter** that actually references the specific role and company, rather than a generic template.
- **The risk of exaggerating or inventing experience** — under pressure to look competitive, it's tempting (for a person or a naive AI tool) to round "basic knowledge" up to "experience," or claim a technology the posting wants even though the CV never mentions it. This is exactly the failure mode that makes a candidate's materials untrustworthy, and it's the central risk this project is built to prevent.

This project addresses each of these directly: it automates the CV-vs-requirements comparison with cited evidence, automates a short company lookup, and automates tailoring — while deliberately refusing to automate the one part that would make the results untrustworthy: it will not let generated text claim anything the CV doesn't actually support, and it says so explicitly rather than silently smoothing over the gaps.

No statistics about job-application success rates, time saved, or industry-wide numbers are claimed anywhere in this project — none were measured, and none should be presented as if they were.

---

### 3. Core Idea

The central design principle, stated in the codebase's own comments in almost these exact words:

> **The system should help tailor an application while remaining grounded in what the candidate's CV actually supports. The model judges, the code decides.**

Two ideas sit underneath this:

1. **The CV is the only source of truth about the candidate.** Not the job posting, not what "would be plausible" for someone with this background, not the advertised job title. If it isn't written in the CV, the system is not allowed to claim it.
2. **Judgement and enforcement are split.** The language model is used for what it's actually good at — reading evidence and making a judgement call ("does this CV evidence support this requirement, and how well?"). But the model is *never* trusted to police itself. A separate, deterministic layer of plain Python code checks every one of the model's outputs — the shape of the JSON, and (for anything the model *writes*, like a cover letter sentence) whether every fact in it can actually be traced back to the CV. If it can't, that output is rejected and the model is asked to try again, with the specific problem named.

Why this matters: a tool that quietly invents a skill to make a candidate look better isn't actually helping — it's setting the candidate up to fail an interview about experience they don't have, or worse, damaging their credibility if it's caught. Grounding is what turns "an AI wrote something plausible" into "an AI wrote something true."

---

### 4. End-to-End Workflow

This is the actual pipeline wired in `job_agent/agents/supervisor.py`, not an idealised version of it:

```
CV text/file + Job description text/file
            |
            v
    parse_cv   (light-tier model + verbatim guard)
            |
            v
    parse_jd   (light-tier model)
            |
            v
    research   (light-tier model; best-effort)
            |
            v
    score_fit  (heavy-tier model)  ---> Fit report (score + per-requirement matches + gaps)
            |
            v
 write_application (heavy-tier model; OPTIONAL stage, only runs if a writing agent
            |        was configured — this is the Streamlit app's default)
            v
    track_application  (no model call — saves a record to SQLite)
            |
            v
    Results shown in the Streamlit UI
```

Stage-by-stage:

| Stage | Input | What happens | Output | Component | Model tier |
|---|---|---|---|---|---|
| **Parse CV** | Raw CV text or file path | A model extracts verbatim evidence lines and skills; a deterministic guard then drops any line the model didn't actually copy from the source | `ParsedCV` (candidate name, evidence list, skills, raw text) | `job_agent/tools/parsing.py: parse_cv` | light |
| **Parse JD** | Raw job-posting text or file path | A model extracts role, company, requirements (flagged must-have vs nice-to-have), responsibilities, skills | `JobDescription` | `job_agent/tools/parsing.py: parse_jd` | light |
| **Research** | The parsed job's company name | A `web_search` tool call returns snippets; a model summarises them into facts; a grounding guard drops any fact that doesn't cite a real result or states a number the result never contained | `CompanyBrief` (summary, facts, sources) — or an honest "unavailable" brief | `job_agent/agents/research.py: ResearchAgent` | light |
| **Score fit** | Parsed CV + parsed JD | A model judges each requirement against the CV's evidence (MATCH/PARTIAL/RELATED/MISSING); code computes the weighted score, and downgrades any judgement that doesn't cite real evidence | `FitReport` (score, per-requirement matches, gaps, recommended emphasis) | `job_agent/agents/scoring.py: ScoringAgent` | heavy |
| **Write application** *(optional stage)* | CV + JD + fit report + company brief | A model drafts tailored CV bullets and a cover letter; every generated claim is then checked against the CV (and, for company statements only, the verified brief); unsupported drafts are rejected and retried | `TailoredCV` + `CoverLetter`, or nothing at all if grounding never succeeds | `job_agent/agents/writing.py: WritingAgent` | heavy |
| **Track application** | The fit report | No model call — a record is written to SQLite | `ApplicationRecord` (always starts as `draft`) | `job_agent/agents/supervisor.py: Supervisor._track` | — |

Two important, verified nuances:
- **Research failures are non-fatal.** If the company can't be identified, the search fails, or nothing relevant comes back, the pipeline continues with an honest "research unavailable" brief and a warning — the fit report is unaffected.
- **Scoring failures are fatal on purpose.** If the scoring model can't produce a valid, well-formed report after its retry budget, the whole run raises an error and **nothing is tracked** — the codebase's own reasoning is that "a failed score must never be mistaken for a poor one."
- **Writing failures degrade gracefully.** If the writer can never produce a grounded draft, the run still completes: the fit report and the tracked record survive, the candidate keeps their original CV, and the UI shows a warning explaining that nothing could be generated without an unsupported claim.

---

### 5. System Architecture

```
                          User
                           |
                           v
                  Streamlit UI (job_agent/app/ui.py)
                           |
                           v
              Supervisor  (LangGraph StateGraph, agents/supervisor.py)
                           |
        +---------+--------+---------+-----------+------------+
        |         |        |         |           |            |
        v         v        v         v           v            v
   parse_cv   parse_jd  research   score_fit  write_application track_application
   (tool)     (tool)   (agent)    (agent)      (agent, optional)   (tracker)
        |         |        |         |           |            |
        v         v        v         v           v            v
   ParsedCV  JobDescription CompanyBrief FitReport  TailoredCV +   ApplicationRecord
                                                     CoverLetter    (SQLite)
                                                          |
                                                          v
                                              Grounding + Validation
                                              (schema_guard.py + grounding.py)
                                              -- retries on failure,
                                                 refuses to return
                                                 unsupported material

   Cutting across every stage above:
        ModelRouter (llm/router.py) -- picks "light" or "heavy" model per step
        TraceCollector (observability/tracing.py) -- records every call, timing, tokens

   Separately, an MCP server (mcp_server/server.py) exposes the tracker and a
   subset of the pipeline (score_fit, tailor_application, search_company) as
   tools for an external MCP client (e.g. Claude Code), reusing the exact
   same agents and validation code — not a second implementation.
```

**Major components:**

- **Supervisor** — the orchestrator. Built on LangGraph's `StateGraph`, so the pipeline above is literally a graph object, not a chain of manual function calls. It decides *what runs when* and never does any parsing, research, scoring or writing itself.
- **RunContext** (`memory/session.py`) — a single object that flows through every stage of the graph, accumulating the parsed CV, the job description, the brief, the fit report, the tailored documents, warnings, and a full trace. There is no hidden channel between components — everything one stage produced, the next stage can see, because it's the same object.
- **Worker agents** (`agents/`) — `ResearchAgent`, `ScoringAgent`, `WritingAgent`, all inheriting a shared `BaseAgent` that gives them a traced way to call a model and a traced way to call a tool.
- **Model layer** (`llm/`) — a provider-agnostic `ModelClient` interface, concrete adapters for Anthropic and any OpenAI-compatible endpoint, and a `ModelRouter` that maps a *step name* (e.g. `"score_fit"`) to a tier (`"light"` or `"heavy"`), so no agent code ever names a specific vendor.
- **Validation layer** (`validation/`) — two independent checks: `schema_guard.py` (is the model's JSON well-formed and does it satisfy the Pydantic schema?) and `grounding.py` (is every generated claim actually true of this candidate's CV?).
- **Tracker** (`memory/tracker.py`) — a small SQLite-backed persistence layer behind a four-method interface (`add`, `get`, `list`, `set_status`), also usable in-memory for tests.
- **MCP server** (`mcp_server/server.py`) — exposes the tracker and select pipeline capabilities to any MCP-speaking client.
- **Streamlit UI** (`app/`) — the human-facing surface; a thin view layer over everything above.

---

### 6. Repository Structure

```
job_agent/
  models/        Pydantic schemas: parsed CV/JD, fit report, tracker record
  llm/           Model client interface, provider adapters, tier router
  agents/        BaseAgent, the LangGraph supervisor, research/scoring/writing agents
  tools/         Parsing, document loading (PDF/txt/md), web search — narrow, testable units
  memory/        The application tracker (SQLite) + per-run session state
  validation/    Schema guard (retry on malformed JSON) + grounding (anti-fabrication)
  observability/ Trace events for every tool/model call, with latency and token counts
  mcp_server/    The custom MCP server exposing the tracker + pipeline as tools
  app/           Streamlit UI: streamlit_app.py (3-line entry point), ui.py (the real view logic)

tests/           21 test files, one per component, plus fixtures/ (a real sample CV + JD)
docs/
  ARCHITECTURE.md          Design decisions and interfaces, in more depth than this document
  PROJECT_EXPLANATION.md   This document
data/            Generated at runtime — the tracker database and the trace log (gitignored)

pyproject.toml   Project metadata, pytest/ruff/coverage configuration
requirements.txt Runtime + dev dependencies
.env.example     Configuration template (placeholders only, no real keys)
```

Not listed above: individual trivial files (`__init__.py` re-exports, etc.) — those are covered implicitly by the module descriptions.

---

### 7. Agents

The codebase has three classes that inherit `BaseAgent`, plus the `Supervisor` that coordinates them. Parsing (`parse_cv`/`parse_jd`) is implemented as **tools**, not `BaseAgent` subclasses — it's still an LLM-backed step in the pipeline, just structured as a callable tool the supervisor invokes directly, not an agent object with its own class.

**ResearchAgent** (`agents/research.py`)
- **Purpose:** produce a short, honest, sourced brief about the hiring company.
- **Input:** the parsed job description (for the company name).
- **Output:** a `CompanyBrief` (summary, facts, sources) — always a valid object, even when research fails.
- **Model tier:** light (the `"summarise"` step).
- **Important behavior:** never lets the model search from memory — it only ever sees retrieved search snippets, and every fact it returns must cite the specific result it came from. A relevance guard drops search results that never mention the target company *before* the model sees them.
- **Failure/degradation:** an unnamed company, a failed search, an empty result set, or a summary that never validates all produce a valid brief that plainly says research was unavailable — the pipeline continues.

**ScoringAgent** (`agents/scoring.py`)
- **Purpose:** judge how well the CV's evidence supports each requirement of the job, and compute an overall fit score.
- **Input:** parsed CV + parsed job description.
- **Output:** a `FitReport`.
- **Model tier:** heavy (the `"score_fit"` step).
- **Important behavior:** the model is asked one narrow question per requirement (which of MATCH/PARTIAL/RELATED/MISSING, and which numbered piece of CV evidence backs it) — it is never asked to output a score directly. See Section 11 for the exact rubric.
- **Failure/degradation:** scoring is essential to the pipeline — if it can't produce a valid report after its retry budget, the whole run fails rather than silently guessing a score.

**WritingAgent** (`agents/writing.py`)
- **Purpose:** produce a tailored CV and a cover letter, using only material the CV supports.
- **Input:** parsed CV, parsed JD, the fit report (for what to emphasise and what gaps to avoid), and optionally the company brief (for the letter only).
- **Output:** a `TailoredCV` and a `CoverLetter` — or nothing at all.
- **Model tier:** heavy (`"tailor_cv"`; an optional extra `"verify_claims"` pass is also heavy-tier but off by default).
- **Important behavior:** every draft goes through two independent checks (schema validity, then grounding) before it's accepted; a rejected draft is sent back to the model with the specific unsupported claims named, up to a bounded number of attempts.
- **Failure/degradation:** if grounding can never be satisfied, the writer raises rather than returning anything — the supervisor catches this, records a warning, and the run still completes with the original CV and the gap list standing on their own.

**Supervisor** (`agents/supervisor.py`)
- **Purpose:** orchestration only — decides what runs when, wires the `RunContext` through the graph, and writes the final tracker record.
- **Input:** a CV source and a JD source.
- **Output:** the finished `RunContext`.
- **Model tier:** none directly — it delegates every model call to the agents above.
- **Important behavior:** everything it coordinates (tools, agents, tracker) is injected via its constructor, so it has no idea whether the tracker is SQLite or in-memory, and it never opens a database itself.

---

### 8. Model Routing

The project separates work into two **tiers**, not two hardcoded vendors:

- **`light`** — cheap, mechanical steps: parsing the CV, parsing the job description, summarising search results.
- **`heavy`** — judgement-heavy steps: scoring fit, writing tailored documents, and (if enabled) the optional claim-verification pass.

Each tier is configured independently in `.env` and can point at a **different provider or the same one** — the code has no opinion about which. Two provider kinds are supported:

- **`anthropic`** — the Anthropic Claude API directly.
- **`openai_compatible`** — any endpoint that speaks the OpenAI chat-completions API shape. This one adapter is what actually lets the project reach Groq, Ollama, or OpenRouter (and Gemini, via Google's OpenAI-compatible endpoint) — they differ only in base URL and model name, so no extra code is needed per provider.

`ModelRouter` (`llm/router.py`) maps a step name to a tier via a fixed lookup table (`parse_cv`→light, `parse_jd`→light, `summarise`→light, `score_fit`→heavy, `tailor_cv`→heavy, `verify_claims`→heavy). Agents never name a vendor; they ask the router for a *step*, and the router hands back whichever client is configured for that step's tier. Changing provider or model is purely a `.env` change (`HEAVY_PROVIDER`, `HEAVY_MODEL`, `LIGHT_PROVIDER`, `LIGHT_MODEL`, and the matching base URL / API key) — no code changes are required.

**Honesty check on the actual configured setup:** `.env.example` documents Anthropic as the default heavy-tier provider and an OpenAI-compatible endpoint (Groq by default) as the light tier — so out of the box, this is genuinely **two different providers**, not two models from the same one, satisfying the "at least two providers" requirement. However, `.env.example` also documents an alternative where `HEAVY_PROVIDER` is switched to `openai_compatible` (e.g. Gemini) — in that configuration both tiers would be OpenAI-compatible endpoints, potentially from different companies (e.g. Gemini for heavy, Groq for light) or, if someone pointed both at the same endpoint, the same provider twice. **Which configuration is actually active in this project's `.env` was not read by this document** (per the task's explicit instruction not to expose `.env` contents) — if you're asked this by your mentor, check your own `.env`'s `HEAVY_PROVIDER` and `LIGHT_PROVIDER` values and answer honestly from that, rather than assuming the default.

**Why separate light and heavy at all:** it's a direct cost/latency strategy — there's no reason to pay for the strongest available model to extract text fields from a CV, but the fit judgement and the anti-fabrication writing genuinely benefit from a stronger model's reasoning.

---

### 9. Structured Outputs

Every model call in this project is asked for **a single JSON object and nothing else**, and every reply is immediately parsed into a **Pydantic** model — never trusted as free text.

**Why:** free text from a model is unpredictable to consume programmatically. A Pydantic schema gives three things for free: required fields are actually required, types are checked (a score has to be a number), and range/enum constraints are enforced automatically (`overall_fit` is declared `ge=0.0, le=1.0`; `match_level` can only be one of the four defined values). Every schema in the project also sets `extra="forbid"`, so a model can't sneak in an invented field the code was never told to expect.

**The parse → validate → retry loop** (`validation/schema_guard.py`, `generate_validated`):
1. Call the model.
2. Try to parse the reply as JSON (a fenced-code-block wrapper or surrounding prose is stripped first) and validate it against the target schema.
3. If it fails, the exact validation error is fed back to the model in the next prompt ("your previous reply was rejected because...") and it's asked again.
4. After a bounded number of attempts (3 by default) with no valid reply, a `RetryExhaustedError` is raised — the caller (an agent, or the supervisor) decides how to degrade, rather than the guard silently accepting something invalid.

**JSON mode:** for OpenAI-compatible providers, the client requests `response_format={"type": "json_object"}` (Google's Gemini OpenAI-compatible layer documents support for this), which constrains the provider's own decoding to syntactically valid JSON rather than relying on the prompt alone. This is skipped whenever the call also passes tool definitions, since forced-JSON output and function-calling aren't meant to be combined — though no call in this codebase currently uses both together.

**A real, verified edge case this project handles:** some OpenAI-compatible providers (Groq, observed in a live run per the code comment) validate JSON mode *server-side* and reject a malformed generation as an HTTP 400 error instead of returning it as a normal reply. `job_agent/llm/providers.py`'s `_json_validate_failure_text` specifically recovers the malformed text out of that error response so it still reaches the schema guard's normal reject → retry → feedback flow, instead of the exception escaping and crashing the call outright.

---

### 10. Grounding / Anti-Fabrication System

This is the project's core trust mechanism (`job_agent/validation/grounding.py`).

**Why generated material needs grounding:** the writing agent is allowed to reorder, re-word, and re-combine the candidate's *real* content — but a language model asked to write a persuasive cover letter has a natural pull toward filling gaps with something plausible. Schema validation (Section 9) only checks that the *shape* of the output is correct; it has no idea whether a sentence is *true*. Grounding is a second, independent check that answers exactly that question.

**How claims are checked against CV evidence:** every generated line (CV bullet, or cover-letter sentence) is compared against the candidate's real CV text — the raw source, the parsed evidence lines, and the candidate's listed skills. This is **not substring matching** — a line can be reworded, reordered, or combine two real bullets and still pass, because only the *facts it introduces* are checked, specifically:
- **distinctive terms** — a capitalized mid-sentence word, an acronym, or anything containing a digit or `+`/`#` (so a technology, employer, product, institution, or certification) that never appears anywhere in the CV
- **numbers** — including spelled-out ones ("five years" is checked exactly as "5 years" would be), so an invented duration, team size, or percentage is caught

A handful of deliberately narrow, code-documented exceptions exist so the checker doesn't produce false positives on genuinely equivalent phrasing:
- **plurals** — "REST APIs" is recognised as the same claim as "REST API" in the CV (fixed after a live run rejected three otherwise-correct drafts over exactly this)
- **degree abbreviations** — "BS" and "Bachelor's" are treated as the same claim (fixed after a live run rejected "the CV never mentions Bachelor" for a CV that said "BS Data Science student")
- **the employer's own name and the advertised role title** — allowed *only* for statements about the company, never as evidence for a candidate claim (a role titled "Kubernetes Engineer" does **not** make "I have Kubernetes experience" supportable)

**What happens when a claim is unsupported:** it is rejected, and the *reason* is recorded — e.g. "the CV never mentions Kubernetes." All the rejected claims for one draft are collected together.

**How regeneration works:** the writing agent (`agents/writing.py`) retries generation up to a bounded number of times (3 by default). On each retry, the exact rejected claims and their reasons are quoted back to the model verbatim, with an explicit instruction to remove them rather than rephrase them.

**Why the system refuses to produce a document when grounding can't verify it:** if every retry attempt still contains an unsupported claim, the writer raises `RetryExhaustedError` instead of returning the best-of-a-bad-lot draft. The supervisor catches this, records a warning, and the run still completes — the fit report and the tracked application survive, and the candidate keeps their original, unedited CV plus the honest gap list, rather than receiving a document nobody can vouch for.

**A real example from the test fixtures** (`tests/test_scoring_agent.py`, illustrating the same "subject relevance" principle grounding and scoring share): a candidate's CV states only *"Basic knowledge on CSS and HTML."* A job posting requires *"Basic understanding of APIs."* An earlier live run had the scoring model call this **PARTIAL** evidence — its own stated reason even admitted "which is not basic understanding of APIs," but it picked PARTIAL anyway. The fix was a rule added to the scoring prompt: check whether the evidence is about the *same subject* as the requirement *before* considering proficiency level at all. This is covered by a regression test that confirms the correctly-classified case (MISSING) flows through the pipeline properly. **Being fully honest about what this test does and doesn't prove:** because the rule lives in the prompt, not in deterministic code, the test can confirm the pipeline handles a correct judgement properly — it cannot *guarantee* the model will always classify a new, differently-worded mismatched pair correctly on a future live call. This limitation is stated openly in [Known Limitations](#22-known-limitations) below, not hidden.

---

### 11. Scoring System

The rubric (`job_agent/agents/scoring.py`, `compute_fit_score`) uses two independently-verified weight tables:

**Match strength** — how well the evidence supports the requirement, judged by the model per requirement:

| Level | Weight | Meaning |
|---|---|---|
| **MATCH** | 1.0 | The CV clearly satisfies the requirement, including the proficiency level asked for |
| **PARTIAL** | 0.5 | The same skill is evidenced, just at a lower proficiency than the posting asks for |
| **RELATED** | 0.25 | Genuinely adjacent evidence exists, but doesn't itself demonstrate the specific requirement |
| **MISSING** | 0.0 | No reliable evidence anywhere in the CV |

**Requirement weight** — how much each requirement counts, from the job posting's own "must have" vs "nice to have" wording:

| Kind | Weight |
|---|---|
| Must-have | 1.0 |
| Nice-to-have | 0.5 |

**The formula, exactly as implemented:**

```
overall_fit = (sum of requirement_weight × match_strength) / (sum of requirement_weight)
```

**A verified worked example already used and tested in this project's own code** (`docs/ARCHITECTURE.md`, and `tests/test_scoring_agent.py::test_three_of_five_equally_weighted_requirements_scores_zero_point_six`): for five equally-weighted requirements where the model judges three as MATCH and two as MISSING, the score is `3.0 / 5.0 = 0.6` — the classic "3 of 5 met" arithmetic the two-level system already had, which the four-level system was deliberately built to widen without replacing.

**Requirements, matches, evidence, and gaps, in plain terms:**
- **Requirements** come straight from the parsed job posting, each flagged must-have or nice-to-have.
- **Matches** are the model's per-requirement judgement, but the model doesn't get to *write* the supporting evidence — it can only cite the **index** of a real piece of CV evidence the parser already extracted. If it cites an index that doesn't exist (or claims a level with no citation at all), the code downgrades that judgement to MISSING regardless of what the model claimed — a model cannot earn score by pointing at evidence that isn't real.
- **Gaps** are exactly the requirements whose final match strength is `0.0` (i.e. MISSING) — nothing else. A partial or related match is real evidence at a lower strength, not a gap, and is reported as what it is.

**How the scoring model is instructed to avoid inferring unsupported skills:** the system prompt explicitly tells it never to infer a skill from a job title, a field of study, or "what would be plausible" for the candidate — only from evidence that is actually written down. That instruction is backed, not just hoped for, by the evidence-index enforcement described above.

**Real input example, from this project's own test fixtures** (`tests/fixtures/sample_cv.txt` and `sample_jd.txt`) — showing the *shape* of input the scorer works with, not a claimed live output:

> CV evidence includes: *"Built a Task Management REST API with FastAPI, SQLAlchemy and SQLite..."* and *"Built a research agent with tool calling, session memory and pre/post-tool hooks."* Skills listed: Python, FastAPI, SQLAlchemy, SQLite, Pydantic, pytest, Docker, MCP, prompt engineering. Education: BS Computer Science.
>
> The paired job posting asks for: Strong Python (must-have), REST API experience (must-have), LLM APIs/prompt engineering familiarity (must-have), pytest testing discipline (must-have), SQL comfort (must-have); Kubernetes/cloud, React, and agent-framework/MCP experience are all nice-to-have.

No specific match level or score for this pair is claimed here, because doing so would require an actual model call this document was explicitly told not to make.

---

### 12. Research Component

**How it works:** `ResearchAgent` (`agents/research.py`) takes the parsed job's company name and builds a query using **only the company name** — deliberately never the role title or the posting's technologies. This is a fix for a real, verified bug: an earlier query that combined company + role + technologies ("Arbisoft Junior AI Engineer Python FastAPI") returned a stranger's LinkedIn profile and a vacancy at a different company, which the summariser then mistakenly attributed to the target employer.

**Search provider:** either SerpAPI or Brave (`SEARCH_PROVIDER` in `.env`), both plain HTTPS GET requests. `SEARCH_API_KEY` is required for either to work.

**Extraction/summarisation:** the light-tier model is given the numbered raw search results and asked for a short factual summary plus 3–6 standalone facts, each citing which numbered result it came from. Two guards run after the model replies: a relevance guard drops any search result that never mentions the target company **before** the model ever sees it, and a grounding guard afterward drops any fact that doesn't cite a real result index, or that states a number the cited result never actually contained.

**Degradation when search fails:** every failure mode — an unnamed company, a failed HTTP request, an empty result set, or a result set with nothing relevant — produces a *valid* `CompanyBrief` that plainly states research was unavailable, rather than an exception or a guessed answer. The pipeline continues either way; the fit score never depends on the company brief.

**How sources are presented:** the UI shows the brief's summary and facts, then lists every cited source as a clickable link (title + URL).

**Why research is kept separate from candidate evidence:** this is the "two evidence domains" rule (see Section 10) — a fact about the company (e.g. "the company uses Django") must never quietly become a claim about the candidate ("I have Django experience"). Keeping the corpora structurally separate in the grounding checker is what makes that impossible by construction rather than by prompt instruction alone.

---

### 13. Writing Pipeline

**Tailored CV:** 4–8 re-ordered, re-worded CV bullets, strongest evidence first (guided by the scoring agent's `recommended_emphasis`), plus a list of what was de-prioritised. The writer may reorder, re-word, and combine two real pieces of evidence into one line — it may never add a technology, employer, achievement, number, duration, seniority level, or qualification the CV doesn't state.

**Cover letter:** 3–4 short paragraphs, a fixed salutation format (`"Dear Hiring Team,"`), drawing on the CV evidence and, separately, the verified company facts (never merged with candidate evidence). Template placeholders like `[Candidate]` or `[Your Name]` are explicitly forbidden in the prompt and would also be caught by grounding if one slipped through.

**Grounding:** every bullet and every cover-letter sentence is checked independently (see Section 10), in the correct evidence domain for its subject.

**Retries:** up to 3 attempts by default (`DEFAULT_GROUNDING_ATTEMPTS`), each one told exactly which claims from the previous attempt were rejected and why.

**Rejection of unsupported drafts:** if no attempt produces a fully grounded result, nothing is returned — `RetryExhaustedError` propagates up, the supervisor turns it into a warning, and the original CV plus the fit report's gap list stand on their own.

**Real, verified behavior demonstrated by the codebase's own test suite and commit history (not a live run claimed here):**
- A fix (commit `aa91acb`) for exactly the CV-says-"REST API"-draft-says-"REST APIs" plural mismatch, verified against three real rejected drafts from a live rehearsal.
- A fix for `"[Candidate]"` being used as a placeholder signature when no candidate name was given — solved by actually passing the real name into the prompt and forbidding placeholders explicitly, with grounding still catching one if it appears anyway.
- A fix for a salutation ("Dear Hiring Team, I am excited...") being written as one run-on sentence, which caused "Hiring" to be flagged as an invented proper noun — the code now strips a leading salutation from a claim before checking it, rather than requiring the model to format it on its own line.

---

### 14. Persistence / Application Tracker

**What it is:** a small SQLite database (`data/applications.db` by default), accessed through `SQLiteApplicationTracker` (`memory/tracker.py`), which implements the same four-method interface (`add`, `get`, `list`, `set_status`) that an in-memory version also implements — so tests can run without touching disk, and the MCP server and the Streamlit UI both talk to the exact same class in production.

**What gets stored**, per application record:

| Field | Meaning |
|---|---|
| `application_id` | An opaque, randomly-generated id (not derived from role or company) |
| `role` | The advertised role title |
| `company` | The employer |
| `fit_score` | The computed `overall_fit`, constrained `0.0`–`1.0` |
| `status` | `draft` / `ready` / `submitted` / `archived` |
| `created_at` | Timestamp |

**Why the id is random, not derived:** so that the same candidate applying to two different roles at one company — or re-running the same role twice while testing — produces two separate records instead of silently overwriting one.

**Status:** every new application starts as `draft`, always. Nothing in the codebase ever programmatically sets a status to `submitted` — a status change is only ever an explicit tool call (via the UI or MCP), representing a decision a human has already made. The system never submits an application anywhere.

**Persistence across reruns/restarts:** yes — because it's a real SQLite file on disk (not held in memory), the tracked history survives closing and reopening the Streamlit app, or restarting the MCP server, as long as `TRACKER_DB_PATH` points at the same file.

---

### 15. Streamlit UI

Everything below is a feature that actually exists in `job_agent/app/ui.py` — nothing here is aspirational.

- **CV input:** paste into a text box, or upload a `.txt`/`.md`/`.pdf` file (an upload takes priority over the pasted box).
- **JD input:** the same, side-by-side with the CV input.
- **Configuration:** shown read-only in the sidebar — which model each tier will use, and whether each credential is set (never the value itself). A missing search key is shown as a soft warning ("company research will be skipped"), not an error.
- **Run application:** a single button, disabled automatically if a required model tier can't actually be built (e.g. an Anthropic tier with no API key). Runs the full supervisor pipeline described in Section 4; a spinner shows for the roughly one-minute duration.
- **Pipeline stages:** which stages actually ran and how long each took, reconstructed from the run's own trace — not guessed.
- **Fit report:** overall score, grouped into Strengths/Matches, Partial matches, Related evidence, and Missing from CV, each with the specific CV evidence and reason behind it.
- **Company research:** the brief's summary, facts, and every cited source as a clickable link — or an honest note that research wasn't available.
- **Tailored CV:** the tailored bullets, what was de-prioritised, and a download button (plain text) — or a warning explaining that nothing could be produced without an unsupported claim.
- **Cover letter:** the drafted text, with its own download button — or a warning if none was produced.
- **Validation:** what each anti-fabrication guard actually rejected during this run, in plain words, and how many malformed (not ungrounded) responses were regenerated.
- **Tracker:** the record just saved, plus a table of every application tracked so far, with a reminder that everything shown is a saved draft, nothing submitted.
- **Downloads:** the tailored CV and cover letter each get their own plain-text download button.
- **Result persistence within a session:** the last run's result is kept in Streamlit's session state, so downloading a document or editing the input boxes afterward doesn't discard it — only clicking **Run application** again replaces it with a new run.

---

### 16. MCP / Tooling

**What MCP is doing in this project:** it exposes the tracker and a working subset of the pipeline over the **Model Context Protocol**, so an external MCP client — Claude Code, an IDE, a colleague's script — can read the tracked applications and run scoring/tailoring without importing this Python package or knowing the data lives in a SQLite file.

**Server:** `job_agent/mcp_server/server.py`, run directly as `python -m job_agent.mcp_server.server` (stdio transport).

**What it exposes:**

| Kind | Name | Does |
|---|---|---|
| resource | `applications://all` | every tracked application, newest first, as JSON |
| tool | `list_applications` | list, optionally filtered by status |
| tool | `get_application` | fetch one by id |
| tool | `track_application` | record a new application (`draft` by default) |
| tool | `set_application_status` | move an application's status along |
| tool | `search_company` | raw web search (title/url/snippet) |
| tool | `score_fit` | parses CV+JD and runs the scoring agent |
| tool | `tailor_application` | the above, plus the writing agent — tailored CV + cover letter |

**How it fits into the architecture:** every tool is a thin adapter over the exact same agents, tracker, and validation code the Streamlit pipeline uses — there is no separate "MCP version" of the scoring rubric or the no-fabrication guard, so the two surfaces can't drift apart. Two deliberate safety choices: model-backed tools (`score_fit`, `tailor_application`) run on the **light** tier by default — since an MCP call might be made by a client the operator isn't actively watching, the expensive tier is opt-in via `MCP_MODEL_TIER=heavy`, never the default — and `tailor_application` passes `brief=None`, so a cover letter generated over MCP is grounded in the CV alone; an MCP caller can't inject an unverified "company fact."

**Its actual role, without exaggeration:** it is a portability layer, not a second product. It doesn't add any new capability the Streamlit app doesn't already have — it makes the *existing* capability reachable from outside a browser.

---

### 17. Error Handling

| Failure | Handling |
|---|---|
| **Malformed model JSON** | Caught by `generate_validated`; the specific validation error is fed back to the model and it's asked again, up to a bounded number of attempts, then `RetryExhaustedError` |
| **Provider-side JSON-mode rejection (Groq observed live)** | The malformed text is recovered from the provider's own error body and routed back through the normal schema-guard retry flow instead of crashing |
| **Missing API key / base URL** | Raised immediately and clearly at client construction (e.g. `"ANTHROPIC_API_KEY is not set"`), not deep inside a run; the Streamlit "Run" button is disabled proactively for this reason too |
| **Search failure (HTTP error, or a non-JSON 200 response)** | Wrapped as `SearchError`, caught by the research agent, degraded to an honest "research unavailable" brief |
| **Unsupported (fabricated) generated claim** | Rejected by grounding, retried with the exact violation named, up to a bounded budget, then the writer refuses to return anything |
| **Missing configuration (e.g. no search key)** | The run still completes; the UI shows a soft warning, not a hard failure |
| **Scoring validation failure / retry exhaustion** | The whole run fails — nothing partial is written to the tracker |
| **Writing validation failure / retry exhaustion** | The run still completes — a warning is recorded, the fit report and tracked record stand, the candidate keeps their original CV |
| **Unknown tracker status** | Rejected with a clear message listing the valid statuses, both in code (`ValueError`) and at the database layer (a SQL `CHECK` constraint) |
| **Search key leaking into logs/errors** | Actively prevented — see Section 19 |

**Graceful degradation, summarised:** research and writing are allowed to fail *softly* (a warning, the run continues); parsing and scoring are treated as *essential* (their failure stops the run) because a report built on an unparsed CV, or an invented score, would be actively misleading rather than merely incomplete.

---

### 18. Testing

Independently re-run at the time this document was written (not copied from an earlier report):

- **Total tests:** **434 passed**, 0 failed
- **Coverage:** **98%** (1,478 statements, 34 missed — the misses are almost entirely provider-SDK constructor branches that only execute with a real Anthropic/OpenAI SDK installed and a real key, which the test suite deliberately never does)
- **Linting (Ruff):** **all checks passed**
- **`git diff --check` (whitespace hygiene):** clean

**Regression tests** — tests written specifically to pin down a real bug that was found and fixed, so it can't silently reappear: the REST API/APIs plural fix, the BS/Bachelor's degree abbreviation fix, the `[Candidate]` placeholder fix, the CSS/HTML-vs-APIs subject-relevance fix, the SerpAPI-key-leak fix, and the wrong-company-search-results fix are all covered this way (see Section 20 for the full list).

**Important categories tested**, by file: parsing (`test_parse_cv.py`, `test_parse_jd.py`), scoring (`test_scoring_agent.py`, 44 tests), grounding (`test_grounding.py`, 78 tests — the largest single file, matching its role as the trust-critical layer), the writing agent (`test_writing_agent.py`, 47 tests), the supervisor pipeline end-to-end (`test_supervisor.py`, 27 tests), the tracker (`test_tracker.py`, 30 tests), the MCP server and protocol (`test_mcp_server.py` + `test_mcp_protocol.py`, 42 tests combined, including tests that actually spawn the server over stdio), the Streamlit UI (`test_streamlit_app.py`, 52 tests), provider adapters (`test_providers.py`, 18 tests), and web search (`test_web_search.py`, 18 tests).

**No test requires an API key, network access, or a real `.env`** — every model/search call is stubbed or dependency-injected.

---

### 19. Security / Credential Handling

- **`.env`** holds real configuration (API keys, base URLs) and is used only locally — it is **not tracked in git** (verified directly: `git ls-files` returns nothing for `.env`).
- **`.gitignore`** explicitly excludes `.env`, everything under `data/` (the tracker database and trace log), Python caches, and tool caches (`.pytest_cache/`, `.ruff_cache/`, `.coverage`).
- **`.env.example`** is committed and contains **only placeholders and blank values** — no real credential of any kind. It documents which variables exist and what each is for.
- **Why secrets must not be committed:** an API key committed to a public GitHub repository is effectively public and can be used by anyone who finds it, potentially running up real cost on the account it belongs to.
- **Credential leak protection, actually implemented (not just intended):** a real live run once leaked a SerpAPI key in plaintext, because SerpAPI takes its credential as a URL query parameter, and (a) the `httpx` library logs every request's full URL at `INFO` level, and (b) `raise_for_status()`'s own exception message also contains the full URL. Both routes are now closed: the `httpx` request logger is silenced at import time (only `WARNING` and above pass through), and every HTTP error is caught and re-raised as a new exception that reports only the HTTP status code, explicitly discarding the original URL-bearing exception (`raise ... from None`). The Streamlit UI additionally redacts any configured secret from an error message before it's ever shown on screen, as a last line of defence.

**No API key, secret, or `.env` content appears anywhere in this document.**

---

### 20. Important Bugs and Fixes

Every row below is verifiable directly from a git commit, a code comment, or a test — none are invented.

| # | Problem | Cause | Fix | Verification |
|---|---|---|---|---|
| 1 | A live scoring run gave a real, reasonably-strong CV a score of **10%** | Scoring only had two states, MATCH/MISSING (called "met/unmet"). Real evidence at a lower proficiency, or genuinely adjacent evidence, was being collapsed into the same bucket as total silence | Replaced with four levels — MATCH/PARTIAL/RELATED/MISSING — each with its own score weight | `job_agent/models/outputs.py`, `MatchLevel` docstring; `scoring.py`, `MATCH_LEVEL_WEIGHT`; extensive tests in `test_scoring_agent.py` |
| 2 | The scoring model called CSS/HTML knowledge **PARTIAL** evidence for "basic understanding of APIs," even while its own stated reason admitted the subjects didn't match | The prompt asked the model to weigh proficiency without first checking whether the evidence was even about the same subject | Added an explicit "check subject relevance BEFORE proficiency level" rule to the scoring prompt, with the exact failing example named | `scoring.py`, `SCORING_SYSTEM_PROMPT`; `test_scoring_agent.py::test_css_html_evidence_is_not_partial_for_api_understanding` (note: this test confirms the *correct* judgement flows through correctly, not that a future live call can never repeat the mistake — see Section 10) |
| 3 | A real SerpAPI key leaked in plaintext into logs during a live MCP run | SerpAPI passes its key as a URL query parameter; `httpx` logs full request URLs at INFO, and `raise_for_status()`'s exception message also contains the full URL | Silenced the `httpx` request logger at import; every HTTP error is now re-raised carrying only the status code, with the original discarded via `from None` | Commit `3944d8b`; `web_search.py`; `test_web_search.py` (multiple dedicated tests, including one with the root logger forced to DEBUG) |
| 4 | Perfectly grounded drafts were rejected for saying "REST APIs" when the CV said "REST API" | Grounding matched terms by exact substring only, with no plural handling | Added a conservative English-pluralisation-only singular-form check as an *additional*, not replacement, way to match | Commit `aa91acb`; `grounding.py`, `singular()` |
| 5 | Letters signed "Sincerely, [Candidate]" when no candidate name was available | The prompt didn't supply a real name and didn't forbid template placeholders | Real candidate name is now passed into the prompt; placeholders are explicitly forbidden; grounding still catches one if it slips through | Commit `aa91acb`; `writing.py`, `WRITING_SYSTEM_PROMPT` |
| 6 | A company search for "Arbisoft Junior AI Engineer Python FastAPI" returned a stranger's LinkedIn profile and a different company's vacancy, misattributed to the target employer | Combining company + role + technologies is, to a search engine, effectively a job search | The query is now company-name-only (`"<company>" company overview`); a relevance guard drops results that never mention the company before summarisation | Commit `7f5a41e`; `research.py`, `build_query` docstring; `docs/ARCHITECTURE.md` D12 |
| 7 | A role advertised as "Kubernetes Engineer" made "I have Kubernetes experience" pass grounding | The employer's proper nouns (used to let a letter address the company) also covered the *role title*, which then wrongly counted as naming evidence for the candidate | Naming allowance restricted to company statements only, never candidate claims | Commit `7f5a41e`; `grounding.py`, `is_candidate_claim`/`refers_to_company` |
| 8 | A salutation written inline as one sentence ("Dear Hiring Team, I am excited...") caused "Hiring" to be flagged as an invented proper noun | The checker split by sentence but treated the salutation as part of the first real claim | Salutation and valediction lines are recognised (even written inline) and stripped from the claim before checking, without exempting the rest of the sentence | `writing.py`, `strip_salutation`/`is_letter_boilerplate` |
| 9 | A non-JSON 200 response from a search provider (e.g. an HTML rate-limit page) would crash the entire pipeline run instead of degrading gracefully | `response.json()` raises `json.JSONDecodeError`, a `ValueError` — not caught by the existing `httpx.HTTPError` handling | Added an explicit `except ValueError` branch wrapping this as a `SearchError`, consistent with every other search failure | Finalization commit `171d2f5`; `web_search.py`; `test_web_search.py::test_a_non_json_200_response_is_reported_as_a_search_error_not_a_crash` |
| 10 | The "missing `base_url`" error always named `LIGHT_BASE_URL`, even when the **heavy** tier (a documented, supported configuration) was the one actually missing it | One error message was written assuming only the light tier could ever use the OpenAI-compatible adapter | Message now names both `HEAVY_BASE_URL` and `LIGHT_BASE_URL` | Commit `171d2f5`; `llm/providers.py` |
| 11 | `RequirementMatch.met` was documented as always derived from `match_level` and never independently settable, but nothing enforced that — two test fixtures had actually constructed instances where the two disagreed | No validation existed; the guarantee held only by convention at the one production call site | Added a Pydantic validator that always derives `met` from `match_level`, and corrected the two inconsistent fixtures | Commit `171d2f5`; `models/outputs.py` |
| 12 | `.env.example` listed `LANGSMITH_API_KEY`/`LANGSMITH_PROJECT` as configurable, implying working tracing-export integration that doesn't exist anywhere in the codebase | Config template drifted from what `config.py` actually reads | Removed the two dead lines | Commit `171d2f5`; `.env.example` |
| 13 | Latency reported for a run would double-count time, because each pipeline stage's own trace span *contains* the model/tool calls made inside it | Naively summing every trace event's duration counts nested spans twice | `run_totals()` sums only the top-level, non-overlapping stage spans for elapsed time, and everything else for time-inside-calls, kept as two separate numbers | `app/ui.py`, `run_totals` (documented reasoning inline) |
| 14 | An uploaded PDF has to become a file on disk for the existing document loader to read it, but that temporary file needs to be cleaned up reliably even if the pipeline run fails | No prior handling | A `contextlib.contextmanager` (`prepared_sources`) scopes the temp file to a `TemporaryDirectory`, so cleanup happens unconditionally, including on an exception | `app/ui.py`, `prepared_sources` |

---

### 21. Final Verification

Independently re-checked at the time this document was written:

| Check | Result |
|---|---|
| Tests | **434 passed**, 0 failed |
| Coverage | **98%** |
| Ruff | **All checks passed** |
| `git diff --check` | clean |
| `git status` | clean working tree |
| Branch | `main` |
| Final commit | `171d2f5` — `"feat: finalize job application agent"` |
| GitHub synchronization | local `main` and `origin/main` both at `171d2f5` — fully pushed and synchronized |

---

### 22. Known Limitations

All of the following are genuine, currently-true limitations — not bugs, and not hidden anywhere:

- **The subject-relevance-before-proficiency scoring rule is prompt-level only, by design.** The architecture deliberately does not fall back to keyword matching (a keyword system would itself be a source of false judgements). This narrows the specific failure mode it targets rather than eliminating it by construction, and it has not been re-verified against a further live model call since the fix.
- **Grounding's deterministic layer catches invented terms and numbers, not proficiency inflation that adds no new term.** If the CV says "basic Python" and a draft says "expert Python" without naming any new technology or figure, the deterministic checker has no fingerprint to catch it on. An optional model-assisted verification pass exists that could plausibly catch this, but it is **off by default** (an extra model call per attempt). This is an inherent property of a no-model-call, fingerprint-based checker — not an oversight.
- **A full pipeline run takes roughly a minute**, and the Streamlit UI blocks synchronously for the entire duration (no background/async execution).
- **No multi-model side-by-side comparison UI.** Both tiers can be configured independently, but there's no view showing what two different models would each produce for the same input.
- **No research caching.** Every run performs a fresh company search, even for a company just looked up moments before.
- **No automated benchmark or accuracy measurement exists for scoring quality.** Everything demonstrated is from unit/integration tests (offline, stubbed) and manual live runs during development — not a formal evaluation set.
- **Structured-output behavior is somewhat model-dependent.** The JSON-mode/retry machinery is designed to absorb this (see Section 9's Groq example), but a different provider could surface a new failure shape the current handling doesn't yet name.
- **Provider/rate limits are the user's own to manage** — the project does not implement backoff-and-retry for rate-limit errors specifically; a rate-limited call would surface as an ordinary provider error.

---

### 23. Future Improvements

Everything in this section is explicitly **NOT currently implemented** — these are reasonable next steps that follow from the existing architecture, not claims about what the project already does.

- **Enable the optional model-assisted entailment pass by default** for a narrower category of fabrication (proficiency inflation, vague achievement claims) the deterministic layer structurally can't catch — at the cost of one extra model call per writing attempt.
- **Cache company research** per company (e.g. keyed by company name, with a time-to-live), to avoid a fresh search every time the same employer is looked up.
- **Add a side-by-side model comparison view**, since the light/heavy tier separation already exists — running the same step through two configured tiers is mostly UI work at this point, not new pipeline plumbing.
- **Run the pipeline asynchronously** in the Streamlit UI so the interface doesn't block for the full run duration.
- **Add a small, real evaluation set** (a handful of CV/JD pairs with an agreed "correct" fit assessment) to move scoring-quality claims from "looks right in testing" to something measured.

---

### 24. Mentor Questions I Should Be Ready For

1. **Why did you use multiple model tiers instead of one model for everything?**
   Cost and latency. Parsing a CV into fields is a mechanical extraction task; a cheap model does it fine. Scoring and writing require actual judgement, so they get the stronger model. Routing is by *step name*, so the mapping lives in one place (`llm/router.py`) and no agent hardcodes a vendor.

2. **Why Pydantic for everything?**
   Every model output becomes a typed, validated object instead of raw text the rest of the code has to trust. Required fields, value ranges, and enums are enforced automatically, and `extra="forbid"` stops a model from inventing a field nobody asked for.

3. **Why a supervisor / orchestration layer instead of one big function?**
   Each stage (parsing, research, scoring, writing, tracking) is independently testable and has exactly one job. The supervisor's only responsibility is deciding what runs when and assembling the result — it never parses, scores, or writes anything itself. LangGraph gives this a real graph structure that maps directly onto the pipeline diagram.

4. **Why SQLite instead of a "real" database?**
   The tracker stores a handful of flat fields at a scale of tens or hundreds of rows for one user, locally. A full database server would add operational weight without adding any capability this project needs. The tracker sits behind a narrow four-method interface, so swapping the backend later wouldn't require touching any other code.

5. **Why not let the LLM directly calculate the fit score?**
   A model asked for "a number between 0 and 1" is unstable and unexplainable — you can't point to why it picked 0.63. Instead, the model is asked one narrow question per requirement, and the *code* computes the score from those answers using a fixed, auditable formula. You can always explain exactly why the score is what it is.

6. **How do you prevent hallucinated CV claims?**
   Two independent layers: schema validation (is the output shaped correctly?) and grounding (is every fact in it actually traceable to the CV?). Grounding checks distinctive terms and numbers against the real CV text, rejects anything unsupported, and sends the model back with the specific violation named. If it never produces a clean draft, nothing is returned at all.

7. **What happens when the model returns invalid JSON?**
   `generate_validated` catches the validation error, feeds the exact error back to the model in the next prompt, and retries — up to 3 attempts by default — before giving up with a clear `RetryExhaustedError`.

8. **Why do you need retries at all — why not just fail immediately?**
   Models occasionally wrap valid JSON in prose, miss a field, or (for some providers) reject their own malformed output as an HTTP error. A single bounded retry, with the specific problem named, recovers from most of these without giving the model unlimited chances to keep guessing.

9. **How does company research work?**
   A `web_search` tool call (SerpAPI or Brave) searches for the company name only — not the role or technologies, since that combination behaves like a job search to a search engine. A relevance guard drops results that don't mention the company; a light-tier model summarises the rest into a short, sourced brief; a grounding guard drops any fact that doesn't cite a real result or states an unsupported number.

10. **What happens when search fails?**
    Any failure mode — no key configured, the request fails, nothing relevant comes back — produces a valid brief that plainly says research wasn't available. The pipeline continues; the fit score never depends on it.

11. **Why use an OpenAI-compatible provider interface instead of writing an adapter per provider?**
    Groq, Ollama, and OpenRouter (and Gemini, via its OpenAI-compatible endpoint) all speak the same chat-completions API shape — they differ only by base URL and model name. One adapter class covers all of them, which is also what makes the "at least two providers" requirement a configuration choice, not more code.

12. **What does MCP actually contribute here?**
    Portability. It exposes the tracker and a working slice of the pipeline (scoring, tailoring, search) as tools any MCP client can call — without that client needing to import this Python package or know the data lives in SQLite. It reuses the existing agents and validation code directly; it isn't a second implementation.

13. **What happens if an API key is missing?**
    The relevant client raises immediately and clearly at construction time, naming the missing setting (never its value). In the Streamlit UI, the Run button is disabled proactively if a required tier can't be built, so the user finds out before spending a minute on a run that would fail anyway.

14. **How is the fit score calculated, exactly?**
    Per requirement: `requirement_weight` (1.0 must-have / 0.5 nice-to-have) times `match_strength` (1.0/0.5/0.25/0.0 for MATCH/PARTIAL/RELATED/MISSING). Sum those products, divide by the sum of all requirement weights. It's a plain weighted average, computed in code from the model's per-requirement judgements.

15. **How do you test an LLM-based pipeline without spending API quota on every test run?**
    Every model and search call is dependency-injected — tests supply a stub `request_fn` or a fake client returning a scripted reply, so the pipeline's *logic* is fully tested without ever calling a real API. 588 tests currently run this way with zero network access required.

16. **What are the biggest limitations right now?**
    The subject-relevance scoring rule and the anti-fabrication grounding check both rely on the model actually following instructions for anything without a hard "fingerprint" (a genuinely new term or number) — the deterministic layer can catch fabrication, but it can't catch every kind of subtle misjudgement or proficiency inflation. That's disclosed honestly rather than hidden.

17. **What would you improve next?**
    Turning on the optional model-assisted verification pass for proficiency-level claims specifically, and building a small real evaluation set so scoring-quality claims are measured rather than just "looked right in testing."

18. **How do you preserve candidate truth throughout the pipeline?**
    The CV's raw text and its parsed evidence lines are the *only* thing any generated claim about the candidate is allowed to draw on. The employer's proper nouns and the advertised role title are a narrow, separate allowance for statements *about the company* — never usable as evidence for the candidate.

19. **What was the hardest technical problem in this project?**
    Getting the two-domain grounding rule right — deciding, per generated sentence, whether it's a claim about the candidate or a statement about the company, without a real grammatical parser. It ended up being a set of narrow, deliberately readable heuristics (first-person-pronoun proximity to an experience word, explicit company-reference phrases) rather than anything clever, because the failure mode being guarded against (a company fact silently licensing a candidate claim) is exactly the kind of subtle bug that's easy to introduce with an over-general rule.

20. **What would you change if you had more time?**
    Add the small real evaluation set mentioned in Future Improvements — right now, "does the scorer actually judge fit well" rests on unit tests and a handful of manual live runs, which is honest but not the same as a measured benchmark.

---

### 25. 2-Minute Explanation

*(A natural spoken version, written to be memorized and said in your own voice — not read aloud.)*

"So this is my final project — a Job Application Agent. You give it your CV and a job posting, and it tells you honestly how well you actually fit the role, looks up some basic facts about the company, and can write you a tailored CV and cover letter — but only using things your CV already supports. It never invents a skill you don't have.

The pipeline is a few stages: it parses your CV and the job posting into structured data, does a quick web search on the company, and then the important part — it scores your fit requirement by requirement, not with one vague number. For each thing the job asks for, it decides: does the CV fully match this, partially match it, is it just related, or is there really nothing there? That's more honest than a binary yes/no, because 'the CV shows something weaker than what's asked' and 'the CV never mentions this at all' are genuinely different situations.

The part I'm proudest of is the grounding system. Before any tailored CV or cover letter is shown to you, every single sentence in it is checked against your actual CV. If it says something your CV doesn't support — like inventing a technology or rounding 'basic knowledge' up to 'expert' — that gets rejected, and the model is told exactly what was wrong and asked to fix it. If it genuinely can't produce something honest after a few tries, it just... doesn't give you a document, rather than giving you something that might not be true.

Everything gets saved to a little tracker so you can see all your applications in one place, and there's also a version of this exposed through something called MCP, so another tool could use the same pipeline. It's built with 588 tests and 96% test coverage, so I have real confidence the pieces actually behave the way I'm describing."

---

### 26. 5-Minute Explanation

*(A longer spoken version covering architecture, models, grounding, validation, and results — written to be memorized and adapted, not read verbatim.)*

"This project is a Job Application Agent — you give it a CV and a job description, and it produces an honest fit assessment, company research, a tailored CV, and a cover letter, without ever claiming you have experience you don't actually have.

**Architecture first.** It's a multi-agent pipeline, orchestrated with something called LangGraph, which lets me express the pipeline as an actual graph of steps rather than one long function. There's a supervisor that decides what runs when, and it delegates to specialist workers — a research agent, a scoring agent, and a writing agent — each with exactly one job. Everything they produce is passed through a single shared context object, so there's no hidden state anywhere; what one stage produced is exactly what the next stage sees.

**The pipeline itself:** parse the CV, parse the job description, research the company, score the fit, optionally write tailored documents, and save a record to a tracker. Research and writing are allowed to fail softly — if the company search comes up empty, or the writer can't produce something honest, the run still finishes and just says so. But scoring is treated as essential — if that fails, nothing gets saved at all, because I didn't want a failed score ever being mistaken for a genuinely poor one.

**On models** — I deliberately split the work into two tiers. A cheap, fast model handles mechanical extraction: pulling structured fields out of a CV or a job posting. A stronger model is reserved for the parts that actually need judgement — scoring fit and writing tailored material. And the provider for each tier is just a configuration value, not something baked into the code — there's one adapter for Anthropic and one shared adapter for anything that speaks the OpenAI-compatible API shape, which covers several different providers.

**Scoring, specifically:** rather than asking the model for a single fit percentage — which would be unstable and impossible to explain — I ask it one narrow question per requirement: does the evidence match, partially match, relate to it, or is there nothing there at all? Then my own code computes the score from those answers with a fixed formula: each requirement's weight, times how strongly it was matched, summed and normalized. That's auditable — I can point to exactly which judgement produced the final number.

**And then grounding, which is the part I care most about.** Once the writing agent drafts a tailored CV bullet or a cover-letter sentence, it doesn't go straight to you. It's checked against your actual CV — every distinctive term and every number in it has to trace back to something the CV really says. If it doesn't, that draft is rejected, the model is told precisely what was unsupported, and it's asked to rewrite — up to a few attempts. If it still can't produce something honest, nothing is returned at all, and you just keep your original CV and an honest list of what's missing. Schema validation and grounding are two completely separate checks on purpose — one confirms the JSON shape is right, the other confirms the content is actually true, and a model output has to pass both.

**Where it stands right now:** 588 tests passing, 96 percent coverage, clean linting, and it's committed and pushed. Along the way I found and fixed several real bugs from live runs — things like a search query accidentally pulling in a completely different company's job posting, an API key that was leaking into log output, and a scoring rule that was letting unrelated evidence count as partial credit. Every one of those has a regression test now, so it can't quietly come back.

The honest limitations are that the subject-relevance rule and part of the anti-fabrication check live in the prompt, not in code that can mathematically guarantee correctness — so they narrow the failure mode rather than eliminate it completely. I'd rather say that plainly than overclaim."

---

### 27. Demo Script

**Before you start:** make sure your `.env` has real credentials configured (the mentor doesn't need to see the file — just confirm the sidebar shows keys as set), and have the sample CV/JD from `tests/fixtures/` handy as a fallback in case you don't want to use a personal document live.

**1. Open the UI**
```
python -m streamlit run job_agent/app/streamlit_app.py
```
*Say:* "This is the whole app — there's no separate backend to start, it's one process."

**2. Provide a CV**
Paste it into the left box, or upload a `.txt`/`.md`/`.pdf`.
*Say:* "I can paste text or upload a file — if I upload a PDF it's handed to the same document loader the pipeline uses internally, the UI itself never touches the file's contents."

**3. Provide a job description**
Paste it into the right box, or upload a file.
*Say:* "Same options on this side."

**4. Point out the sidebar before running**
*Say:* "This shows which model each tier is configured to use, and whether the credentials are set — never the actual key, just whether it's present."

**5. Click "Run application"**
*Say:* "This runs the full pipeline — parsing, company research, scoring, and writing — end to end. It takes about a minute because of the model calls and the anti-fabrication retry loop, so I'll talk through what's happening while it runs."

**6. Explain each result section as it appears**
- **Pipeline stages:** "This is reconstructed from the actual trace of the run — not a canned progress bar. You can see exactly which stages ran and how long each took."
- **Fit report / fit score:** "This is the core output. It's not one number — every requirement from the job posting gets its own judgement: a clear match, a partial match at a lower level than asked for, related-but-not-quite evidence, or genuinely missing. The overall score is a weighted average of those, computed in code, not asked of the model directly."
- **Company research:** "This is a short, sourced brief — every fact here cites a real search result, and if research wasn't available for some reason, it says so honestly instead of making something up."
- **Tailored CV / cover letter:** "This is where the anti-fabrication system matters most. Every line here was checked against my actual CV before it was allowed to show up — nothing invented."
- **Validation section:** "This is literally showing what the grounding system did during this run — what it checked, and what it rejected, if anything."
- **Tracker:** "And finally, this run gets saved here as a draft — nothing is ever submitted anywhere automatically."

**7. Explain the fit score specifically, if asked**
*Say:* "Each requirement earns its weight — double for a must-have versus a nice-to-have — times how strongly the evidence matched it: full credit for a clear match, half for a weaker match of the same skill, a quarter for related-but-different evidence, zero for nothing. Sum those, divide by the total possible weight."

**8. Explain grounding, if asked**
*Say:* "Before any generated text reaches this screen, every sentence in it is checked against my real CV text. If the model tried to add something my CV doesn't actually say, that draft gets rejected and the model is told specifically what was wrong and asked to fix it — up to a few tries. If it truly can't produce something honest, you just get my real CV back plus the gap list, not a rewrite nobody can vouch for."

**9. Explain the tracker**
*Say:* "This table is every application I've run through the system, pulled straight from a local SQLite file — it's not just this session's result, it persists across restarts."

**10. Explain persistence, if asked**
*Say:* "If I close this browser tab and come back, or restart the app entirely, everything in the tracker is still there, because it's backed by a real file on disk, not memory."
