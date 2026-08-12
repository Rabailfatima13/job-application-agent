# Week 5.5 – prompts.md

## Prompt 1 – Project Setup & Initial Architecture

I am starting a completely separate project for my internship's final AI project.

I am attaching my approved project proposal to this conversation. Please read the proposal carefully before making any changes. The proposal is the source of truth for the project's scope, requirements, architecture, technologies, agent responsibilities, validation strategy, testing strategy, and Week 6–8 timeline.

The project is the Job Application Agent — a multi-agent system that, given a job description and a CV, researches the target company, scores candidate–role fit, tailors the CV emphasis, drafts a grounded cover letter, and tracks the application.

Requirements:
- Do NOT modify my existing Week 4 / Week 5 internship repository; this must be a new, independent project.
- Week 5 code may be inspected for patterns only.
- Keep the no-fabrication trust requirement in mind from the very beginning of the design.
- Establish clean interfaces for agents, tools, structured outputs, memory/tracker, validation, and model clients.
- Make it easy to add or replace models later (Week 7 requires multi-provider routing).
- Do not over-engineer; prefer simple, testable, modular, demonstrable code.
- Every major component must be testable, starting now.
- Use meaningful commits; no single giant commit.

Before writing substantial application code:
1. Summarize your understanding of the project.
2. Identify the exact Week 6 core milestone.
3. Identify which existing Week 5 patterns are useful as reference.
4. Propose the new project's directory structure.
5. Explain the implementation order.
6. Identify any important technical decision that needs to be made now.
7. Then create the initial project scaffold only.

Stop after the scaffold and show me what was created, what each folder is responsible for, which dependencies were added and why, the first Week 6 milestone, how to run the project, and what to implement next.

---

# Week 6 – prompts.md

## Prompt 1 – CV and Job Description Parsing Layer

We are now moving from project scaffolding into the actual Week 6 implementation. The scaffold is already created and tested — do not redesign the architecture or restart the project. The approved proposal remains the source of truth.

Implement ONLY the parsing layer.

Requirements:
- Implement `parse_jd`, producing the existing `JobDescription` model: role, company if available, requirements, responsibilities, skills/technologies.
- Implement `parse_cv`, producing the existing `ParsedCV` model: skills, experience, projects, education, achievements.
- Preserve the candidate's original wording — the Week 7 no-fabrication validator compares generated claims against this evidence. Do not paraphrase or embellish.
- Do not invent information absent from the source document.
- Use the existing input handling conventions; do not add a second input system or unnecessary dependencies.
- Use the existing model abstraction and light-tier routing; do not hard-code a provider in the parsers.
- Validate every parser result through the existing Pydantic/schema guard, with bounded retry/fallback. No free-form model output may leak into the rest of the app.
- Register both tools through the existing ToolRegistry; do not create a second registration system.
- Add focused unit tests for both parsers using the existing fixtures and a deterministic fake model — no live API calls in tests.

Do NOT implement: Research Agent, Scoring Agent, Writing Agent, tailored CV, cover letter, no-fabrication grounding, MCP server, UI changes, multi-model comparison, vector memory.

Then run the full test suite, the parser tests and lint, and report the files changed, how each parser works, how they use the existing architecture, the tests added, the results, any limitations, and the next recommended Week 6 step.

Stop after the parsing layer is complete and verified.

---

## Prompt 2 – Persistent SQLite Application Tracker

We are continuing the Week 6 implementation. The parsing layer is complete and verified (55 tests passing, ruff clean). Do not redesign the existing architecture.

Implement the SQLite-backed `ApplicationTracker` described in the proposal.

Requirements:
- Implement the concrete `SQLiteApplicationTracker` behind the EXISTING tracker interface; do not create a second tracker abstraction or a parallel API.
- Use the existing `ApplicationRecord` model and the existing status enum (draft/ready/submitted/archived); do not duplicate the schema or add statuses.
- Records must survive process restarts: write in one run, reopen the same SQLite file in another, and the application is still there.
- Use SQLite directly. Do not add an ORM. Keep it simple and local.
- Initialize the database/schema safely if it does not exist.
- Do not hard-code the production database path; use configuration/injection so tests use temporary databases.
- Invalid records must not silently enter the database; handle reasonable DB errors cleanly without over-engineering.

Tests must cover create, read, list, status update, persistence across tracker instances, and schema constraints (invalid fit scores, statuses, missing required fields). Tests must not touch a developer's real application database.

Do NOT implement: Supervisor, Research/Scoring/Writing agents, grounding, MCP server, Streamlit tracker UI, multi-model comparison, vector memory, new DB abstraction, ORM.

Run `python -m pytest -q` and `python -m ruff check .`, keep the suite green, then report files changed, how the tracker works, the schema, the operations, tests added, results, design decisions/limitations, and the next recommended Week 6 step.

Then STOP. Do not automatically implement the scoring agent.

---

## Prompt 3 – Scoring Agent

We are continuing the Week 6 implementation. Parsing and the persistent SQLite tracker are complete (78 tests passing, ruff clean). Do not redesign the architecture.

Implement only the Scoring Agent and its tests. It must consume `ParsedCV` + `JobDescription` and produce the existing `FitReport`.

Requirements:
- Match each job requirement against the structured CV evidence, decide whether it is met, provide the evidence, identify gaps, produce recommended CV emphasis, and an overall fit score in [0,1].
- Evidence must originate from the existing `ParsedCV` data. Do not invent candidate evidence, and do not treat the raw LLM response as authoritative.
- Use the existing `FitReport` / `RequirementMatch` models; do not create a parallel schema or add fields for convenience.
- The score must be explainable, not "ask the LLM for a number": establish a rubric (e.g. matched weighted requirements / total weighted requirements) and calculate it in application logic. Document the rule.
- Keep requirements and responsibilities separate — responsibilities must not become scored requirements.
- Recommended emphasis must refer to existing CV evidence; the Week 7 Writing Agent will consume it.
- Use the existing ModelRouter and tier design; no direct provider imports. Use the existing validation/retry infrastructure.
- Add a deterministic evidence-validation layer: every match has evidence, `met=True` without valid evidence is not accepted, unsupported claims never enter the FitReport, score stays in [0,1].
- Tests with the existing stub model, including one where the expected score is calculable from known requirements (e.g. 3 of 5 → 0.6). Do not hard-code an LLM-generated number.

Do NOT implement: Research Agent, Supervisor, Writing Agent, tailored CV, cover letter, full Week 7 grounding, MCP server, UI, multi-model comparison, vector memory.

Run `python -m pytest -q` and `python -m ruff check .`; keep the suite green without weakening existing tests. Then report files changed, how the agent works, the exact scoring methodology, how evidence is validated, how the FitReport is generated, tests added, results, limitations, and the next recommended Week 6 step.

Then STOP. Do not automatically implement the Research Agent or Supervisor.

---

## Prompt 4 – Research Agent

We are continuing the Week 6 implementation. Parsing, the SQLite tracker and the Scoring Agent are complete (101 tests passing, ruff clean). Do not redesign the architecture.

Implement only the Research Agent and the web-search capability it strictly requires. It should consume the parsed job/company information and produce the existing `CompanyBrief`.

Requirements:
- Identify the company from the parsed JD, search the web, select useful factual findings, summarise them into `CompanyBrief`, and preserve source information.
- Keep the research role-relevant — what the company does, its products, its technology/domain, recent relevant information — not a general encyclopedia.
- Follow the same `BaseAgent` pattern as `ScoringAgent`; reuse the existing models, tools, llm, observability and configuration. No second model client or tool registry.
- Implement web search per the proposal (SerpAPI or Brave), configurable through environment/configuration. Never hard-code an API key.
- Separate search from summarisation: the tool returns structured results, the model summarises them. The search tool must not generate an LLM summary.
- Keep search results small and bounded (title, URL, snippet) to control latency and cost.
- Handle the existing `Unknown` company sentinel explicitly — do not search for "Unknown company" and do not guess the employer.
- Use the existing ModelRouter and tier design; use the existing validation/retry infrastructure; the final brief must be Pydantic validated.
- Add a lightweight deterministic grounding check: the summary must be grounded in the retrieved results. Do not build the Week 7 no-fabrication system here. Document the limitation.
- Caching: only if it fits naturally; otherwise document and defer.
- Tests must not call a real search API or need a key/network.

Do NOT implement: Supervisor, LangGraph workflow, Writing Agent, tailoring, cover letter, MCP server, UI, multi-model comparison, vector memory, full document validator.

Run `python -m pytest -q` and `python -m ruff check .`; keep the suite green without weakening existing tests. Then report files changed, the search provider chosen and why, how `web_search` works, how the ResearchAgent works, how results reach summarisation, how grounding is handled, tests added, results, configuration requirements, limitations, and the next Week 6 step.

Then STOP. Do not connect parse → research → score → track through the Supervisor yet.

---

## Prompt 5 – Supervisor and End-to-End Pipeline

We are now implementing the FINAL major Week 6 component: the Supervisor and end-to-end pipeline. All components are complete and tested (135 tests passing, ruff clean). Do not redesign the architecture and do not implement Week 7 features.

Implement the Supervisor connecting the existing components:

CV + JD → parse_cv → parse_jd → Research Agent → Scoring Agent → ApplicationRecord → SQLite tracker → final pipeline result.

Requirements:
- Use LangGraph, as already decided. The Supervisor orchestrates; it does not perform research or scoring itself.
- Inspect the existing `RunContext` first and extend state only if necessary.
- Accept the input formats the parsing layer already supports (raw text and file paths); do not create another document-loading system.
- Build the `ApplicationRecord` from the validated `FitReport`, starting in `draft`. Never auto-submit.
- Persist through the existing `ApplicationTracker` interface — no SQL in the Supervisor, and dependency injection so tests can use the in-memory tracker.
- Research failure must not stop the run: continue scoring and tracking, and preserve the warning.
- Scoring failure must not produce an invalid record or a defaulted score; surface the failure cleanly.
- Application IDs must allow the same candidate to apply to multiple roles at the same company.
- Return enough structured information for a future UI, preferring existing models.
- Preserve the existing tracing architecture so Week 8 can report latency and token usage.

Tests must not need a live LLM, live web search, or external services. Cover the happy path, tracker persistence, two JDs against one CV, research failure, unknown company, scoring failure, human-in-the-loop status, raw-text and file inputs, stage ordering, state hand-off, dependency injection, and the LangGraph workflow building, executing, reaching END and propagating failure.

Do NOT implement: Writing Agent, tailoring, cover letters, full grounding, MCP server, multi-model comparison, Streamlit UI, vector memory, auto-submission, job board integrations.

Run `python -m pytest -q` and `python -m ruff check .`, plus coverage if available. Update only the architecture documentation. Then report files changed, the LangGraph structure, Supervisor responsibilities, state, injection, failure behaviour, the ID strategy, E2E tests, test count, ruff and coverage results, limitations, and whether the Week 6 pipeline now works end to end.

Then STOP. Do not begin Week 7 automatically.

---
