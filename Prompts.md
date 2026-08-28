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

# Week 7 – prompts.md

## Prompt 1 – No-Fabrication Grounding and the Writing Agent

We are starting Week 7. Week 6 is complete and pushed to GitHub. Do not rewrite or unnecessarily refactor the Week 6 architecture, do not break any existing tests, and run the full suite first to record a baseline.

In this task implement ONLY: (A) the no-fabrication grounding system, (B) the writing agent, and (C) the minimum supervisor integration needed to test them. No MCP, no Streamlit UI.

**A — Grounding.** Implement the existing `job_agent/validation/grounding.py` stub. The writing agent must never invent information about the candidate. Generated claims must be supported by the parsed CV evidence / raw CV. Grounding must stay an independent layer from schema validation, because a schema-valid document can still be fabricated. It must detect invented jobs, projects, technologies, achievements, years of experience, education, certifications and numerical claims — but must NOT work by checking whether the generated sentence is an exact substring of the CV. Reordering, rewording, combining supported evidence and re-emphasis are all allowed. Return a structured result (passed / violations / checked claims). Prefer deterministic checks; if an LLM-assisted step is needed, isolate it behind the existing ModelClient abstraction, traced and testable, with no Anthropic SDK calls inside the grounding module.

**B — Writing agent** in `job_agent/agents/writing.py`, consuming `ParsedCV`, `JobDescription`, `CompanyBrief` and `FitReport` and producing a tailored CV and a cover letter as validated Pydantic output. The flow is: build prompt → heavy model → structured output → schema validation → grounding validation → accept or retry. A failed grounding check must not be silently accepted: retry explaining the violations, and if fabrication persists raise the project's existing retry exception rather than returning unsafe output.

**C — Supervisor.** Only after A and B are independently tested, make the minimum change so the writing stage can consume the Week 6 outputs. Keep application tracking intact and never auto-submit.

Tests must be offline with stub models and must cover: a grounded CV and cover letter passing; invented technology, years of experience, metric, project and certification each rejected; reworded and reordered evidence allowed; schema retry; schema-valid-but-ungrounded output rejected and retried; persistent grounding failure raising; writing model calls traced; grounding failures traced; and all existing Week 6 tests still passing unchanged.

Update `docs/ARCHITECTURE.md`, `Prompts.md` and the README where necessary, marking Week 6 complete and Week 7 in progress, and documenting why grounding is separate from schema validation. Run pytest, ruff and coverage. Do not push or open a PR. Do not commit `.env`, keys, databases, logs or personal documents.

Then STOP — no MCP, Streamlit, or multi-model comparison work.

---

## Note — Remaining Week 7 work (reconstructed from commit history, not a verbatim transcript)

The sessions between Prompt 1 above and Week 8 below were not logged verbatim at the time. This note summarises what they actually built, reconstructed honestly from the real git history so this log stays complete without fabricating prompt text that was never recorded:

- **`aa91acb` — fix: accept singular/plural variants in grounding, drop letter placeholders.** A live rehearsal rejected three correctly-grounded drafts because the CV said "REST API" and the writer wrote "REST APIs." Fixed with a conservative singular-form match (`singular()`) added as an *extra* way to match, never a replacement for the existing substring check. Also fixed "Sincerely, [Candidate]" by passing the real candidate name into the writing prompt and forbidding template placeholders outright. 222 tests passing, ruff clean, 94% coverage at that point.
- **`7f5a41e` — feat: complete week 7 core - grounded tailoring with two evidence domains.** Split grounding into two evidence domains (candidate CV vs. verified company brief) so a company fact (e.g. "Arbisoft uses Django") can support "your company uses Django" but never "I have Django experience." Closed a hole where the advertised role title itself counted as naming evidence ("Kubernetes Engineer" no longer licenses "my Kubernetes skills"). Fixed the research query to search for the company name alone rather than company + role + technologies, after a live run returned a stranger's LinkedIn profile and a different company's job posting. 283 tests passing, 94% coverage, verified live against real models and real web search.
- **`3944d8b` — fix: prevent search API key leakage in HTTP logs.** A live MCP run printed a real SerpAPI key in plaintext, because the key travels as a URL query parameter and `httpx` logs full request URLs at INFO. Fixed by silencing the httpx request logger at import and having every HTTP error report only its status code, with `from None` keeping the URL-bearing exception out of the traceback.
- **`3ccaa6f` — feat: complete MCP server integration.** Exposed the tracker (`applications://all` resource) and seven tools (`list_applications`, `get_application`, `track_application`, `set_application_status`, `search_company`, `score_fit`, `tailor_application`) as thin adapters over the existing agents — no business logic re-implemented. Model-backed tools default to the light tier so an MCP call never quietly bills the expensive provider. 44 tests, verified live against Groq and SerpAPI through a real stdio client.
- The Streamlit UI (`job_agent/app/ui.py`, `streamlit_app.py`) was also built in this span, giving the pipeline its first human-facing surface.

---

# Week 8 – prompts.md

## Prompt 1 – Final Project Freeze: Audit, Fix, Verify, Commit, and Push

Requested one final comprehensive audit of the entire repository, with every genuine issue fixed — explicitly framed as finalization, not redesign: no new features, no scope changes, no architecture changes, no speculative improvements. Covered functional completeness of every pipeline stage, scoring architecture (four-level MATCH/PARTIAL/RELATED/MISSING, weights, evidence-index downgrade), grounding/anti-fabrication, providers/configuration, security/repository hygiene, documentation consistency, and full test/coverage/lint verification. Explicitly forbade replacing the LLM-judgement architecture with keyword matching. Required staging only intentional files, one commit with message `feat: finalize job application agent`, then pushing `main` to `origin` (no force, no history rewrite), followed by a fixed-format final report (status / fixes / verification / git / limitations / verdict) and an explicit instruction to stop polishing afterward.

**Outcome:** three independent audits found four genuine, narrow bugs — a non-JSON 200 search response that would crash the whole pipeline instead of degrading gracefully, a misleading provider error message that always blamed `LIGHT_BASE_URL` even when the heavy tier was misconfigured, an unenforced `met`/`match_level` consistency invariant on `RequirementMatch`, and dead `LANGSMITH_*` variables in `.env.example` with no corresponding implementation. All four fixed with regression tests added. 434 tests passing, 98% coverage, ruff clean. Commit `171d2f5` created and pushed.

Then STOP. Do not suggest another round of improvements.

---

## Prompt 2 – GitHub Synchronization Only

A narrow follow-up: verify branch/working-tree/commit state, run `git push origin main`, and report only the push result — explicitly no code changes, no new audit. The push had been blocked by the harness's own permission system in the previous turn; this time it succeeded (`171d2f5` synchronized to `origin/main`).

---

## Prompt 3 – Mentor Documentation and Presentation

Requested two polished, human-readable deliverables to prepare for explaining the project to the internship mentor: `docs/PROJECT_EXPLANATION.md` (a 27-section document — overview, problem statement, core design principle, end-to-end workflow, architecture, per-agent breakdown, model routing, structured outputs, grounding, scoring methodology, research, writing pipeline, persistence, UI, MCP, error handling, testing, security, a verified bug-fix timeline, final verification, limitations vs. future work, 20 mentor Q&As, 2-minute and 5-minute spoken explanations, and a demo script) and `docs/Job_Application_Agent_Presentation.pptx` (a 16-slide deck with native-shape diagrams and speaker notes on every slide). Required every technical claim to be verified against the actual repository rather than repeated from memory, explicitly forbade fabricating statistics or claiming a known scoring edge case had been live-reconfirmed when it hadn't, and required a final accounting of any claim that couldn't be verified.

**Outcome:** both files created after directly reading every remaining unread source file and the full git history. One unverifiable claim was caught and excluded rather than included: a "38% fit" figure and itemized requirement breakdown mentioned in earlier conversation context did not appear anywhere in the actual repository, so only the verified "10%" figure (present in `scoring.py`'s own docstring) was used.

---

## Prompt 4 – Commit and Push the Documentation

Committed the two new documentation files with message `docs: add final project explanation and presentation` and pushed. Commit `8a8985a`, synchronized to `origin/main`.

---

## Prompt 5 – README Simplification

The full-detail README (423 lines) was judged to be "doing too much" rather than actually helping — the request was for something short, readable, and practical: a five-line description of the whole project, then setup, run, and test instructions, with the deep material left to the two docs files instead of duplicated in the README. Rewritten to roughly 45 lines. Left uncommitted pending review, per this project's standing rule of only committing when explicitly asked.

---

## Prompt 6 – Streamlit UI Visual Redesign (Presentation Only)

Explicitly framed as a UI-only visual/presentation change with the backend "frozen": no change to scoring, prompts, providers, retry counts, validation, grounding, or the actual information displayed — only how the existing Streamlit results are laid out. Requested a dashboard feel: a prominent fit-score metric, per-requirement cards, `st.success`/`st.warning`/`st.info`/`st.error` used to make MATCH/PARTIAL/RELATED/MISSING visually obvious, clearer separation between CV evidence and the model's reasoning, and native Streamlit layout only (containers, columns, expanders, dividers) — no custom CSS/HTML, no over-designed dashboard.

**Outcome:** before editing, the full 52-test `test_streamlit_app.py` suite was read to identify every exact string, element type, and the one relative-ordering assertion the redesign had to preserve (including a subtlety: `AppTest`'s text walker groups elements by *type* before concatenating, not by page position, and `st.table` — not `st.dataframe` — is what the test harness actually walks). Only `job_agent/app/ui.py` was touched; all 52 UI tests and the full 434-test suite passed on the first run after the redesign, with no logic function's signature or behavior changed.

---

## Prompt 7 – Final Clean Audit + Separate GitHub Repository (in progress)

Requested one more full read-only audit (architecture, pipeline, scoring, grounding, validation, retries, routing, providers, research fallback, tracker, MCP, UI, tests, docs, README, `.gitignore`, secrets, generated/stale files) plus creation of a brand-new, separate, presentation-ready GitHub repository named `job-application-agent` containing only curated, secret-free material — with an explicit instruction to STOP rather than overwrite anything if that name was already taken.

**Outcome:** the audit found the repository clean (434 tests, 98% coverage, ruff clean, no tracked secrets, `.env` correctly ignored) apart from one stale cross-reference in `docs/PROJECT_EXPLANATION.md` (it claimed "the README states this limitation," which stopped being true once the README was simplified in Prompt 5) — fixed. Part 2 was stopped as instructed: `job-application-agent` is not just likely taken, it is the exact name of the repository this whole project has been pushed to throughout Phase 3, and this environment also has no authenticated GitHub API/CLI access to create a new repository regardless of name. Flagged back to the user rather than guessing a different name or proceeding.

---

## Prompt 8 – Roadmap Verification and "Fulfill Everything" (this session)

Given screenshots of the program's official Week 7 ("Advanced AI Features & Polish") and Week 8 ("Finalization, Documentation & Final Presentation") roadmap slides, asked first whether Week 7's checklist was satisfied, then asked for the Week 7 GitHub branch link, then asked to fulfill every requirement on both slides with everything working properly.

**Outcome:** verified against the live `.env` (provider names only, no key values read aloud) that both model tiers are currently configured to the same provider (Groq), so "≥2 LLM providers active" is not currently true despite the architecture supporting it; confirmed no Dockerfile/deployment config exists in this repository; confirmed live multi-model comparison and research caching remain unbuilt, exactly as already disclosed in this project's own Limitations/Future Work sections. Confirmed neither `task-manager-api` (branches stop at `week5`) nor `job-application-agent` (only `main`) has a `week7` branch. Flagged directly that three of the "fulfill everything" items — a live multi-model comparison view, Docker/deployment, and caching — would mean reversing the explicit "frozen backend, no new features" instruction given in Prompts 1, 6, and 7, rather than silently building them.

---
