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
