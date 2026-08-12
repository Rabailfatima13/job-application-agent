# Job Application Agent

A multi-agent system that takes a CV and a job description, researches the target
company, scores candidate–role fit with evidence, tailors the CV emphasis and drafts
a cover letter **using only claims already present in the CV**, and tracks each
application in a persistent store.

Arbisoft AI Internship 2026 — Phase 3 (Weeks 5.5–8). Built to the approved project
specification, which is the source of truth for scope and requirements.

> **Status: scaffold (Week 5.5).** Interfaces, structured outputs, validation, tracing
> and tests are in place. The agents themselves land in Week 6.

## The pipeline

```
CV + Job Description
        |
        v
   Supervisor  ── parses inputs, routes work, assembles the result, tracks the application
        |
   +----+----------------+-----------------+
   |                     |                 |
   v                     v                 v
Research              Scoring           Writing          (Week 7)
(web search)      (requirements vs.   (re-emphasis +
   |               CV evidence)        cover letter)
   v                     v                 v
Company Brief       Fit Report      Tailored CV + Cover Letter
   \                     |                 /
    +--------------------+----------------+
                         |
                    Validation  ── schema + retry, then no-fabrication grounding
                         |
                         v
                Application Tracker
```

## Core rule: no fabricated experience

The writing agent may **re-order, re-emphasise, and re-word** the candidate's real
content. It may **never** invent a job, project, skill, achievement, technology, or
number of years. Every generated claim must trace back to text in the source CV; the
grounding check strips anything that does not, and if tailoring cannot be validated the
system returns the original CV plus the gap list rather than a bad rewrite.

## Layout

| Path | Responsibility |
| --- | --- |
| `job_agent/models/` | Pydantic structured outputs: parsed CV/JD, company brief, fit report, tracker record |
| `job_agent/llm/` | Provider-agnostic model clients + tier routing (cheap model for parsing, strong model for judgement) |
| `job_agent/agents/` | `BaseAgent`, then supervisor + research/scoring/writing workers |
| `job_agent/tools/` | Narrow, independently testable tools and the registry agents are given |
| `job_agent/memory/` | Persistent application tracker + per-run session context |
| `job_agent/validation/` | Schema guard with bounded retry; no-fabrication grounding check |
| `job_agent/observability/` | Trace events per tool/model call with latency and token counts |
| `job_agent/mcp_server/` | Custom MCP server exposing the tracker resource and tools (Week 7) |
| `job_agent/app/` | Streamlit demo surface (thin — no business logic) |
| `tests/` | Unit tests per component + fixtures for the end-to-end run |

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate
```

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Fill in `.env`: an Anthropic key for the heavy tier, any OpenAI-compatible endpoint
(Groq / Ollama / OpenRouter) for the light tier, and a search key for the research
agent. No key is needed to run the tests.

## Running

```bash
pytest
```

```bash
streamlit run job_agent/app/streamlit_app.py
```

## Roadmap

- **Week 6 — core pipeline.** CV/JD parsing, research agent, scoring agent, fit report,
  SQLite tracker, supervisor orchestration, happy-path end to end, tests.
- **Week 7 — advanced features.** Writing agent, no-fabrication validation, retry and
  fallback, second provider routed and compared, MCP server, UI polish.
- **Week 8 — finalisation.** Feature freeze, full test suite, two-role E2E, tracing and
  cost/latency numbers, documentation, demo.
