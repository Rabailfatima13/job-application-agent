# Job Application Agent

A multi-agent system that takes a CV and a job description, researches the target
company, scores candidate–role fit with evidence, tailors the CV emphasis and drafts
a cover letter **using only claims already present in the CV**, and tracks each
application in a persistent store.

Arbisoft AI Internship 2026 — Phase 3 (Weeks 5.5–8). Built to the approved project
specification, which is the source of truth for scope and requirements.

> **Status: Week 6 complete · Week 7 in progress.** The Week 6 pipeline runs end to end
> — parse, research, score, track, persist. Week 7 has added the no-fabrication
> grounding system and the writing agent (tailored CV + cover letter). The MCP server,
> multi-model comparison and the user-facing UI are **not** implemented yet.

## The Week 6 pipeline

```
CV + Job Description
        |
        v
     parse_cv          ─ verbatim evidence extracted from the CV
        |
        v
     parse_jd          ─ requirements vs. responsibilities, must-have vs. nice-to-have
        |
        v
     research          ─ web search -> sourced company brief (best effort)
        |
        v
    score_fit          ─ evidence-backed FitReport, score computed in code
        |
        v
 write_application     ─ Week 7: tailored CV + cover letter, schema-validated
        |                AND grounded in the real CV, or not returned at all
        v
 track_application     ─ ApplicationRecord written to SQLite, always as `draft`
        |
        v
       END             ─ full RunContext returned: inputs, brief, report, record,
                         warnings, trace
```

Orchestrated by a LangGraph `StateGraph` in
[`job_agent/agents/supervisor.py`](job_agent/agents/supervisor.py). Full detail in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Core rule: no fabricated experience

The writing agent may **re-order, re-emphasise, and re-word** the candidate's real
content. It may **never** invent a job, project, skill, achievement, technology, or
number of years.

This is enforced at every stage, deterministically and without an LLM:

- **CV parsing** drops any evidence line the model did not copy from the source CV.
- **Scoring** lets the model return only an *index* into that evidence — it cannot write
  evidence text — and a match claiming evidence that does not exist is downgraded to
  unmet.
- **Research** searches for the *company* (not the job), discards results that never
  mention it, and drops any fact citing no real result or carrying a number absent from
  the result it cites.
- **Writing** validates every generated bullet and every cover-letter sentence. A line
  may be reworded, reordered or combined from two real bullets — but a technology,
  employer, qualification, metric or duration the CV never stated is rejected, the model
  is told exactly what was unsupported, and it rewrites. If it cannot, no documents are
  returned at all and the original CV plus the gap list stand.

### Two evidence domains

The cover letter may cite the researched company; the candidate's claims may not.

| Domain | Corpus | Supports |
| --- | --- | --- |
| **Candidate** | CV raw text, parsed evidence, listed skills, candidate name | every claim about the candidate's experience, skills, education, employers, certifications and metrics |
| **Company** | the verified company brief — summary, facts, source titles | statements about the company only |

They are never merged. So `"Arbisoft uses Django"` lets the letter say *"your company uses
Django"* or *"your Django work interests me"*, and never *"I have Django experience"* —
unless Django is independently in the CV. The tailored CV is checked against the
candidate domain alone. A claim whose subject is not clearly the employer defaults to the
strict candidate domain, and the advertised role title grants no evidence either: a role
called "Kubernetes Engineer" does not make Kubernetes claimable.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (D8–D10) for why grounding is a
separate layer from schema validation.

## Layout

| Path | Responsibility |
| --- | --- |
| `job_agent/models/` | Pydantic structured outputs: parsed CV/JD, company brief, fit report, tracker record |
| `job_agent/llm/` | Provider-agnostic model clients + tier routing (cheap model for parsing, strong model for judgement) |
| `job_agent/agents/` | `BaseAgent`, LangGraph supervisor, research + scoring workers (writing: Week 7) |
| `job_agent/tools/` | Narrow, independently testable tools and the registry agents are given |
| `job_agent/memory/` | Persistent application tracker + per-run session context |
| `job_agent/validation/` | Schema guard with bounded retry (no-fabrication grounding: Week 7) |
| `job_agent/observability/` | Trace events per tool/model call with latency and token counts |
| `job_agent/mcp_server/` | Custom MCP server exposing the tracker resource and tools (Week 7) |
| `job_agent/app/` | Streamlit placeholder — shows configuration only, does not run the pipeline yet (Week 7) |
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

## Configuration

`.env.example` is a template holding placeholders only — copy it to `.env` and fill in
your own values there.

> **`.env` is gitignored and must stay that way.** Never commit real API keys, and never
> paste them into source files, the README, or `.env.example`. Everything under `data/`
> is ignored too, so a CV or job posting kept there is not committed either.

| Variable | Needed for a real run | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | yes | Heavy tier — fit scoring |
| `HEAVY_MODEL` | optional | Defaults to `claude-sonnet-5` |
| `LIGHT_BASE_URL`, `LIGHT_API_KEY`, `LIGHT_MODEL` | yes | Cheap tier — parsing and summarising, via any OpenAI-compatible endpoint (Groq / Ollama / OpenRouter) |
| `SEARCH_PROVIDER`, `SEARCH_API_KEY` | recommended | `serpapi` or `brave`. Without a key the run still completes; research degrades to a warning |
| `TRACKER_DB_PATH`, `TOOL_CALL_LOG_PATH` | optional | Default to `data/applications.db` and `data/tool_calls.log` |

**No key of any kind is needed to run the tests** — every test uses stubs.

## Running the Week 6 pipeline

There is no CLI or UI yet (Week 7). The pipeline is driven directly:

```python
from job_agent.agents import build_supervisor
from job_agent.config import load_settings

context = build_supervisor(load_settings()).run("path/to/cv.pdf", "path/to/jd.txt")

print(context.job.role, context.job.company, context.job.location)
print(context.fit_report.overall_fit, context.fit_report.gaps)
print(context.application.application_id, context.application.status)
print(context.warnings, context.trace.totals())
```

Both arguments accept raw text or a path to a `.pdf`, `.txt` or `.md` file.

## Where data is stored

| What | Where | Committed? |
| --- | --- | --- |
| Tracked applications (SQLite) | `data/applications.db` | no |
| Tool/model call trace log (JSON lines) | `data/tool_calls.log` | no |
| In-memory trace for one run | `context.trace.events` / `.totals()` | n/a |

## Tests

```bash
python -m pytest -q
```

```bash
python -m ruff check .
```

```bash
python -m pytest --cov=job_agent --cov-report=term-missing
```

## Status

**Completed in Week 6**

- CV and job-description parsing from text, `.txt`/`.md`, and PDF
- Structured Pydantic outputs at every step, with bounded schema retry
- Verbatim CV evidence protection
- Requirements separated from responsibilities; must-have vs. nice-to-have preserved
- Location extracted from the posting
- Research agent with SerpAPI/Brave search, preserved sources, number grounding
- Evidence-grounded scoring with a deterministic, explainable fit score
- Persistent SQLite application tracker surviving restarts
- LangGraph supervisor running the pipeline end to end
- Tracing of every tool and model call with latency and token counts
- 156 tests, ruff clean, ~91% coverage

**In progress — Week 7**

- ✅ No-fabrication grounding system (`validation/grounding.py`), with two strictly
  separated evidence domains
- ✅ Writing agent — tailored CV and cover letter, grounded or not returned
- ✅ Supervisor integration: optional `write_application` stage
- ✅ Company-oriented research query plus a relevance guard on search results
- ✅ Verified end to end against live models and live web search
- ⬜ Custom MCP server
- ⬜ Multi-model routing comparison
- ⬜ Additional LangGraph branches (fit threshold, next-role loop)
- ⬜ Streamlit UI and a user-facing runner
- ⬜ Research caching and cost/latency tuning

**Week 8** — feature freeze, two-role end-to-end run, cost/latency numbers, final
documentation and demo.
