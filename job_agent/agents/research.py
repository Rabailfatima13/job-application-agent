"""Research agent: job posting -> `CompanyBrief` (FR-2, US-2).

Two clearly separated steps, in line with the proposal's tool split:

    web_search tool  ->  raw SearchResults  ->  summarisation model  ->  brief

The tool does not summarise and the model does not search. The model is also
never given the company name as something to describe from memory - it only
ever sees retrieved snippets, and each fact it returns must cite the result it
came from by index. Facts citing nothing real are dropped.

Degrading is preferred to guessing. A posting with no named employer, a search
that fails, and a search that returns nothing all produce a valid
`CompanyBrief` that honestly says research was unavailable, so the pipeline
keeps going and the user can see why.
"""

import re

from pydantic import BaseModel, ConfigDict, Field

from ..memory.session import RunContext
from ..models import CompanyBrief, JobDescription, Source
from ..observability import traced_tool_call
from ..tools.parsing import UNKNOWN_COMPANY
from ..tools.web_search import DEFAULT_MAX_RESULTS, SearchError, SearchResult
from ..validation import generate_validated
from .base import BaseAgent

RESEARCH_STEP = "summarise"
RESEARCH_MAX_TOKENS = 1024

# Appended to the company name so the search leans towards company information
# rather than vacancies. Generic on purpose - nothing here is specific to any
# one employer.
COMPANY_QUERY_SUFFIX = "company overview"

NO_COMPANY_SUMMARY = (
    "The job description does not name an employer, so no company research "
    "was performed."
)
NO_RESULTS_SUMMARY = "No search results were returned, so no company facts are available."
NO_RELEVANT_RESULTS_SUMMARY = (
    "The search returned nothing that mentions the company, so no company facts "
    "are available."
)
SEARCH_FAILED_SUMMARY = (
    "Company research was unavailable because the web search could not be completed."
)


# --- What the model is asked to return ---------------------------------------


class ResearchFact(BaseModel):
    """One fact, with the search result it came from."""

    model_config = ConfigDict(extra="forbid")

    text: str
    source_index: int


class ResearchExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    facts: list[ResearchFact] = Field(default_factory=list)


RESEARCH_SYSTEM_PROMPT = """\
You summarise web search results about a company for someone applying to a \
specific role there. You are a summariser, not an author.

Rules:
- Use only what the numbered SEARCH RESULTS say. If they do not answer \
something, leave it out. Never add anything you happen to know about the \
company, and never invent a figure, a date, a product or a client.
- Keep it role-relevant: what the company does, its products or services, the \
technologies and domain it works in, and anything recent that an applicant \
could reasonably use. Skip careers-page boilerplate and generic filler.
- summary: 2-4 sentences, factual, no marketing language.
- facts: 3-6 short standalone facts. Each one cites source_index, the number \
of the result it came from.

Reply with a single JSON object and nothing else:
{"summary": str, "facts": [{"text": str, "source_index": int}]}"""


# --- Deterministic grounding guard -------------------------------------------

_DIGIT_RUN = re.compile(r"\d+")


def unsupported_numbers(claim: str, source_text: str) -> list[str]:
    """Numbers in `claim` that do not appear in `source_text`.

    A cheap, high-precision fabrication check: invented company facts are
    overwhelmingly numeric - funding rounds, headcounts, founding years,
    client counts - and a number the retrieved text never mentioned cannot
    have come from it. Formatting is ignored ("$500 million" -> "500").
    """
    source_numbers = set(_DIGIT_RUN.findall(source_text))
    return [n for n in _DIGIT_RUN.findall(claim) if n not in source_numbers]


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


class ResearchAgent(BaseAgent):
    """Produces a short, sourced `CompanyBrief`. Owns research only - it does
    not score the candidate and does not write application material."""

    name = "research"

    def run(self, context: RunContext) -> CompanyBrief:
        if context.job is None:
            raise ValueError(
                "ResearchAgent needs a parsed job description on the RunContext; "
                "run parse_jd first."
            )
        brief = self.research(context.job)
        if not brief.sources:
            context.warnings.append(brief.summary)
        return brief

    def research(self, job: JobDescription, max_results: int = DEFAULT_MAX_RESULTS):
        """Search for the company and summarise what comes back."""
        if job.company == UNKNOWN_COMPANY:
            # Searching for "Unknown" would return noise, and guessing the
            # employer from the role text would be a fabrication.
            return self._unavailable(job.company, NO_COMPANY_SUMMARY)

        try:
            results = self.call_tool(
                "web_search", query=build_query(job), max_results=max_results
            )
        except SearchError as exc:
            return self._unavailable(job.company, f"{SEARCH_FAILED_SUMMARY} ({exc})")

        if not results:
            return self._unavailable(job.company, NO_RESULTS_SUMMARY)

        results = self._keep_relevant(results, job.company)
        if not results:
            return self._unavailable(job.company, NO_RELEVANT_RESULTS_SUMMARY)

        extraction = self._summarise(job, results)
        return self._ground(extraction, results, job.company)

    def _keep_relevant(
        self, results: list[SearchResult], company: str
    ) -> list[SearchResult]:
        """Drop results that never name the company, before they are summarised.

        Cheaper and safer than trying to catch the misattribution afterwards:
        material the summariser never sees cannot become a fact about the wrong
        employer.
        """
        kept = [r for r in results if mentions_company(r, company)]
        dropped = len(results) - len(kept)
        if dropped and self.collector is not None:
            with traced_tool_call(
                self.collector,
                agent=self.name,
                name="research:relevance_guard",
                arguments={"kept": len(kept), "dropped": dropped},
            ) as outcome:
                outcome["result"] = "; ".join(
                    r.title for r in results if r not in kept
                )
        return kept

    @staticmethod
    def _unavailable(company: str, summary: str) -> CompanyBrief:
        """A valid brief that says, honestly, that there is nothing to report."""
        return CompanyBrief(company=company, summary=summary, facts=[], sources=[])

    def _summarise(
        self, job: JobDescription, results: list[SearchResult]
    ) -> ResearchExtraction:
        prompt = build_summarisation_prompt(job, results)

        def produce(feedback: str | None) -> str:
            content = prompt
            if feedback:
                content = (
                    f"{prompt}\n\n---\n"
                    f"Your previous reply was rejected by the schema validator:\n"
                    f"{feedback}\n"
                    "Return corrected JSON only, following the schema exactly."
                )
            result = self.call_model(
                RESEARCH_STEP,
                [{"role": "user", "content": content}],
                system=RESEARCH_SYSTEM_PROMPT,
                max_tokens=RESEARCH_MAX_TOKENS,
            )
            return result.text

        return generate_validated(
            produce,
            ResearchExtraction,
            collector=self.collector,
            agent=self.name,
        )

    def _ground(
        self,
        extraction: ResearchExtraction,
        results: list[SearchResult],
        company: str,
    ) -> CompanyBrief:
        """Keep only what the retrieved results support.

        A fact survives if it cites a real result *and* introduces no number
        that result never mentioned. Summary sentences are filtered the same
        way against everything retrieved. Sources are the results actually
        cited by surviving facts - so a brief's sources are never decorative.
        """
        corpus = "\n".join(f"{r.title} {r.snippet}" for r in results)
        rejected: list[str] = []

        facts: list[str] = []
        cited: list[SearchResult] = []
        for fact in extraction.facts:
            if not 0 <= fact.source_index < len(results):
                rejected.append(f"uncited fact: {fact.text}")
                continue
            result = results[fact.source_index]
            invented = unsupported_numbers(fact.text, f"{result.title} {result.snippet}")
            if invented:
                rejected.append(f"unsupported figure {invented} in: {fact.text}")
                continue
            facts.append(fact.text)
            if result not in cited:
                cited.append(result)

        kept_sentences = []
        for sentence in _split_sentences(extraction.summary):
            invented = unsupported_numbers(sentence, corpus)
            if invented:
                rejected.append(f"unsupported figure {invented} in summary: {sentence}")
            else:
                kept_sentences.append(sentence)
        summary = " ".join(kept_sentences)

        if rejected and self.collector is not None:
            with traced_tool_call(
                self.collector,
                agent=self.name,
                name="research:grounding_guard",
                arguments={"rejected": len(rejected)},
            ) as outcome:
                outcome["result"] = "; ".join(rejected)

        return CompanyBrief(
            # The company is taken from the parsed posting, never from the
            # model, so the brief cannot drift onto a different employer.
            company=company,
            summary=summary or NO_RESULTS_SUMMARY,
            facts=facts,
            sources=[Source(title=r.title, url=r.url) for r in cited],
        )


def build_query(job: JobDescription) -> str:
    """A query about the *company*, not about the job.

    A live run searched "Arbisoft Junior AI Engineer Python FastAPI" and got
    back a stranger's LinkedIn profile, an Instagram post and a vacancy at a
    different company - because a company name plus a role title plus
    technologies is, to a search engine, a job search. The summariser then
    attributed those postings to the target company.

    So the role and the posting's technologies are deliberately *not* in the
    query. Role relevance is applied where it belongs: the summariser is told
    which role is being applied for (see `build_summarisation_prompt`) and asked
    to keep the brief relevant to it. The company name is quoted so it stays the
    primary target rather than one keyword among many.
    """
    company = job.company.strip()
    return f'"{company}" {COMPANY_QUERY_SUFFIX}'.strip()


def mentions_company(result: SearchResult, company: str) -> bool:
    """Whether a result is plausibly about the target company at all.

    A deliberately blunt relevance guard: the company has to be named in the
    title or the snippet. It is what stops a vacancy at some other employer
    becoming a "fact" about this one - the number check cannot catch a
    misattribution, because the figure really is in the cited text.

    Matching allows the first significant word of a multi-word company name, so
    "Arbisoft Ltd" still matches a page that just says "Arbisoft".
    """
    haystack = f"{result.title} {result.snippet}".lower()
    name = company.strip().lower()
    if not name:
        return False
    if name in haystack:
        return True
    head = name.split()[0]
    return len(head) > 2 and head in haystack


def build_summarisation_prompt(job: JobDescription, results: list[SearchResult]) -> str:
    """The numbered results, plus just enough role context to stay relevant."""
    formatted = "\n\n".join(
        f"[{index}] {result.title}\n{result.url}\n{result.snippet}"
        for index, result in enumerate(results)
    )
    return (
        f"COMPANY: {job.company}\n"
        f"ROLE BEING APPLIED FOR: {job.role}\n\n"
        f"SEARCH RESULTS:\n{formatted}"
    )
