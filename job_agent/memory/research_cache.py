"""Persistent cache for public company research (mentor-requested "cache
company research"): the `CompanyBrief` a company's own name already fully
determines - the same web search plus LLM summarization, repeated for
nothing, every time a second application names an employer already looked
up (see `agents/research.py`'s `build_query`, deliberately built from the
company name alone, never the role or the candidate).

Reuses the same SQLite file every other store in this project writes to
(`Settings.tracker_db_path`) - one more table, not a second database, not a
second cache service. Plain sqlite3, connection-per-operation, the same
pattern `BaselineCVStore`/`TailoredCVVersionStore` already use.

Deliberately holds nothing but what `CompanyBrief` itself already holds:
company, summary, facts, sources. No CV text, no candidate evidence, no
user id, no job description, no application data - there is no column here
any of that could land in, and the table is keyed by company name only, so
it is structurally unable to ever distinguish (let alone leak between)
users. Kept entirely separate from `baseline_cvs`, `tailored_cv_versions`
and `applications` - a different table, with no foreign key or shared key
to any of them.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from ..config import DEFAULT_RESEARCH_CACHE_TTL_HOURS
from ..models import CompanyBrief, Source

SCHEMA = """
CREATE TABLE IF NOT EXISTS company_research (
    company    TEXT PRIMARY KEY,
    summary    TEXT NOT NULL,
    facts      TEXT NOT NULL,
    sources    TEXT NOT NULL,
    cached_at  TEXT NOT NULL
)
"""


def normalize_company(company: str) -> str:
    """The cache key: "Google", " google " and "GOOGLE" must all resolve to
    the same entry, so the raw name is never used as a key directly."""
    return company.strip().casefold()


class CompanyResearchCache:
    """One row per normalized company name - a cache miss (nothing stored,
    or a stale row) is always safe: the caller just performs research again,
    exactly as if this class did not exist."""

    def __init__(
        self,
        db_path: str | Path,
        ttl_hours: float = DEFAULT_RESEARCH_CACHE_TTL_HOURS,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        with self._connect() as connection:
            connection.execute(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:  # commits on success, rolls back on exception
                yield connection
        finally:
            connection.close()

    def get(self, company: str) -> CompanyBrief | None:
        """The cached brief for `company`, or None on a miss or an expired
        entry - the two are indistinguishable to the caller on purpose, so
        an expired row behaves exactly like nothing was ever cached."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary, facts, sources, cached_at FROM company_research "
                "WHERE company = ?",
                (normalize_company(company),),
            ).fetchone()
        if row is None:
            return None
        cached_at = datetime.fromisoformat(row["cached_at"])
        if datetime.now() - cached_at >= self.ttl:
            return None
        return CompanyBrief(
            # The caller's own spelling/casing for *this* run, not whatever
            # was stored - the cache key is normalized, the display text
            # never needs to be.
            company=company,
            summary=row["summary"],
            facts=json.loads(row["facts"]),
            sources=[Source(**s) for s in json.loads(row["sources"])],
        )

    def set(self, company: str, brief: CompanyBrief) -> None:
        """Cache `brief` under `company`'s normalized name, replacing
        whatever was cached for it before. Callers decide *whether* a brief
        is worth caching (see `ResearchAgent.research`, which only ever
        calls this for a brief that actually carries sources) - this method
        stores whatever it is given."""
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO company_research "
                "(company, summary, facts, sources, cached_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(company) DO UPDATE SET "
                "summary = excluded.summary, facts = excluded.facts, "
                "sources = excluded.sources, cached_at = excluded.cached_at",
                (
                    normalize_company(company),
                    brief.summary,
                    json.dumps(brief.facts),
                    json.dumps([s.model_dump() for s in brief.sources]),
                    datetime.now().isoformat(),
                ),
            )
