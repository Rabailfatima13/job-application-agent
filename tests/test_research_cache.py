"""The company research cache (mentor-requested "cache company research").

Every test uses a tmp_path database; nothing here can touch a real one. What
is under test is the contract `ResearchAgent` relies on: a miss is always
safe, a hit returns exactly what was stored, entries never collide across
companies, normalization makes "Google"/" google "/"GOOGLE" the same entry,
and an expired entry behaves exactly like nothing was ever cached.
"""

from datetime import datetime, timedelta

import pytest

from job_agent.memory import CompanyResearchCache
from job_agent.memory.research_cache import normalize_company
from job_agent.models import CompanyBrief, Source


def brief(company: str = "Arbisoft") -> CompanyBrief:
    return CompanyBrief(
        company=company,
        summary="Arbisoft builds data platforms.",
        facts=["Builds data platforms.", "Founded in Lahore."],
        sources=[Source(title="Arbisoft", url="https://arbisoft.com")],
    )


@pytest.fixture
def cache(tmp_path) -> CompanyResearchCache:
    return CompanyResearchCache(tmp_path / "app.db", ttl_hours=24)


# --- normalization -----------------------------------------------------------


def test_normalize_company_strips_and_casefolds():
    assert normalize_company("Google") == "google"
    assert normalize_company(" google ") == "google"
    assert normalize_company("GOOGLE") == "google"


# --- miss / store / retrieve --------------------------------------------------


def test_a_cache_miss_returns_none(cache):
    assert cache.get("Arbisoft") is None


def test_a_stored_brief_can_be_retrieved(cache):
    cache.set("Arbisoft", brief())

    fetched = cache.get("Arbisoft")

    assert fetched is not None
    assert fetched.company == "Arbisoft"
    assert fetched.summary == "Arbisoft builds data platforms."
    assert fetched.facts == ["Builds data platforms.", "Founded in Lahore."]
    assert fetched.sources == [Source(title="Arbisoft", url="https://arbisoft.com")]


def test_facts_and_sources_round_trip_through_json_intact(cache):
    """Requirement: existing cached JSON correctly reconstructs facts and
    sources - not just a truthy value, the exact structured content."""
    original = CompanyBrief(
        company="Nexus",
        summary="Nexus is a fintech company.",
        facts=["Fact one.", "Fact two.", "Fact three."],
        sources=[
            Source(title="Nexus homepage", url="https://nexus.example/"),
            Source(title="Nexus news", url="https://nexus.example/news"),
        ],
    )
    cache.set("Nexus", original)

    fetched = cache.get("Nexus")

    assert fetched.facts == original.facts
    assert fetched.sources == original.sources


def test_setting_again_replaces_the_cached_value(cache):
    cache.set("Arbisoft", brief())
    replacement = CompanyBrief(
        company="Arbisoft", summary="Updated.", facts=["New fact."], sources=[]
    )

    cache.set("Arbisoft", replacement)

    assert cache.get("Arbisoft").summary == "Updated."


# --- no collisions across companies -------------------------------------------


def test_different_companies_do_not_collide(cache):
    cache.set("Sephora", brief("Sephora"))
    cache.set("Nexus", brief("Nexus"))

    assert cache.get("Sephora").company == "Sephora"
    assert cache.get("Nexus").company == "Nexus"


def test_setting_one_company_never_touches_another(cache):
    cache.set("Sephora", brief("Sephora"))
    cache.set("Nexus", brief("Nexus"))

    cache.set("Sephora", CompanyBrief(company="Sephora", summary="Changed.", sources=[]))

    assert cache.get("Sephora").summary == "Changed."
    assert cache.get("Nexus").summary == "Arbisoft builds data platforms."


# --- normalization at the store level -----------------------------------------


def test_google_variants_all_hit_the_same_entry(cache):
    cache.set("Google", brief("Google"))

    assert cache.get("Google") is not None
    assert cache.get(" google ") is not None
    assert cache.get("GOOGLE") is not None


def test_a_lookup_keeps_the_callers_own_spelling(cache):
    """The cache key is normalized; the returned brief's `company` reflects
    whatever spelling *this* caller used, not whatever was originally
    stored - there is nothing for the caller to reconcile."""
    cache.set("Google", brief("Google"))

    fetched = cache.get("GOOGLE")

    assert fetched.company == "GOOGLE"


# --- TTL / expiry --------------------------------------------------------------


def test_a_fresh_entry_within_the_ttl_is_returned(tmp_path):
    cache = CompanyResearchCache(tmp_path / "app.db", ttl_hours=24)
    cache.set("Arbisoft", brief())

    assert cache.get("Arbisoft") is not None


def test_an_expired_entry_behaves_exactly_like_a_miss(tmp_path):
    cache = CompanyResearchCache(tmp_path / "app.db", ttl_hours=1)
    cache.set("Arbisoft", brief())
    # Backdate the row past the 1-hour TTL without waiting for real time to
    # pass - the same "control the clock, not sleep()" approach used
    # elsewhere in this project's tests.
    stale = (datetime.now() - timedelta(hours=2)).isoformat()
    with cache._connect() as connection:
        connection.execute(
            "UPDATE company_research SET cached_at = ? WHERE company = 'arbisoft'",
            (stale,),
        )

    assert cache.get("Arbisoft") is None


def test_a_zero_ttl_treats_every_entry_as_expired(tmp_path):
    cache = CompanyResearchCache(tmp_path / "app.db", ttl_hours=0)
    cache.set("Arbisoft", brief())

    assert cache.get("Arbisoft") is None
