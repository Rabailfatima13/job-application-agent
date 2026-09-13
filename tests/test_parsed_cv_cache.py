"""The parsed-CV cache (the CV-side counterpart to `CompanyResearchCache`).

Every test uses a tmp_path database; nothing here can touch a real one. What
is under test is the contract `parse_cv` relies on: a miss is always safe, a
hit returns exactly what was stored, entries never collide across different
CV texts, and editing so much as one character produces a fresh cache key.
"""

from job_agent.memory import ParsedCVCache
from job_agent.models import CVEvidence, ParsedCV

CV_TEXT = "Built a Task Management REST API with FastAPI and SQLite."


def parsed(text: str = CV_TEXT) -> ParsedCV:
    return ParsedCV(
        candidate_name="Mahnoor Rauf",
        raw_text=text,
        evidence=[CVEvidence(text=text, section="Experience", skills=["FastAPI"])],
        skills=["Python", "FastAPI"],
    )


def test_a_miss_returns_none(tmp_path):
    cache = ParsedCVCache(tmp_path / "app.db")

    assert cache.get(CV_TEXT) is None


def test_a_hit_returns_exactly_what_was_stored(tmp_path):
    cache = ParsedCVCache(tmp_path / "app.db")
    original = parsed()

    cache.set(CV_TEXT, original)
    hit = cache.get(CV_TEXT)

    assert hit is not None
    assert hit.candidate_name == original.candidate_name
    assert hit.skills == original.skills
    assert [e.text for e in hit.evidence] == [e.text for e in original.evidence]
    assert [e.section for e in hit.evidence] == [e.section for e in original.evidence]
    assert [e.skills for e in hit.evidence] == [e.skills for e in original.evidence]
    assert hit.raw_text == CV_TEXT  # the caller's own text, not a stored copy


def test_a_single_character_edit_is_a_guaranteed_miss(tmp_path):
    # The whole point of a content-addressed cache: no separate invalidation
    # logic is needed because any real edit changes the key.
    cache = ParsedCVCache(tmp_path / "app.db")
    cache.set(CV_TEXT, parsed())

    assert cache.get(CV_TEXT + ".") is None


def test_two_different_cvs_never_collide(tmp_path):
    cache = ParsedCVCache(tmp_path / "app.db")
    other_text = "Taught mathematics for three years."
    cache.set(CV_TEXT, parsed(CV_TEXT))
    cache.set(other_text, parsed(other_text))

    assert cache.get(CV_TEXT).raw_text == CV_TEXT
    assert cache.get(other_text).raw_text == other_text


def test_setting_the_same_text_again_overwrites_cleanly(tmp_path):
    cache = ParsedCVCache(tmp_path / "app.db")
    cache.set(CV_TEXT, parsed())
    updated = ParsedCV(
        candidate_name="A Different Name",
        raw_text=CV_TEXT,
        evidence=[],
        skills=[],
    )

    cache.set(CV_TEXT, updated)

    assert cache.get(CV_TEXT).candidate_name == "A Different Name"
