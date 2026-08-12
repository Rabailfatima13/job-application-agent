"""Input handling shared by both parsers: raw text or a .txt/.md/.pdf file."""

import pytest

from job_agent.tools import load_document_text
from job_agent.tools.document_io import looks_like_path


def test_raw_text_is_returned_as_is():
    assert load_document_text("  Junior AI Engineer\nPython  ") == (
        "Junior AI Engineer\nPython"
    )


def test_reads_a_text_file(tmp_path):
    path = tmp_path / "cv.txt"
    path.write_text("Mahnoor Rauf\nPython", encoding="utf-8")

    assert load_document_text(path) == "Mahnoor Rauf\nPython"


def test_multiline_text_is_never_mistaken_for_a_path():
    assert not looks_like_path("Some role\nRequirements: Python .txt")


def test_missing_file_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_document_text(tmp_path / "absent.txt")


def test_unsupported_file_type_is_rejected(tmp_path):
    path = tmp_path / "cv.docx"
    path.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document_text(path)


def test_empty_document_is_rejected():
    with pytest.raises(ValueError):
        load_document_text("   \n  ")
