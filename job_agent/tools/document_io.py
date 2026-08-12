"""Getting text out of whatever the user supplied.

One entry point for both parsers, so there is a single answer to "what counts
as a CV?" - raw text, a .txt/.md file, or a PDF. Nothing here interprets the
content; it only produces the source string the parsers treat as authoritative.
"""

from pathlib import Path

TEXT_SUFFIXES = {".txt", ".md"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES

# A path never realistically exceeds this; anything longer is pasted document
# text, and calling Path() on it would be pointless (and raises on Windows).
_MAX_PATH_LENGTH = 260


def looks_like_path(source: str | Path) -> bool:
    if isinstance(source, Path):
        return True
    if "\n" in source or len(source) > _MAX_PATH_LENGTH:
        return False
    return Path(source).suffix.lower() in SUPPORTED_SUFFIXES


def read_pdf_text(path: Path) -> str:
    from pypdf import PdfReader  # imported lazily; only PDFs pay for it

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_document_text(source: str | Path) -> str:
    """Return the document's text.

    `source` is either the text itself or a path to a .txt/.md/.pdf file.
    Raises FileNotFoundError for a missing file and ValueError for an empty
    document - the parsers need a real source string, and failing here gives a
    clearer message than an empty parse.

    Only a `Path` with an unsupported suffix raises ValueError; a plain string
    that happens to end in `.docx` is treated as document text, because a bare
    string is ambiguous and guessing "path" would fail confusingly.
    """
    if looks_like_path(source):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"No such document: {path}")
        suffix = path.suffix.lower()
        if suffix in PDF_SUFFIXES:
            text = read_pdf_text(path)
        elif suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
        else:
            raise ValueError(
                f"Unsupported document type '{suffix}' "
                f"(expected one of {sorted(SUPPORTED_SUFFIXES)})."
            )
    else:
        text = str(source)

    text = text.strip()
    if not text:
        raise ValueError("Document is empty.")
    return text
