"""My Profile / Baseline CV UI (mentor-requested baseline CV system):

    Login/Signup -> My Profile (baseline CV) -> Entry (job description)
    -> Loading -> Results

A view, same philosophy as auth.py and ui.py: it only collects the raw CV
text and hands it to the existing `BaselineCVStore`. Parsing still happens
exactly where it always has - inside the pipeline's own `parse_cv` step, the
moment a run actually needs it - never here.
"""

import tempfile
from pathlib import Path

import streamlit as st

from ..config import Settings
from ..memory.baseline_cv import BaselineCVStore
from ..models.baseline_cv import BaselineCV
from ..models.user import User
from ..tools.document_io import load_document_text


def _store(settings: Settings) -> BaselineCVStore:
    return BaselineCVStore(settings.tracker_db_path)


def _resolve_cv_text(uploaded, pasted: str) -> str:
    """The actual CV text to store - never a transient upload path.

    Reuses the pipeline's own document loader for PDF/txt/md extraction
    (`load_document_text`) rather than a second implementation - an upload
    wins over the pasted box, exactly like the existing entry-page inputs.
    """
    if uploaded is None:
        return pasted.strip()
    with tempfile.TemporaryDirectory(prefix="baseline-cv-upload-") as directory:
        if uploaded.name.lower().endswith(".pdf"):
            path = Path(directory) / uploaded.name
            path.write_bytes(uploaded.getvalue())
            return load_document_text(path)
        return uploaded.getvalue().decode("utf-8", errors="replace")


def _render_edit_form(store: BaselineCVStore, user: User, baseline: BaselineCV) -> None:
    """Edit the baseline CV's own text directly - no upload, no temp file,
    just the same `store.save` the initial-save and replace flows already
    use, so an edit and a replace are the same guarantee (one row per user,
    `created_at` preserved, never touching `tailored_cv_versions`) reached
    two different ways.

    The widget key is tied to `updated_at` on purpose: a Streamlit
    `text_area` keeps whatever the user has typed across reruns once its key
    exists, regardless of a fresh `value=` - so without this, the box would
    keep showing stale text after the baseline changed some other way (a
    save from "Replace baseline CV" below). Tying the key to `updated_at`
    makes it a new widget - reseeded from the current `value=` - every time
    the underlying baseline actually changes, from either box.
    """
    key = f"baseline_edit_text_{baseline.updated_at.isoformat()}"
    edited = st.text_area(
        "Edit your baseline CV", value=baseline.cv_text, height=350, key=key
    )
    if st.button(
        "💾 Save Changes",
        type="primary",
        key="baseline_save_edit",
        width="stretch",
    ):
        try:
            store.save(user.id, edited)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.success("Baseline CV updated.")
        st.rerun()


def _render_save_form(store: BaselineCVStore, user: User, *, replacing: bool) -> None:
    suffix = "replace" if replacing else "initial"
    file = st.file_uploader(
        "Upload (.txt, .md, .pdf)",
        type=["txt", "md", "pdf"],
        key=f"baseline_file_{suffix}",
    )
    pasted = st.text_area("...or paste it", height=220, key=f"baseline_text_{suffix}")
    label = "Save as new baseline CV" if replacing else "Save baseline CV"

    if st.button(
        label, type="primary", key=f"baseline_save_{suffix}", width="stretch"
    ):
        try:
            cv_text = _resolve_cv_text(file, pasted)
            store.save(user.id, cv_text)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.success("Baseline CV saved.")
        st.rerun()


def render_baseline_cv_section(settings: Settings, user: User) -> BaselineCV | None:
    """'My Profile': the baseline CV every application is tailored from.

    Returns the current baseline (or None), so the entry page below can use
    it without querying storage a second time in the same run.
    """
    store = _store(settings)
    baseline = store.get(user.id)

    with st.container(key="profile_card", border=True):
        st.caption("BASELINE CV")
        st.subheader("My profile", anchor="my-profile")

        if baseline is None:
            st.caption(
                "Upload your CV once - it becomes your baseline, and every job "
                "application is tailored from it. You will not need to upload "
                "it again."
            )
            _render_save_form(store, user, replacing=False)
        else:
            st.badge("CV ready", icon=":material/check_circle:", color="green")
            st.caption(
                "Baseline CV on file — saved "
                f"{baseline.updated_at.strftime('%Y-%m-%d %H:%M')}. Every "
                "tailored application is built from this CV. Tailoring for "
                "a specific job never changes it - only editing or "
                "replacing it here does."
            )
            with st.expander("✏️ Edit CV", expanded=True):
                _render_edit_form(store, user, baseline)
            with st.expander("Replace baseline CV"):
                _render_save_form(store, user, replacing=True)

    return baseline
