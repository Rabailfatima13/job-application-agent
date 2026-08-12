"""Scaffold demo surface: confirms the project is installed and configured.

Grows in Week 6 into the real run view (CV + JD in, fit report and tracker out)
and is polished in Week 7. It stays a view over the pipeline's structured
outputs - no scoring or writing logic ever moves in here.
"""

import streamlit as st

from job_agent import __version__
from job_agent.config import load_settings

st.set_page_config(page_title="Job Application Agent", page_icon="[]")
st.title("Job Application Agent")
st.caption(f"scaffold v{__version__} - pipeline lands in Week 6")

settings = load_settings()

st.subheader("Configuration")
st.write(
    {
        "heavy model": f"{settings.heavy.provider} / {settings.heavy.model}",
        "light model": f"{settings.light.provider} / {settings.light.model}",
        "search provider": settings.search_provider,
        "tracker db": str(settings.tracker_db_path),
    }
)

missing = [
    label
    for label, value in (
        ("ANTHROPIC_API_KEY", settings.heavy.api_key),
        ("LIGHT_API_KEY", settings.light.api_key),
        ("SEARCH_API_KEY", settings.search_api_key),
    )
    if not value
]
if missing:
    st.warning("Not configured yet: " + ", ".join(missing) + " (see .env.example)")
else:
    st.success("All credentials configured.")
