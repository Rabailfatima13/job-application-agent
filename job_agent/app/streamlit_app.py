"""Streamlit entry point (FR-15).

    streamlit run job_agent/app/streamlit_app.py

Deliberately three lines: the view lives in `ui.py`, which is an ordinary
importable module. Streamlit re-executes *this* file on every interaction and
Streamlit's own AppTest re-executes it as a fresh module, so keeping the
behaviour in an imported module is what makes the surface testable at all.
"""

from job_agent.app import ui

ui.main()
