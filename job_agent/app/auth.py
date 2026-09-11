"""Login/signup UI - the first stage of the eventual

    Login/Signup -> CV + JD entry -> pipeline progress -> results dashboard

flow. A view, same philosophy as ui.py: it collects a name/email/password
and hands them to the existing `UserStore`; no hashing or validation logic
lives here. Only Login/Signup is implemented in this pass - the other pages
are the existing CV/JD form and results view in ui.py, unchanged.
"""

import streamlit as st

from ..config import Settings
from ..memory.users import EmailAlreadyRegistered, UserStore
from ..models.user import User

#: Session state key holding the signed-in User for the rest of this
#: Streamlit session - the same "held across reruns" pattern ui.py already
#: uses for RESULT_KEY/ERROR_KEY.
CURRENT_USER_KEY = "current_user"


def _store(settings: Settings) -> UserStore:
    return UserStore(settings.tracker_db_path)


def _render_login(settings: Settings) -> None:
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button(
        "Sign in →", type="primary", key="login_submit", use_container_width=True
    ):
        user = _store(settings).authenticate(email, password)
        if user is None:
            st.error("Incorrect email or password.")
        else:
            st.session_state[CURRENT_USER_KEY] = user
            st.rerun()


def _render_signup(settings: Settings) -> None:
    name = st.text_input("Name", key="signup_name")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_password")
    confirm = st.text_input(
        "Confirm password", type="password", key="signup_confirm"
    )
    if st.button(
        "Create account →",
        type="primary",
        key="signup_submit",
        use_container_width=True,
    ):
        if password != confirm:
            st.error("Passwords do not match.")
            return
        try:
            user = _store(settings).create(name, email, password)
        except (EmailAlreadyRegistered, ValueError) as exc:
            st.error(str(exc))
            return
        st.session_state[CURRENT_USER_KEY] = user
        st.rerun()


def render_gate(settings: Settings) -> User | None:
    """Render the login/signup screen and return the signed-in user.

    Returns None when nobody is signed in yet - the caller (`ui.main`) must
    render nothing else in that case, so the rest of the app is only ever
    reachable after a successful login or signup.
    """
    user = st.session_state.get(CURRENT_USER_KEY)
    if user is not None:
        return user

    st.markdown(
        """
        <div class="jaa-brand">
            <div class="jaa-brand-mark">✦</div>
            <div class="jaa-brand-title">Job Application Agent</div>
            <div class="jaa-brand-tagline">
                AI-powered applications, tailored to your strengths.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="login_card"):
        st.caption("Sign in to continue.")
        tab_login, tab_signup = st.tabs(["Log in", "Sign up"])
        with tab_login:
            _render_login(settings)
        with tab_signup:
            _render_signup(settings)
    return None


def render_account_sidebar(user: User) -> None:
    """Who is signed in, and the way out - kept separate from ui.py's own
    configuration sidebar rather than merged into it."""
    with st.sidebar:
        st.caption(f"Signed in as **{user.name}**")
        if st.button("Log out"):
            st.session_state.pop(CURRENT_USER_KEY, None)
            st.rerun()
        st.divider()
