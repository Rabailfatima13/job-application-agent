"""Memory, in the two forms the proposal calls for (S11):

  tracker.py  - persistent store of the application pipeline, across runs
  session.py  - the CV and role context held during a single run

(The optional vector memory for reusing past strong bullets is a stretch goal
and is not part of the required build.)
"""

from .baseline_cv import BaselineCVStore
from .research_cache import CompanyResearchCache
from .session import RunContext
from .tailored_cv_version import TailoredCVVersionStore
from .tracker import (
    ApplicationTracker,
    InMemoryApplicationTracker,
    SQLiteApplicationTracker,
)
from .users import EmailAlreadyRegistered, UserStore

__all__ = [
    "ApplicationTracker",
    "BaselineCVStore",
    "CompanyResearchCache",
    "EmailAlreadyRegistered",
    "InMemoryApplicationTracker",
    "RunContext",
    "SQLiteApplicationTracker",
    "TailoredCVVersionStore",
    "UserStore",
]
