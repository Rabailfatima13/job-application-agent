"""Runtime configuration, read once from the environment (.env).

Deliberately plain: environment variables in, a frozen dataclass out. Every
component takes what it needs as an argument rather than reaching for globals,
so tests can build a Settings object directly without touching os.environ.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Which model tier each pipeline step runs on. Parsing/summarising are cheap,
# mechanical steps -> LIGHT; judgement-heavy steps -> HEAVY. This is the
# concrete form of the proposal's cost/latency strategy (NFR-1, NFR-2).
LIGHT = "light"
HEAVY = "heavy"

# Company research (agents/research.py) changes slowly - funding, headcount,
# product focus - so a week is a conservative default: long enough to skip a
# repeat search+summarize for the same employer looked up again soon after,
# short enough that a cached brief does not go stale for long. Configurable
# rather than fixed, since how "slowly" is a judgement call a deployment
# should be free to make differently.
DEFAULT_RESEARCH_CACHE_TTL_HOURS = 168.0


@dataclass(frozen=True)
class ModelConfig:
    """How to reach one model tier."""

    provider: str  # "anthropic" | "openai_compatible"
    model: str
    api_key: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class Settings:
    heavy: ModelConfig
    light: ModelConfig
    search_provider: str
    search_api_key: str | None
    tracker_db_path: Path
    tool_call_log_path: Path
    # Testing/config escape hatch: skip the writing/tailoring stage entirely
    # (no tailored CV, no cover letter) so the rest of the pipeline - parsing,
    # research, scoring, tracking - can be exercised without the extra model
    # calls and grounding-retry loop the writer costs. Off by default: a
    # normal run always tailors, exactly as before this field existed.
    skip_tailoring: bool = False
    # How long a cached company research result stays valid (see
    # memory/research_cache.py). Not a bool escape hatch like skip_tailoring -
    # a real, tunable value, so it gets its own default constant rather than
    # a bare number repeated in two places.
    research_cache_ttl_hours: float = DEFAULT_RESEARCH_CACHE_TTL_HOURS


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Build Settings from the environment, loading .env first.

    Missing API keys are not an error here - a key is only required when the
    component that needs it is actually used, so tests and offline runs work
    without any credentials configured.
    """
    load_dotenv(env_file or PROJECT_ROOT / ".env")

    def path_from(var: str, default: str) -> Path:
        raw = Path(os.getenv(var) or default)
        return raw if raw.is_absolute() else PROJECT_ROOT / raw

    def float_from_env(var: str, default: float) -> float:
        """A numeric setting from the environment - the default for anything
        unset, blank, or unparseable, exactly the same "never fail the whole
        app over one optional setting" spirit `skip_tailoring` already
        applies to its own env var below."""
        raw = os.getenv(var)
        if raw is None or not raw.strip():
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    return Settings(
        heavy=ModelConfig(
            provider=os.getenv("HEAVY_PROVIDER", "anthropic"),
            model=os.getenv("HEAVY_MODEL", "claude-sonnet-5"),
            # HEAVY_API_KEY lets the heavy tier move to any OpenAI-compatible
            # provider (e.g. Gemini); ANTHROPIC_API_KEY is the fallback so an
            # existing anthropic-provider .env keeps working unchanged.
            api_key=os.getenv("HEAVY_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
            base_url=os.getenv("HEAVY_BASE_URL"),
        ),
        light=ModelConfig(
            provider=os.getenv("LIGHT_PROVIDER", "openai_compatible"),
            model=os.getenv("LIGHT_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("LIGHT_API_KEY"),
            base_url=os.getenv("LIGHT_BASE_URL"),
        ),
        search_provider=os.getenv("SEARCH_PROVIDER", "serpapi"),
        search_api_key=os.getenv("SEARCH_API_KEY"),
        tracker_db_path=path_from("TRACKER_DB_PATH", "data/applications.db"),
        tool_call_log_path=path_from("TOOL_CALL_LOG_PATH", "data/tool_calls.log"),
        skip_tailoring=(os.getenv("SKIP_TAILORING") or "").strip().lower()
        in ("1", "true", "yes"),
        research_cache_ttl_hours=float_from_env(
            "RESEARCH_CACHE_TTL_HOURS", DEFAULT_RESEARCH_CACHE_TTL_HOURS
        ),
    )
