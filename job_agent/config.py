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
    )
