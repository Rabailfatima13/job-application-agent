"""Configuration reads from the environment and never demands credentials just
to import the package."""

from job_agent.config import (
    DEFAULT_RESEARCH_CACHE_TTL_HOURS,
    HEAVY,
    LIGHT,
    load_settings,
)


def test_defaults_apply_when_nothing_is_configured(monkeypatch, tmp_path):
    for var in (
        "HEAVY_PROVIDER",
        "HEAVY_MODEL",
        "ANTHROPIC_API_KEY",
        "LIGHT_PROVIDER",
        "LIGHT_MODEL",
        "LIGHT_API_KEY",
        "SEARCH_PROVIDER",
        "TRACKER_DB_PATH",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = load_settings(env_file=tmp_path / "absent.env")

    assert settings.heavy.provider == "anthropic"
    assert settings.light.provider == "openai_compatible"
    assert settings.heavy.api_key is None  # missing keys are not an import-time error
    assert settings.tracker_db_path.name == "applications.db"
    assert settings.tracker_db_path.is_absolute()


def test_environment_overrides_are_picked_up(monkeypatch, tmp_path):
    monkeypatch.setenv("HEAVY_MODEL", "claude-opus-5")
    monkeypatch.setenv("LIGHT_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LIGHT_MODEL", "llama3")

    settings = load_settings(env_file=tmp_path / "absent.env")

    assert settings.heavy.model == "claude-opus-5"
    assert settings.light.base_url == "http://localhost:11434/v1"


def test_tier_constants_are_the_two_supported_tiers():
    assert {HEAVY, LIGHT} == {"heavy", "light"}


def test_skip_tailoring_defaults_to_false(monkeypatch, tmp_path):
    monkeypatch.delenv("SKIP_TAILORING", raising=False)

    settings = load_settings(env_file=tmp_path / "absent.env")

    assert settings.skip_tailoring is False


def test_skip_tailoring_accepts_common_truthy_spellings(monkeypatch, tmp_path):
    for value in ("1", "true", "True", "YES", " yes "):
        monkeypatch.setenv("SKIP_TAILORING", value)

        assert load_settings(env_file=tmp_path / "absent.env").skip_tailoring is True


def test_skip_tailoring_stays_false_for_anything_else(monkeypatch, tmp_path):
    for value in ("0", "false", "no", ""):
        monkeypatch.setenv("SKIP_TAILORING", value)

        assert load_settings(env_file=tmp_path / "absent.env").skip_tailoring is False


def test_research_cache_ttl_defaults_to_a_conservative_week(monkeypatch, tmp_path):
    monkeypatch.delenv("RESEARCH_CACHE_TTL_HOURS", raising=False)

    settings = load_settings(env_file=tmp_path / "absent.env")

    assert settings.research_cache_ttl_hours == DEFAULT_RESEARCH_CACHE_TTL_HOURS
    assert settings.research_cache_ttl_hours == 168.0


def test_research_cache_ttl_is_picked_up_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("RESEARCH_CACHE_TTL_HOURS", "24")

    settings = load_settings(env_file=tmp_path / "absent.env")

    assert settings.research_cache_ttl_hours == 24.0


def test_research_cache_ttl_accepts_fractional_hours(monkeypatch, tmp_path):
    monkeypatch.setenv("RESEARCH_CACHE_TTL_HOURS", "0.5")

    settings = load_settings(env_file=tmp_path / "absent.env")

    assert settings.research_cache_ttl_hours == 0.5


def test_research_cache_ttl_falls_back_to_the_default_when_invalid(
    monkeypatch, tmp_path
):
    """Invalid configuration is handled the same way `skip_tailoring` already
    handles an unrecognised value: fall back to a safe default rather than
    raising and taking the whole app down over one optional setting."""
    for value in ("not-a-number", "  ", "seven"):
        monkeypatch.setenv("RESEARCH_CACHE_TTL_HOURS", value)

        settings = load_settings(env_file=tmp_path / "absent.env")

        assert settings.research_cache_ttl_hours == DEFAULT_RESEARCH_CACHE_TTL_HOURS


def test_research_cache_ttl_falls_back_to_the_default_when_unset(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("RESEARCH_CACHE_TTL_HOURS", raising=False)

    settings = load_settings(env_file=tmp_path / "absent.env")

    assert settings.research_cache_ttl_hours == DEFAULT_RESEARCH_CACHE_TTL_HOURS
