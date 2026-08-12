"""Configuration reads from the environment and never demands credentials just
to import the package."""

from job_agent.config import HEAVY, LIGHT, load_settings


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
