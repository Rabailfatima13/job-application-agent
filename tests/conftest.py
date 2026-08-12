"""Shared test doubles.

No test in this project may require an API key or network access: the model is
always a stub satisfying the ModelClient protocol.
"""

from datetime import datetime

import pytest

from job_agent.config import ModelConfig, Settings
from job_agent.llm import ChatResult, ModelRouter
from job_agent.models import ApplicationRecord


class StubModelClient:
    """A ModelClient that replays canned replies and records what it was asked."""

    def __init__(self, replies: list[str] | None = None, name: str = "stub") -> None:
        self.name = name
        self._replies = list(replies or ["ok"])
        self.calls: list[dict] = []

    def complete(self, messages, system, tools=None, max_tokens=1024) -> ChatResult:
        self.calls.append({"messages": messages, "system": system, "tools": tools})
        text = self._replies.pop(0) if self._replies else "ok"
        return ChatResult(
            text=text, model=self.name, input_tokens=10, output_tokens=5
        )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        heavy=ModelConfig(provider="anthropic", model="heavy-model", api_key="test"),
        light=ModelConfig(
            provider="openai_compatible",
            model="light-model",
            api_key="test",
            base_url="http://localhost/v1",
        ),
        search_provider="serpapi",
        search_api_key="test",
        tracker_db_path=tmp_path / "applications.db",
        tool_call_log_path=tmp_path / "tool_calls.log",
    )


@pytest.fixture
def router(settings) -> ModelRouter:
    """A router whose tiers resolve to stubs, so no provider SDK is touched."""
    clients = {
        "heavy-model": StubModelClient(name="stub-heavy"),
        "light-model": StubModelClient(name="stub-light"),
    }
    return ModelRouter(settings, client_factory=lambda cfg: clients[cfg.model])


@pytest.fixture
def application_record() -> ApplicationRecord:
    return ApplicationRecord(
        application_id="app-1",
        role="ML Engineer",
        company="Arbisoft",
        fit_score=0.75,
        created_at=datetime(2026, 8, 12, 9, 0),
    )
