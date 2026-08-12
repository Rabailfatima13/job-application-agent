"""Concrete adapters: Anthropic, and anything OpenAI-compatible.

One OpenAI-compatible adapter covers Groq, Ollama, and OpenRouter - they
differ only by base_url and model name - which is why the "second provider"
required by FR-11 is a configuration choice rather than more code.

Message format shared by both adapters (provider-neutral):
    {"role": "user"|"assistant", "content": str}
    {"role": "tool_results", "results": [{"tool_use_id": str, "content": str}]}
"""

import json

from ..config import ModelConfig
from .base import ChatResult, ToolCall


def _tool_specs_anthropic(tools) -> list[dict]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


def _tool_specs_openai(tools) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


class AnthropicClient:
    """Claude, used for the judgement-heavy steps (scoring, writing)."""

    def __init__(self, config: ModelConfig) -> None:
        from anthropic import Anthropic  # imported lazily so tests need no SDK

        if not config.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example).")
        self.name = f"anthropic:{config.model}"
        self.model = config.model
        self._client = Anthropic(api_key=config.api_key)

    def complete(
        self,
        messages: list[dict],
        system: str,
        tools: list | None = None,
        max_tokens: int = 1024,
    ) -> ChatResult:
        converted = []
        for message in messages:
            if message["role"] == "tool_results":
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": r["tool_use_id"],
                                "content": r["content"],
                            }
                            for r in message["results"]
                        ],
                    }
                )
            else:
                converted.append(message)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=converted,
            **({"tools": _tool_specs_anthropic(tools)} if tools else {}),
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        ]
        return ChatResult(
            text=text,
            tool_calls=calls,
            stop_reason="tool_use" if calls else "end_turn",
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class OpenAICompatibleClient:
    """Groq / Ollama / OpenRouter - the cheap tier for parsing and summarising."""

    def __init__(self, config: ModelConfig) -> None:
        from openai import OpenAI  # imported lazily so tests need no SDK

        if not config.base_url:
            raise RuntimeError("LIGHT_BASE_URL is not set (see .env.example).")
        self.name = f"openai_compatible:{config.model}"
        self.model = config.model
        # Local runtimes such as Ollama ignore the key but the SDK requires one.
        self._client = OpenAI(
            api_key=config.api_key or "not-needed", base_url=config.base_url
        )

    def complete(
        self,
        messages: list[dict],
        system: str,
        tools: list | None = None,
        max_tokens: int = 1024,
    ) -> ChatResult:
        converted: list[dict] = [{"role": "system", "content": system}]
        for message in messages:
            if message["role"] == "tool_results":
                converted.extend(
                    {
                        "role": "tool",
                        "tool_call_id": r["tool_use_id"],
                        "content": r["content"],
                    }
                    for r in message["results"]
                )
            else:
                converted.append(message)

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=converted,
            **({"tools": _tool_specs_openai(tools)} if tools else {}),
        )
        message = response.choices[0].message
        calls = [
            ToolCall(
                id=c.id,
                name=c.function.name,
                arguments=json.loads(c.function.arguments or "{}"),
            )
            for c in (message.tool_calls or [])
        ]
        usage = response.usage
        return ChatResult(
            text=message.content or "",
            tool_calls=calls,
            stop_reason="tool_use" if calls else "end_turn",
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


def build_client(config: ModelConfig):
    """Return the adapter named by `config.provider`."""
    if config.provider == "anthropic":
        return AnthropicClient(config)
    if config.provider == "openai_compatible":
        return OpenAICompatibleClient(config)
    raise ValueError(
        f"Unknown provider '{config.provider}' "
        "(expected 'anthropic' or 'openai_compatible')."
    )
