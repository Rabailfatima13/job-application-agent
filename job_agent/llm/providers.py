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

# Every prompt in this project asks for "a single JSON object and nothing
# else" and every reply goes straight into `parse_structured` - so an
# OpenAI-compatible call is never asking for open-ended prose. Pinning a
# deterministic, near-zero temperature is the appropriate setting for that:
# it does not replace validation, it just makes the model's own generation
# steadier so what it emits is closer to the single JSON object being asked
# for in the first place.
JSON_MODE_TEMPERATURE = 0.0


def _json_validate_failure_text(exc) -> str | None:
    """The provider's own malformed generation, if `exc` is specifically a
    json_object-mode rejection carrying one - `None` for anything else,
    meaning the caller should re-raise `exc` unchanged.

    Some OpenAI-compatible providers (Groq, observed live) validate JSON
    mode server-side and reject a malformed generation as an HTTP 400
    instead of returning it as an ordinary reply. Before `response_format`
    was added, that same malformed text came back as a normal 200 and
    `parse_structured`/`generate_validated` judged it - rejecting it,
    retrying with feedback, or accepting it. An exception raised out of
    `complete()` skips all of that, so this recovers the text rather than
    letting the provider's own opinion of validity replace ours. It performs
    no validation or repair itself: a schema-shaped string comes back, or
    nothing does.

    Deliberately narrow: only the exact `json_validate_failed` code with a
    string `failed_generation` is recovered. Anything else - a differently
    shaped 400, a missing or non-string field - is a real error this adapter
    has no basis for hiding, so the caller re-raises it untouched.

    `exc.body` is already the *inner* error object, not the raw response
    envelope: the SDK's own `_make_status_error` unwraps
    `{"error": {...}}` to `{...}` before attaching it to the exception
    (`openai/_client.py`: `data = body.get("error", body)`), and it is that
    unwrapped `data` the exception is constructed with - confirmed by
    reading the installed SDK source after a live run raised this and the
    first version of this function (which re-read the now-absent "error"
    key) missed it and re-raised instead of recovering.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    if body.get("code") != "json_validate_failed":
        return None
    failed_generation = body.get("failed_generation")
    return failed_generation if isinstance(failed_generation, str) else None


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


# Reasoning models draw their internal chain-of-thought from the same
# max_tokens budget as the actual answer, and can silently exhaust it on
# reasoning alone - a live CVExtraction run returned an empty completion this
# way. Qwen's family documents reasoning_effort="none" as disabling reasoning
# entirely; gpt-oss (Groq's other reasoning family) only supports
# low/medium/high, never "none", so this stays a name-scoped addition rather
# than a value every openai_compatible model is assumed to accept - an
# unrecognised field could be rejected outright by a stricter server.
_REASONING_EFFORT_NONE_MODELS = ("qwen",)


def _reasoning_kwargs(model: str) -> dict:
    if any(name in model.lower() for name in _REASONING_EFFORT_NONE_MODELS):
        return {"reasoning_effort": "none"}
    return {}


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
            # This adapter serves either tier, so the missing setting is
            # whichever of HEAVY_BASE_URL / LIGHT_BASE_URL configures the
            # tier being built - naming only one would send a heavy-tier user
            # debugging the wrong variable.
            raise RuntimeError(
                "base_url is not set for the openai_compatible provider "
                "(see HEAVY_BASE_URL / LIGHT_BASE_URL in .env.example)."
            )
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
        from openai import BadRequestError  # imported lazily so tests need no SDK

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

        # `response_format={"type": "json_object"}` is the OpenAI "JSON mode"
        # contract: it constrains the provider's own decoding to syntactically
        # valid JSON, rather than relying on the prompt alone to produce it.
        # Google's Gemini OpenAI-compatibility layer documents support for it
        # in real-time (non-batch) calls, which is what this client makes.
        # Skipped when tools are supplied - forced JSON output and
        # function-calling are not meant to be combined, and every call in
        # this codebase today passes `tools=None` regardless.
        json_mode = {} if tools else {"response_format": {"type": "json_object"}}

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=JSON_MODE_TEMPERATURE,
                messages=converted,
                **({"tools": _tool_specs_openai(tools)} if tools else {}),
                **json_mode,
                **_reasoning_kwargs(self.model),
            )
        except BadRequestError as exc:
            text = _json_validate_failure_text(exc)
            if text is None:
                raise
            # No `response` object exists to read usage off - the request
            # was rejected before one came back - so token counts are simply
            # unavailable here, exactly as `ChatResult`'s defaults already say.
            return ChatResult(text=text, model=self.model)

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
