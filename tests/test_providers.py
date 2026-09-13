"""`OpenAICompatibleClient` request-building (the Gemini/Groq adapter).

Regression coverage for the structured-output fix: a live Gemini run against
`gemini-3.6-flash` produced three 200 OK responses that were nonetheless
unusable - truncated JSON, prose instead of JSON, and an empty completion.
`generate_validated`/`parse_structured` correctly rejected all three; the fix
sits one layer down, in what this client actually asks the provider for.

No network call is made anywhere in this file. `OpenAICompatibleClient.__init__`
only constructs an `openai.OpenAI` client object - real, but inert until a
method is called - and every test replaces `._client.chat.completions.create`
with a local stub before `complete()` is ever invoked, so the real
message-building and response-parsing code runs against a captured request and
a canned response instead of the network.
"""

from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError, BadRequestError, OpenAI

from job_agent.config import ModelConfig
from job_agent.llm.providers import (
    JSON_MODE_TEMPERATURE,
    OpenAICompatibleClient,
    build_client,
)


def fake_response(content: str = "ok", tool_calls=None) -> SimpleNamespace:
    """The minimal shape `complete()` actually reads off an OpenAI response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )


def stub_client(config: ModelConfig | None = None):
    """A real `OpenAICompatibleClient` with the network boundary replaced.

    Returns `(client, calls)`; `calls` records the exact keyword arguments
    every `complete()` call in the test sent to `chat.completions.create`.
    """
    config = config or ModelConfig(
        provider="openai_compatible",
        model="gemini-3.6-flash",
        api_key="test-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    client = OpenAICompatibleClient(config)
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return fake_response()

    client._client.chat.completions.create = create
    return client, calls


def api_status_error(raw_body: object, status_code: int = 400):
    """A real `openai` SDK exception, built the way the SDK itself builds one
    from an HTTP response - via its own `_make_status_error_from_response`,
    not a hand-rolled stand-in for it.

    This matters: the SDK unwraps `{"error": {...}}` to `{...}` before
    attaching it as `exc.body` (`openai/_client.py`:
    `data = body.get("error", body)`). A live run caught the first version of
    this helper passing the *wrapped* shape straight through as `.body`
    itself - every test here passed, and the fix it was meant to verify was
    still broken live, because the fixture had quietly stopped matching
    reality. `raw_body` is the wire-format JSON a provider actually sends
    (envelope included); the SDK does the unwrapping, exactly as it would
    for a real response. Constructing a client and a local `httpx.Response`
    perform no I/O - this makes no network call.
    """
    client = OpenAI(api_key="test", base_url="https://example.invalid/v1")
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(status_code, request=request, json=raw_body)
    return client._make_status_error_from_response(response)


def _raiser(exc: Exception):
    """A `chat.completions.create` stub that raises instead of replying."""

    def create(**_kwargs):
        raise exc

    return create


# --- the structured-output fix -----------------------------------------------


def test_a_plain_call_requests_json_mode():
    """The failure this fix targets: a 200 OK reply that isn't valid JSON.
    json_object mode constrains the provider's own decoding, rather than
    leaving JSON-only compliance to the prompt alone."""
    client, calls = stub_client()

    client.complete([{"role": "user", "content": "hi"}], system="Reply with JSON.")

    assert calls[0]["response_format"] == {"type": "json_object"}


def test_a_plain_call_uses_the_deterministic_temperature():
    client, calls = stub_client()

    client.complete([{"role": "user", "content": "hi"}], system="Reply with JSON.")

    assert calls[0]["temperature"] == JSON_MODE_TEMPERATURE
    assert JSON_MODE_TEMPERATURE == 0.0


def test_json_mode_is_skipped_when_tools_are_supplied():
    """Forced JSON output and function-calling are not meant to be combined -
    a defensive guard for the one code path that can pass `tools`, even though
    no agent in this project does so today."""
    from job_agent.tools import Tool

    client, calls = stub_client()
    tool = Tool(
        name="lookup",
        description="Look something up.",
        input_schema={"type": "object", "properties": {}},
        run=lambda: None,
    )

    client.complete(
        [{"role": "user", "content": "hi"}], system="s", tools=[tool]
    )

    assert "response_format" not in calls[0]
    assert calls[0]["tools"][0]["function"]["name"] == "lookup"
    # Determinism is unrelated to tool-calling, so it still applies.
    assert calls[0]["temperature"] == JSON_MODE_TEMPERATURE


def test_gemini_and_groq_both_get_json_mode():
    """The fix lives in the shared adapter, not a per-provider branch - it
    applies identically whichever `openai_compatible` endpoint is configured."""
    gemini, gemini_calls = stub_client(
        ModelConfig(
            provider="openai_compatible",
            model="gemini-3.6-flash",
            api_key="k",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    )
    groq, groq_calls = stub_client(
        ModelConfig(
            provider="openai_compatible",
            model="llama-3.3-70b-versatile",
            api_key="k",
            base_url="https://api.groq.com/openai/v1",
        )
    )

    gemini.complete([{"role": "user", "content": "hi"}], system="Reply with JSON.")
    groq.complete([{"role": "user", "content": "hi"}], system="Reply with JSON.")

    assert gemini_calls[0]["response_format"] == {"type": "json_object"}
    assert groq_calls[0]["response_format"] == {"type": "json_object"}


def test_qwen_requests_disable_reasoning():
    """The fix for the empty-completion failure: a live CVExtraction run
    against a reasoning model returned content='' because its chain-of-thought
    consumed the whole max_tokens budget before it reached the answer. Qwen
    documents reasoning_effort="none" as disabling reasoning outright."""
    client, calls = stub_client(
        ModelConfig(
            provider="openai_compatible",
            model="qwen/qwen3.8-27b",
            api_key="k",
            base_url="https://api.groq.com/openai/v1",
        )
    )

    client.complete([{"role": "user", "content": "hi"}], system="Reply with JSON.")

    assert calls[0]["reasoning_effort"] == "none"
    # Unrelated to the fix - still requested, exactly as for any other model.
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_gpt_oss_requests_use_low_reasoning_effort():
    """A second live failure, this time on the HEAVY tier: a scoring run
    against `gpt-oss:120b-cloud` measured its hidden reasoning alone
    consuming 60-70% of SCORING_MAX_TOKENS with no override, occasionally
    enough to truncate the JSON answer mid-string. Unlike Qwen, gpt-oss never
    documents "none" - only low/medium/high - so this is the lowest value it
    actually accepts; measured against the same real request, it cut
    reasoning's share by roughly 40% and left every response valid."""
    for model in ("gpt-oss:120b-cloud", "openai/gpt-oss-20b"):
        client, calls = stub_client(
            ModelConfig(
                provider="openai_compatible",
                model=model,
                api_key="k",
                base_url="https://example.invalid/v1",
            )
        )

        client.complete([{"role": "user", "content": "hi"}], system="s")

        assert calls[0]["reasoning_effort"] == "low"


def test_non_reasoning_models_never_receive_reasoning_effort():
    """A model that never documented this parameter must see an unchanged
    request - an unrecognised field could be rejected outright by a stricter
    OpenAI-compatible server (Ollama, OpenRouter)."""
    for model in ("gemini-3.6-flash", "llama-3.3-70b-versatile"):
        client, calls = stub_client(
            ModelConfig(
                provider="openai_compatible",
                model=model,
                api_key="k",
                base_url="https://example.invalid/v1",
            )
        )

        client.complete([{"role": "user", "content": "hi"}], system="s")

        assert "reasoning_effort" not in calls[0]


# --- existing behaviour is unchanged ------------------------------------------


def test_the_model_and_token_limit_are_still_sent():
    client, calls = stub_client()

    client.complete(
        [{"role": "user", "content": "hi"}], system="s", max_tokens=512
    )

    assert calls[0]["model"] == "gemini-3.6-flash"
    assert calls[0]["max_tokens"] == 512


def test_the_system_prompt_is_still_prepended_as_a_system_message():
    client, calls = stub_client()

    client.complete([{"role": "user", "content": "hi"}], system="be terse")

    assert calls[0]["messages"][0] == {"role": "system", "content": "be terse"}
    assert calls[0]["messages"][1] == {"role": "user", "content": "hi"}


def test_tool_results_are_still_converted_to_openai_tool_messages():
    client, calls = stub_client()

    client.complete(
        [{"role": "tool_results", "results": [{"tool_use_id": "t1", "content": "42"}]}],
        system="s",
    )

    assert calls[0]["messages"][1] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": "42",
    }


def test_the_response_text_and_token_counts_are_still_parsed_out():
    client, calls = stub_client()
    client._client.chat.completions.create = lambda **kw: fake_response(
        content="hello"
    )

    result = client.complete([{"role": "user", "content": "hi"}], system="s")

    assert result.text == "hello"
    assert result.input_tokens == 11
    assert result.output_tokens == 7
    assert result.model == "gemini-3.6-flash"


def test_a_missing_message_content_still_becomes_an_empty_string():
    """The one failure mode json_object mode cannot rule out by itself - a
    provider that returns no content at all. `complete()` must still hand
    `parse_structured` an empty string, not None, so the existing "Expecting
    value: line 1 column 1" validation error - not a crash - is what surfaces."""
    client, calls = stub_client()
    client._client.chat.completions.create = lambda **kw: fake_response(content=None)

    result = client.complete([{"role": "user", "content": "hi"}], system="s")

    assert result.text == ""


def test_a_missing_base_url_still_raises_before_any_request():
    with pytest.raises(RuntimeError, match="LIGHT_BASE_URL"):
        OpenAICompatibleClient(
            ModelConfig(provider="openai_compatible", model="m", base_url=None)
        )


def test_build_client_rejects_an_unrecognised_provider():
    """Both tiers go through `build_client` on every run - a typo'd
    HEAVY_PROVIDER/LIGHT_PROVIDER must fail loudly, not silently pick one."""
    with pytest.raises(ValueError, match="Unknown provider 'groq'"):
        build_client(ModelConfig(provider="groq", model="m"))


# --- recovering a provider-side json-mode rejection ---------------------------
#
# The regression: response_format={"type": "json_object"} made Groq validate
# JSON server-side and reject a malformed generation as an HTTP 400 instead of
# returning it as an ordinary reply. That bypassed generate_validated() (which
# only catches pydantic.ValidationError) and the supervisor's own best-effort
# handling for a failed research step - a run-ending exception where there
# used to be a retryable malformed response. These tests build the real SDK
# exception locally (no network) and prove the adapter puts its text back on
# the normal path, and nothing else.


def test_json_validate_failed_with_failed_generation_recovers_the_text():
    client, _ = stub_client()
    exc = api_status_error(
        {
            "error": {
                "message": "Failed to generate JSON. Please adjust your prompt.",
                "type": "invalid_request_error",
                "code": "json_validate_failed",
                "failed_generation": '{"summary": Nexora Analytics is a platform',
            }
        },
    )
    assert isinstance(exc, BadRequestError)  # the SDK picked the right class
    client._client.chat.completions.create = _raiser(exc)

    result = client.complete([{"role": "user", "content": "hi"}], system="s")

    assert result.text == '{"summary": Nexora Analytics is a platform'
    assert result.model == client.model


def test_an_ordinary_bad_request_error_still_raises():
    """Not every 400 is a json-mode rejection - an unrelated bad request must
    propagate exactly as it did before this fix existed."""
    client, _ = stub_client()
    exc = api_status_error(
        {"error": {"message": "model not found", "code": "model_not_found"}},
    )
    client._client.chat.completions.create = _raiser(exc)

    with pytest.raises(BadRequestError) as caught:
        client.complete([{"role": "user", "content": "hi"}], system="s")
    assert caught.value is exc


def test_a_missing_failed_generation_still_raises():
    client, _ = stub_client()
    exc = api_status_error(
        {
            "error": {
                "message": "Failed to generate JSON.",
                "code": "json_validate_failed",
            }
        },
    )
    client._client.chat.completions.create = _raiser(exc)

    with pytest.raises(BadRequestError) as caught:
        client.complete([{"role": "user", "content": "hi"}], system="s")
    assert caught.value is exc


def test_a_non_string_failed_generation_still_raises():
    client, _ = stub_client()
    exc = api_status_error(
        {
            "error": {
                "message": "Failed to generate JSON.",
                "code": "json_validate_failed",
                "failed_generation": {"not": "a string"},
            }
        },
    )
    client._client.chat.completions.create = _raiser(exc)

    with pytest.raises(BadRequestError) as caught:
        client.complete([{"role": "user", "content": "hi"}], system="s")
    assert caught.value is exc


def test_authentication_errors_are_not_touched_by_the_recovery_path():
    """Auth failures, rate limits, and other non-400 SDK errors never enter
    the json-mode recovery logic at all - only BadRequestError is caught."""
    client, _ = stub_client()
    exc = api_status_error({"error": {"message": "invalid api key"}}, status_code=401)
    assert isinstance(exc, AuthenticationError)  # the SDK picked this from the 401
    client._client.chat.completions.create = _raiser(exc)

    with pytest.raises(AuthenticationError) as caught:
        client.complete([{"role": "user", "content": "hi"}], system="s")
    assert caught.value is exc


def test_a_successful_response_is_unaffected_by_the_recovery_path():
    """e: the ordinary success path - already covered above by
    test_the_response_text_and_token_counts_are_still_parsed_out - is
    re-asserted here specifically alongside the recovery-path tests, so the
    two behaviours are proven side by side."""
    client, calls = stub_client()

    result = client.complete([{"role": "user", "content": "hi"}], system="s")

    assert result.text == "ok"
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_the_tools_path_is_unaffected_by_the_recovery_path():
    """f: a tools-bearing call that succeeds normally never touches the
    except-branch at all - it isn't in json_object mode in the first place."""
    from job_agent.tools import Tool

    client, calls = stub_client()
    tool = Tool(
        name="lookup",
        description="Look something up.",
        input_schema={"type": "object", "properties": {}},
        run=lambda: None,
    )

    result = client.complete(
        [{"role": "user", "content": "hi"}], system="s", tools=[tool]
    )

    assert result.text == "ok"
    assert "response_format" not in calls[0]
    assert calls[0]["tools"][0]["function"]["name"] == "lookup"
