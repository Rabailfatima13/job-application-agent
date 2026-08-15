"""The MCP server driven through the real protocol (FR-8, NFR-9).

`test_mcp_server.py` calls the server's functions in-process; these tests spawn
it as a subprocess and talk to it over stdio exactly as Claude Code or any
other MCP client would. That is the part FR-8 actually promises: another client
can discover and use the tracker without importing this package.

The subprocess is pointed at a tmp_path database via TRACKER_DB_PATH and a
tmp_path trace log via TOOL_CALL_LOG_PATH, so no real tracker or the real
data/tool_calls.log is touched, and nothing here needs an API key or a
network.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

REPO_ROOT = Path(__file__).resolve().parent.parent

# Talking to a subprocess over stdio is slower than an in-process call; these
# stay a small, deliberate subset rather than a duplicate of the unit tests.
pytestmark = pytest.mark.filterwarnings("ignore::ResourceWarning")


def run_session(db_path: Path, work):
    """Spawn the server, hand an initialised MCP session to `work`, return its
    result. Synchronous wrapper so the tests read like the rest of the suite."""

    async def _main():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "job_agent.mcp_server.server"],
            cwd=str(REPO_ROOT),
            env={
                **os.environ,
                "TRACKER_DB_PATH": str(db_path),
                "TOOL_CALL_LOG_PATH": str(db_path.parent / "tool_calls.log"),
            },
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await work(session)

    return asyncio.run(_main())


def run_session_without_credentials(db_path: Path, work):
    """Same as `run_session`, but the subprocess gets no provider keys.

    Used to observe failure handling without spending a single token: the
    model-backed tools cannot reach a provider, so they fail immediately.
    """
    stripped = {
        key: value
        for key, value in os.environ.items()
        if key not in {"LIGHT_API_KEY", "ANTHROPIC_API_KEY", "SEARCH_API_KEY"}
    }

    async def _main():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "job_agent.mcp_server.server"],
            cwd=str(REPO_ROOT),
            env={
                **stripped,
                "TRACKER_DB_PATH": str(db_path),
                "TOOL_CALL_LOG_PATH": str(db_path.parent / "tool_calls.log"),
                # An unknown provider fails when the client is built, before a
                # socket is ever opened - instant, and impossible to bill.
                # (Pointing at an unreachable URL instead would work, but the
                # provider SDK retries with long backoff and the test crawls.)
                "LIGHT_PROVIDER": "unconfigured-for-tests",
                "LIGHT_API_KEY": "",
            },
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await work(session)

    return asyncio.run(_main())


@pytest.fixture
def db_path(tmp_path) -> Path:
    return tmp_path / "applications.db"


def test_a_client_can_connect_and_discover_the_surface(db_path):
    async def work(session):
        tools = await session.list_tools()
        resources = await session.list_resources()
        return (
            sorted(t.name for t in tools.tools),
            [str(r.uri) for r in resources.resources],
        )

    tool_names, resource_uris = run_session(db_path, work)

    assert tool_names == [
        "get_application",
        "list_applications",
        "score_fit",
        "search_company",
        "set_application_status",
        "tailor_application",
        "track_application",
    ]
    assert resource_uris == ["applications://all"]


def rows_from(result) -> list[dict]:
    """A list-returning MCP tool sends one content block per item."""
    return [json.loads(block.text) for block in result.content]


def test_every_tool_advertises_a_typed_input_schema(db_path):
    async def work(session):
        tools = await session.list_tools()
        return {t.name: t.input_schema for t in tools.tools}

    schemas = run_session(db_path, work)

    assert schemas["track_application"]["properties"]["fit_score"]["type"] == "number"
    assert schemas["track_application"]["properties"]["role"]["type"] == "string"
    assert "role" in schemas["track_application"]["required"]
    assert "application_id" in schemas["get_application"]["required"]
    # Optional arguments are genuinely optional.
    assert "status" not in schemas["list_applications"].get("required", [])


def test_an_application_tracked_over_the_protocol_is_readable_as_a_resource(db_path):
    async def work(session):
        created = await session.call_tool(
            "track_application",
            {"role": "Junior AI Engineer", "company": "Arbisoft", "fit_score": 0.846},
        )
        body = await session.read_resource("applications://all")
        return json.loads(created.content[0].text), json.loads(body.contents[0].text)

    record, rows = run_session(db_path, work)

    assert record["status"] == "draft"
    assert record["fit_score"] == 0.846
    assert [r["application_id"] for r in rows] == [record["application_id"]]


def test_the_pipeline_survives_between_client_sessions(db_path):
    """The whole point of the resource: a later client sees earlier work."""

    async def create(session):
        result = await session.call_tool(
            "track_application",
            {"role": "Data Engineer", "company": "Arbisoft", "fit_score": 0.5},
        )
        return json.loads(result.content[0].text)["application_id"]

    async def read_back(session):
        result = await session.call_tool("list_applications", {})
        return rows_from(result)

    application_id = run_session(db_path, create)
    rows = run_session(db_path, read_back)  # a brand new server process

    assert [r["application_id"] for r in rows] == [application_id]


def test_status_can_be_moved_over_the_protocol(db_path):
    async def work(session):
        created = await session.call_tool(
            "track_application",
            {"role": "Junior AI Engineer", "company": "Arbisoft", "fit_score": 0.7},
        )
        application_id = json.loads(created.content[0].text)["application_id"]
        updated = await session.call_tool(
            "set_application_status",
            {"application_id": application_id, "status": "ready"},
        )
        fetched = await session.call_tool(
            "get_application", {"application_id": application_id}
        )
        return json.loads(updated.content[0].text), json.loads(fetched.content[0].text)

    updated, fetched = run_session(db_path, work)

    assert updated["status"] == "ready"
    assert fetched["status"] == "ready"


def test_an_unknown_application_comes_back_as_a_protocol_error(db_path):
    async def work(session):
        result = await session.call_tool(
            "get_application", {"application_id": "does-not-exist"}
        )
        return result.is_error, result.content[0].text

    is_error, message = run_session(db_path, work)

    assert is_error
    assert "No application tracked" in message


def test_an_invalid_argument_comes_back_as_a_protocol_error(db_path):
    async def work(session):
        result = await session.call_tool(
            "track_application",
            {"role": "A", "company": "Arbisoft", "fit_score": 9.9},
        )
        return result.is_error, result.content[0].text

    is_error, message = run_session(db_path, work)

    assert is_error
    # The same Pydantic bound the pipeline enforces, surfaced over MCP.
    assert "fit_score" in message


def test_the_model_backed_tools_are_discoverable_with_typed_schemas(db_path):
    async def work(session):
        tools = {t.name: t for t in (await session.list_tools()).tools}
        return {
            name: (tools[name].input_schema, tools[name].description)
            for name in ("score_fit", "tailor_application")
        }

    surface = run_session(db_path, work)

    score_schema, score_doc = surface["score_fit"]
    assert score_schema["properties"]["cv"]["type"] == "string"
    assert sorted(score_schema["required"]) == ["cv", "job_description"]
    # The cost is stated in the description a client will actually read.
    assert "model calls" in score_doc

    tailor_schema, tailor_doc = surface["tailor_application"]
    assert sorted(tailor_schema["required"]) == ["cv", "job_description"]
    assert "model calls" in tailor_doc


def test_a_model_backed_tool_reports_failure_over_the_protocol(db_path):
    # No provider credentials are configured for the subprocess, so the call
    # fails - which is what we want to observe: a clean protocol error, and no
    # credential in the message. This test therefore spends nothing.
    async def work(session):
        result = await session.call_tool(
            "score_fit", {"cv": "Built a FastAPI service.", "job_description": "Python"}
        )
        return result.is_error, result.content[0].text

    is_error, message = run_session_without_credentials(db_path, work)

    assert is_error
    assert "API_KEY" not in message.upper() or "not set" in message
    for secret_marker in ("sk-", "gsk_", "Bearer "):
        assert secret_marker not in message


def test_the_error_message_carries_no_configuration_values(db_path):
    async def work(session):
        result = await session.call_tool(
            "set_application_status",
            {"application_id": "does-not-exist", "status": "ready"},
        )
        return result.content[0].text

    message = run_session(db_path, work)

    assert "api_key" not in message.lower()
    assert str(db_path) not in message
