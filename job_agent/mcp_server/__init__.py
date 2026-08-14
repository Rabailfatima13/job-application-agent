"""Custom MCP server (FR-8, NFR-9).

Surface:
  resource  applications://all      - the tracker pipeline, readable by any client
  tools     list_applications, get_application, track_application,
            set_application_status, search_company

The point of exposing them over MCP is portability: another MCP client (e.g.
Claude Code) can use the same tracker without importing this package. The
server is a thin adapter over the existing tracker and search tool rather than
a second implementation of either - see server.py for what is deliberately not
exposed and why.

Run it with:  python -m job_agent.mcp_server.server
"""
