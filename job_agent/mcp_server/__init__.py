"""Custom MCP server (FR-8, NFR-9) - stub for now, built in Week 7.

Planned surface, mirroring the Week 5 MCP work:
  resource  applications://all   - the tracker pipeline, readable by any client
  tools     web_search, score_fit, tailor_cv, track_application

The point of exposing them over MCP is portability: another MCP client (e.g.
Claude Code) can use the same tracker and tools without importing this package.
Because tools/__init__.py already stores each tool's JSON schema, the server is
a thin adapter over the existing registry rather than a second definition of
every tool.
"""
