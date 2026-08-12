"""Job Application Agent.

A supervisor + worker-agent system that, for one candidate CV and a given job
description, researches the company, scores candidate-role fit with evidence,
tailors application material using only claims already present in the CV, and
tracks each application in a persistent store.

Package layout mirrors the proposal's four layers:

    models/         structured inputs and outputs (Pydantic)         [input]
    tools/          narrow, independently testable capabilities      [capability]
    memory/         persistent tracker + per-run session state       [capability]
    agents/         supervisor + research / scoring / writing        [orchestration]
    validation/     schema guards, retry, no-fabrication grounding   [orchestration]
    llm/            provider-agnostic model clients + task routing    [orchestration]
    observability/  per-tool-call trace events (cost, latency)       [orchestration]
    mcp_server/     exposes tracker resource + tools over MCP        [capability]
    app/            Streamlit demo surface                           [presentation]
"""

__version__ = "0.1.0"
