"""Provider-agnostic model access.

Agents never import a vendor SDK. They receive a `ModelClient` and call
`.complete(...)`, so adding or swapping a provider (FR-11) means adding one
adapter in providers.py and one line in router.py - no agent code changes.
"""

from .base import ChatResult, ModelClient, ToolCall
from .router import ModelRouter

__all__ = ["ChatResult", "ModelClient", "ModelRouter", "ToolCall"]
