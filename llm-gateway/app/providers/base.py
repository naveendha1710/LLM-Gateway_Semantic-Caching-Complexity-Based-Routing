"""Base model provider protocol.

Every provider (cloud, local, future ones) implements this protocol so the
router and gateway can treat them uniformly. Adding a new provider means
adding one new file that implements this protocol — nothing else changes.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.api.schemas.requests import ChatCompletionRequest
from app.api.schemas.responses import ChatCompletionResponse


@runtime_checkable
class ModelProvider(Protocol):
    """Interface every model provider must implement."""

    async def generate(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Produce a completion for the given request."""
        ...

    async def health(self) -> bool:
        """Return True if the provider is ready to serve requests."""
        ...

    @property
    def name(self) -> str:
        """Human-readable provider name for logs and metrics."""
        ...

    @property
    def cost_per_1k(self) -> float:
        """Cost in USD per 1K tokens (for accounting / routing decisions)."""
        ...
