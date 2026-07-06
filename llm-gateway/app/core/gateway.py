"""Gateway orchestrator.

Single entry point for serving a chat completion request. In Phase 1 the
flow is:

    request → (cache stub) → provider.generate() → (quality gate stub) → response

The cache and quality-gate branches are stubbed so the gateway works end-to
-end today, but the seams are in place for Phase 2 (semantic cache) and the
router (Phase 3).
"""

from __future__ import annotations

import logging
from typing import Any

from app.api.schemas.requests import ChatCompletionRequest
from app.api.schemas.responses import ChatCompletionResponse
from app.core.exceptions import AllProvidersFailed, ProviderError
from app.providers.base import ModelProvider

logger = logging.getLogger(__name__)


class Gateway:
    """Orchestrates cache lookup, routing, generation, and cache write."""

    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider
        # Cache backend is injected in Phase 2.
        self._cache: Any = None

    async def handle(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Serve a chat completion request."""
        # ── 1. Cache lookup (stubbed for Phase 1) ───────────────
        if not request.bypass_cache:
            cached = await self._cache_lookup(request)
            if cached is not None:
                logger.info("cache_hit")
                return cached

        # ── 2. Route to provider (single provider in Phase 1) ───
        try:
            response = await self._provider.generate(request)
        except ProviderError as exc:
            logger.warning("provider_failed", extra={"error": str(exc)})
            raise AllProvidersFailed(str(exc)) from exc

        # ── 3. Quality gate (stubbed — always passes in Phase 1) ─
        if not self._quality_gate(response):
            logger.warning("quality_gate_rejected")
            # Fall through to return anyway in Phase 1; real gate in Phase 2.
            pass

        # ── 4. Cache write (stubbed for Phase 1) ────────────────
        await self._cache_write(request, response)

        return response

    # ── Cache seams (implemented in Phase 2) ─────────────────────
    async def _cache_lookup(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse | None:
        return None

    async def _cache_write(
        self,
        request: ChatCompletionRequest,
        response: ChatCompletionResponse,
    ) -> None:
        pass

    # ── Quality gate seam (implemented in Phase 2) ───────────────
    @staticmethod
    def _quality_gate(response: ChatCompletionResponse) -> bool:
        return True

    # ── Health checks for /readyz ────────────────────────────────
    async def provider_health(self) -> bool:
        return await self._provider.health()

    async def cache_health(self) -> bool:
        # No cache backend in Phase 1.
        return True
