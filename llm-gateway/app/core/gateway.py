"""Gateway orchestrator.

Provides a single entry point ``handle`` for chat completion requests.
The processing pipeline is:

``request → (optional) cache lookup → router.select_and_execute() →
quality‑gate → (optional) cache write → response``.

All cache interactions are *fail‑open*: any exception is logged and the
request proceeds without caching.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from app.api.schemas.requests import ChatCompletionRequest
from app.api.schemas.responses import ChatCompletionResponse
from app.core.exceptions import ProviderError
from app.router.selector import ProviderSelector

if TYPE_CHECKING:
    from app.cache.embedder import Embedder
    from app.cache.vector_store import VectorStore, SimilarityHit

logger = logging.getLogger(__name__)


class Gateway:
    """Core orchestrator for chat completions with optional caching."""

    def __init__(
        self,
        router: ProviderSelector,
        cache: "VectorStore | None" = None,
        embedder: "Embedder | None" = None,
        similarity_threshold: float = 0.85,
    ) -> None:
        self._router = router
        self._cache = cache
        self._embedder = embedder
        self._threshold = similarity_threshold

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _request_to_text(request: ChatCompletionRequest) -> str:
        """Flatten all message contents into a single string.

        Used for cache key generation and embedding. The test suite expects
        both system and user messages to be present.
        """
        return " ".join(msg.content for msg in getattr(request, "messages", []))

    @staticmethod
    def _cache_key(request: ChatCompletionRequest) -> str:
        """Deterministic SHA‑256 key based on the flattened request text."""
        text = Gateway._request_to_text(request)
        return hashlib.sha256(text.encode()).hexdigest()

    # ---------------------------------------------------------------------
    # Cache interaction
    # ---------------------------------------------------------------------
    async def _cache_lookup(self, request: ChatCompletionRequest) -> ChatCompletionResponse | None:
        """Attempt to retrieve a cached response.

        Returns ``None`` on miss or any failure. On a hit the ``source`` field
        is forced to ``"cache"`` to distinguish cached results from live
        provider responses.
        """
        if self._cache is None or self._embedder is None:
            return None
        try:
            key = self._cache_key(request)
            vector = self._embedder.embed(self._request_to_text(request))
            hit: SimilarityHit | None = await self._cache.search(vector, self._threshold)
            if hit is None:
                return None
            payload = await self._cache.get_payload(hit.key)
            if payload is None:
                return None
            response = ChatCompletionResponse.model_validate(payload)
            # Mark as coming from cache for downstream consumers.
            response.source = "cache"
            return response
        except Exception as exc:  # pragma: no cover – defensive
            logger.warning("cache_lookup_failed", extra={"error": str(exc)})
            return None

    async def _cache_write(self, request: ChatCompletionRequest, response: ChatCompletionResponse) -> None:
        """Write a cache entry for a successful response.

        Stores ``cached_prompt`` and ``source`` (the routing tier) while
        stripping the potentially large ``reasoning`` field.
        """
        if self._cache is None or self._embedder is None:
            return
        try:
            prompt_text = self._request_to_text(request)
            source = getattr(response, "routing_strategy", "unknown")
            payload = response.model_dump(mode="json")
            payload["source"] = source
            payload["cached_prompt"] = prompt_text
            payload.pop("reasoning", None)
            key = hashlib.sha256(prompt_text.encode()).hexdigest()
            vector = self._embedder.embed(prompt_text)
            await self._cache.put(key, vector, payload)
        except Exception as exc:
            logger.warning("cache_write_failed", extra={"error": str(exc)})

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    async def handle(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Process a request, using cache when appropriate.

        If ``request.bypass_cache`` is ``True`` the lookup is skipped but a
        write may still occur on a miss.
        """
        # 1️⃣ Cache lookup (unless bypassed)
        if not getattr(request, "bypass_cache", False):
            cached = await self._cache_lookup(request)
            if cached is not None:
                return cached

        # 2️⃣ Generate via provider selector
        response = await self._router.select_and_execute(request)

        # 3️⃣ Do not cache degraded responses (quality gate will have marked)
        if not response.degraded:
            await self._cache_write(request, response)

        return response

    async def cache_health(self) -> bool:
        """Return ``True`` if the cache is healthy or not configured."""
        if self._cache is None:
            return True
        try:
            return await self._cache.health()
        except Exception:  # pragma: no cover – defensive
            return False

    # ---------------------------------------------------------------------
    # Provider health helpers used by the readiness endpoint
    # ---------------------------------------------------------------------
    async def provider_health(self) -> bool:
        """Delegate health check to the underlying ``ProviderSelector``.

        Returns ``True`` when at least one provider reports healthy.
        """
        try:
            return await self._router.health()
        except Exception:  # pragma: no cover – defensive
            return False

    async def extended_health(self) -> dict[str, object]:
        """Return detailed health information for readiness and monitoring.

        Structure mirrors the ``/health/providers`` endpoint expectations.
        """
        provider_ok = await self.provider_health()
        cache_ok = await self.cache_health()
        return {
            "status": "ok" if provider_ok and cache_ok else "degraded",
            "components": {
                "provider": "ok" if provider_ok else "unhealthy",
                "cache": "ok" if cache_ok else "unavailable",
            },
            "providers": await self._router.get_circuit_breaker_states(),
        }
