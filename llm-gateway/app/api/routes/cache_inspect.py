"""Cache inspection API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/cache", tags=["cache"])


class CacheInspectRequest(BaseModel):
    """Request for cache inspection."""

    query: str = Field(..., description="Query text to search for similar cached entries")
    top_n: int = Field(default=5, ge=1, le=20, description="Number of top results to return")


class CacheInspectResult(BaseModel):
    """Single cache inspection result."""

    key: str = Field(..., description="Cache key")
    score: float = Field(..., description="Similarity score (0-1)")
    payload: dict = Field(..., description="Cached payload")
    # Additional fields for CLI rendering
    prompt: str = Field("", description="Original cached prompt")
    source: str = Field("unknown", description="Cache source/tier")
    age_seconds: float = Field(0.0, description="Age of the cache entry in seconds")


class CacheInspectResponse(BaseModel):
    """Response for cache inspection."""

    results: list[CacheInspectResult] = Field(default_factory=list, description="Top matching cache entries")
    total_checked: int = Field(..., description="Total entries checked")


@router.post("/inspect", response_model=CacheInspectResponse)
async def inspect_cache(
    request: CacheInspectRequest,
    http_request: Request,
) -> CacheInspectResponse:
    """Inspect cache for similar entries to a query."""
    from app.core.gateway import Gateway

    app_state = http_request.app.state
    gateway: Gateway | None = getattr(app_state, "gateway", None)

    if gateway is None or gateway.cache is None:
        raise HTTPException(status_code=503, detail="Cache is not enabled")

    # ``gateway.cache_inspect`` may be missing or broken; perform inspection
    # directly using the underlying VectorStore.
    # Retrieve embedder configuration.
    from app.core.config import get_settings
    from app.cache.embedder import get_embedder

    settings = get_settings()
    # Use the embedder attached to the gateway if available; otherwise create a new one.
    embedder = getattr(gateway, "_embedder", None)
    if embedder is None:
        embedder = get_embedder(settings.cache_embedder_kind, settings.cache_embedder_dimension)

    # Embed the query and perform a top‑N similarity search.
    query_vec = embedder.embed(request.query)
    # Use the same threshold as the gateway (default 0.85).
    threshold = getattr(gateway, "_threshold", 0.85)
    hits = await gateway.cache.search_top_n(query_vec, threshold, request.top_n)

    results: list[dict] = []
    for hit in hits:
        payload = await gateway.cache.get_payload(hit.key) or {}
        # Extract fields for CLI rendering; fall back to defaults.
        prompt = payload.get("cached_prompt", "")
        source = payload.get("source", "unknown")
        # Include the raw payload for completeness.
        result_entry = {
            "key": hit.key,
            "score": hit.score,
            "payload": payload,
            "prompt": prompt,
            "source": source,
        }
        results.append(result_entry)

    total_checked = len(results)
    return CacheInspectResponse(
        results=[
            CacheInspectResult(
                key=item["key"],
                score=item["score"],
                payload=item["payload"],
                prompt=item.get("prompt", ""),
                source=item.get("source", "unknown"),
                age_seconds=item.get("age_seconds", 0.0),
            )
            for item in results
        ],
        total_checked=total_checked,
    )