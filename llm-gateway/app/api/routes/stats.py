"""Stats API routes for gateway statistics."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.schemas.responses import GatewayStatsResponse

router = APIRouter(prefix="/v1", tags=["stats"])


@router.get("/stats", response_model=GatewayStatsResponse)
async def get_stats(request: Request) -> GatewayStatsResponse:
    """Get gateway statistics including request counts, cache hit rates, and latency percentiles."""
    from app.core.gateway import Gateway

    app_state = request.app.state
    gateway: Gateway | None = getattr(app_state, "gateway", None)

    if gateway is None:
        return GatewayStatsResponse()

    stats = await gateway.get_stats()
    return GatewayStatsResponse(**stats)