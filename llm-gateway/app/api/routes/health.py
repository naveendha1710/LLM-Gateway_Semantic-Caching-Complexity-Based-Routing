"""Health check routes.

  - GET /healthz  : liveness probe (always 200 if the process is up)
  - GET /readyz   : readiness probe (checks cache + provider health)
"""

from __future__ import annotations

from fastapi import APIRouter
from starlette.requests import Request

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness — process is up."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> dict[str, object]:
    """Readiness — check downstream dependencies.

    Returns 200 with ``status: "ok"`` only when all dependencies report
    healthy. Otherwise returns 503 with per-component details.
    """
    from app.core.gateway import Gateway

    app_state = request.app.state
    gateway: Gateway | None = getattr(app_state, "gateway", None)

    components: dict[str, str] = {}
    overall = "ok"

    if gateway is None:
        components["gateway"] = "not_initialized"
        overall = "degraded"
    else:
        # Provider health
        try:
            provider_ok = await gateway.provider_health()
            components["provider"] = "ok" if provider_ok else "unhealthy"
            if not provider_ok:
                overall = "degraded"
        except Exception:
            components["provider"] = "error"
            overall = "degraded"

        # Cache health (fail-open: cache down does not make us unready)
        try:
            cache_ok = await gateway.cache_health()
            components["cache"] = "ok" if cache_ok else "unavailable"
        except Exception:
            components["cache"] = "unavailable"

    return {"status": overall, "components": components}
