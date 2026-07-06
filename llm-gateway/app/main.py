"""FastAPI application factory.

Wires together middleware, routes, and the gateway orchestrator. Lifecycle
hooks start/stop the provider client and (in later phases) the cache backend.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.api.middleware.error_handler import ErrorHandlerMiddleware
from app.api.middleware.request_id import RequestIDMiddleware
from app.api.routes import chat, health
from app.core.config import Settings, get_settings
from app.core.gateway import Gateway
from app.observability.logging_config import configure_logging
from app.providers.cloud_provider import CloudProvider

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown lifecycle."""
    settings: Settings = get_settings()
    configure_logging(settings.app_log_level)

    provider = CloudProvider(settings)
    gateway = Gateway(provider=provider)

    app.state.settings = settings
    app.state.provider = provider
    app.state.gateway = gateway

    logger.info("gateway_starting", extra={"env": settings.app_env})

    yield

    # Shutdown
    await provider.close()
    logger.info("gateway_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the FastAPI application."""
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Middleware (order: outermost first → request_id, then error_handler)
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # Routes
    app.include_router(health.router, tags=["health"])
    app.include_router(chat.router, tags=["chat"])

    return app


# Module-level app for `uvicorn app.main:app`
app = create_app()
