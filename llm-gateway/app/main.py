"""FastAPI application factory.

Wires together middleware, routes, and the gateway orchestrator. Lifecycle
hooks start/stop the provider client and the cache backend.

Cache wiring (Phase 2):
* An :class:`Embedder` (default ``HashEmbedder``) turns requests into vectors.
* A :class:`VectorStore` persists embeddings + payloads to Redis (or
  ``fakeredis`` when ``app_env`` is ``test``).
* The :class:`Gateway` receives both and uses them for fail-open caching.

Router wiring (Phase 3):
* A :class:`ProviderSelector` routes requests to providers with failover.
* :class:`RequestClassifier` determines routing strategy.
* Tier configs define provider priority and fallback chains.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.api.middleware.error_handler import ErrorHandlerMiddleware
from app.api.middleware.request_id import RequestIDMiddleware
from app.api.routes import chat, health, stats, cache_inspect
from app.cache.embedder import Embedder, get_embedder
from app.cache.vector_store import VectorStore
from app.core.config import Settings, get_settings
from app.core.gateway import Gateway
from app.observability.logging_config import configure_logging
from app.providers.base import ModelProvider
from app.providers.cloud_provider import CloudProvider
from app.providers.local_provider import LocalProvider
from app.router.selector import ProviderSelector
from app.router.classifier import RequestClassifier
from app.router.tier_config import build_tier_configs_from_settings, ProviderTierConfig

logger = logging.getLogger(__name__)


def _validate_api_key_env_vars(settings: Settings) -> None:
    """Validate that all api_key_env references resolve to actual environment variables.

    Fails fast at startup if any required API key environment variable is missing.
    """
    if not settings.models:
        # Legacy flat config - validation handled by existing _validate_prod_requires_api_key
        return

    missing_vars: list[str] = []
    for tier_name, tier_config in settings.models.items():
        if not tier_config.enabled:
            continue

        # The TierModelsConfig does not have a tier‑level ``api_key_env`` attribute.
        # All API key environment variable references are stored on the individual
        # provider entries. Iterate over ``providers`` and validate each one.
        for provider_entry in tier_config.providers:
            if not provider_entry.enabled:
                continue
            if provider_entry.api_key_env and not os.environ.get(provider_entry.api_key_env):
                missing_vars.append(
                    f"tier '{tier_name}' provider '{provider_entry.name}': {provider_entry.api_key_env}"
                )

    if missing_vars:
        raise RuntimeError(
            "Missing required API key environment variables at startup:\n  - "
            + "\n  - ".join(missing_vars)
            + "\nSet these environment variables before starting the gateway."
        )


def _build_redis_client(settings: Settings):  # type: ignore[no-untyped-def]
    """Create a Redis client (real or fake) based on the environment."""
    if settings.app_env == "test":
        import fakeredis.aioredis

        return fakeredis.aioredis.FakeRedis()
    import redis.asyncio as redis_async

    return redis_async.from_url(settings.cache_redis_url, decode_responses=False)


def _create_providers_from_tier_config(settings: Settings, tier_configs) -> dict[str, ModelProvider]:
    """Create provider instances from tier configuration.
    
    Each provider entry in the tier config gets its own provider instance
    with the correct base_url, api_key, and timeout.
    """
    from app.providers.base import ModelProvider
    
    providers: dict[str, ModelProvider] = {}
    
    for tier_config in tier_configs:
        if not tier_config.enabled:
            continue
            
        for provider_entry in tier_config.providers:
            if not provider_entry.enabled:
                continue
                
            # Resolve API key from environment variable if specified
            api_key = None
            if provider_entry.api_key_env:
                api_key = os.environ.get(provider_entry.api_key_env)
                if not api_key:
                    logger.warning(
                        "API key environment variable not set",
                        extra={"provider": provider_entry.name, "env_var": provider_entry.api_key_env},
                    )
            
            # Create provider instance based on provider_kind
            if provider_entry.provider_kind == "cloud":
                # Use provider-specific timeout or fall back to legacy settings
                timeout = provider_entry.timeout_seconds or settings.cloud_provider_timeout_seconds
                provider = CloudProvider(
                    settings,
                    base_url=provider_entry.base_url or None,
                    api_key=api_key,
                    timeout_seconds=timeout,
                    provider_name=provider_entry.name,
                )
            elif provider_entry.provider_kind == "local":
                # Use provider-specific timeout or fall back to legacy settings
                timeout = provider_entry.timeout_seconds or settings.local_provider_timeout_seconds
                provider = LocalProvider(
                    settings,
                    base_url=provider_entry.base_url or None,
                    timeout_seconds=timeout,
                    provider_name=provider_entry.name,
                )
            else:
                logger.warning(
                    "Unknown provider kind, skipping",
                    extra={"provider": provider_entry.name, "kind": provider_entry.provider_kind},
                )
                continue
            
            providers[provider_entry.name] = provider
            logger.info(
                "Registered provider",
                extra={
                    "provider": provider_entry.name,
                    "kind": provider_entry.provider_kind,
                    "model": provider_entry.model,
                    "base_url": provider_entry.base_url,
                },
            )
    
    return providers


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown lifecycle."""
    settings: Settings = get_settings()
    configure_logging(settings.app_log_level)

    # Validate API key environment variables at startup (fail-fast)
    _validate_api_key_env_vars(settings)

    # ── Providers (Phase 3) ────────────────────────────────────
    # Build tier configs first to get provider definitions
    tier_configs = build_tier_configs_from_settings(settings)
    
    # Create providers from tier config
    providers = _create_providers_from_tier_config(settings, tier_configs)

    # ── Router (Phase 3) ───────────────────────────────────────
    classifier = RequestClassifier(tier_configs)
    router = ProviderSelector(
        providers=providers,
        tier_configs=tier_configs,
        classifier=classifier,
    )

    # ── Cache wiring (Phase 2) ───────────────────────────────
    embedder: Embedder | None = None
    cache: VectorStore | None = None
    if settings.cache_enabled:
        embedder = get_embedder(
            kind=settings.cache_embedder_kind,
            dimension=settings.cache_embedder_dimension,
        )
        redis_client = _build_redis_client(settings)
        cache = VectorStore(
            redis=redis_client,
            embedder_dimension=embedder.dimension,
            ttl_seconds=settings.cache_ttl_seconds,
        )
        await cache.initialize()

    gateway = Gateway(
        router=router,
        cache=cache,
        embedder=embedder,
        similarity_threshold=settings.cache_similarity_threshold,
    )

    app.state.settings = settings
    app.state.providers = providers
    app.state.router = router
    app.state.gateway = gateway
    app.state.cache = cache

    logger.info(
        "gateway_starting",
        extra={"env": settings.app_env, "cache_enabled": settings.cache_enabled},
    )

    yield

    # Shutdown
    if cache is not None:
        await cache.close()
    for provider in providers.values():
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
    app.include_router(stats.router, tags=["stats"])
    app.include_router(cache_inspect.router, tags=["cache"])

    return app


# Module-level app for `uvicorn app.main:app`
app = create_app()
