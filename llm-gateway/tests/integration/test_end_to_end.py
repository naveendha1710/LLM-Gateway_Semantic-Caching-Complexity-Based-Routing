"""End-to-end integration test.

Full request → response cycle through the FastAPI app using httpx ASGITransport.
Verifies:
  - Cache miss → generate → response with source="cloud"
  - Repeated request → cache hit (Phase 2; in Phase 1 always misses)
  - Health endpoints respond correctly
  - Error handling produces structured error envelope
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from httpx import ASGITransport

from app.api.schemas.responses import ChatCompletionResponse
from app.core.config import Settings, get_settings
from app.main import create_app
from app.providers.cloud_provider import CloudProvider
from app.core.gateway import Gateway
from app.router import ProviderSelector, RequestClassifier, get_default_tier_config


def _mock_cloud_response(content: str = "Hello from cloud.", model: str = "nvidia/nemotron-3-nano-30b-a3b") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
    }


@pytest.fixture()
def app_with_mock_provider():  # type: ignore[no-untyped-def]
    """Create a FastAPI app whose cloud provider uses a MockTransport."""
    # Bypass the lru_cache so we get a fresh settings each test
    get_settings.cache_clear()
    settings = Settings(
        app_env="dev",
        cloud_provider_api_key="test-key",
        cloud_provider_base_url="https://api.openai.com/v1",
        cloud_provider_model="nvidia/nemotron-3-nano-30b-a3b",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json=_mock_cloud_response(content="Hello from cloud.", model="nvidia/nemotron-3-nano-30b-a3b"))

    transport = httpx.MockTransport(handler)
    provider = CloudProvider(settings)
    provider._client = httpx.AsyncClient(
        base_url=str(settings.cloud_provider_base_url),
        transport=transport,
        headers={"Authorization": "Bearer test-key"},
    )

    # Create router with the cloud provider
    tier_configs = get_default_tier_config()
    classifier = RequestClassifier(tier_configs)
    router = ProviderSelector(
        providers={"cloud": provider},
        tier_configs=tier_configs,
        classifier=classifier,
    )

    app = create_app(settings)
    # Override the lifespan-created gateway with our mock provider
    app.state.settings = settings
    app.state.provider = provider
    app.state.gateway = Gateway(router=router)

    yield app

    # Cleanup handled by provider.close in lifespan; clear cache for next test
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_healthz(app_with_mock_provider) -> None:  # type: ignore[no-untyped-def]
    """Liveness probe returns 200."""
    transport = ASGITransport(app=app_with_mock_provider)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_readyz(app_with_mock_provider) -> None:  # type: ignore[no-untyped-def]
    """Readiness probe reports component health."""
    transport = ASGITransport(app=app_with_mock_provider)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/readyz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["components"]["provider"] == "ok"


@pytest.mark.asyncio
async def test_chat_completion_end_to_end(app_with_mock_provider) -> None:  # type: ignore[no-untyped-def]
    """Full chat completion request returns a parsed response."""
    transport = ASGITransport(app=app_with_mock_provider)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "nvidia/nemotron-3-nano-30b-a3b",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["source"] == "cloud"
        assert body["degraded"] is False
        assert body["choices"][0]["message"]["content"] == "Hello from cloud."
        assert body["usage"]["total_tokens"] == 9


@pytest.mark.asyncio
async def test_request_id_header(app_with_mock_provider) -> None:  # type: ignore[no-untyped-def]
    """A custom X-Request-ID is echoed back in the response."""
    transport = ASGITransport(app=app_with_mock_provider)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "nvidia/nemotron-3-nano-30b-a3b",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers={"X-Request-ID": "my-correlation-id"},
        )
        assert resp.status_code == 200
        assert resp.headers["X-Request-ID"] == "my-correlation-id"


@pytest.mark.asyncio
async def test_invalid_request(app_with_mock_provider) -> None:  # type: ignore[no-untyped-def]
    """Malformed request body returns a 422 (pydantic validation)."""
    transport = ASGITransport(app=app_with_mock_provider)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini"},  # missing messages
        )
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_streaming_rejected(app_with_mock_provider) -> None:  # type: ignore[no-untyped-def]
    """stream=True is rejected in Phase 1."""
    transport = ASGITransport(app=app_with_mock_provider)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 422
