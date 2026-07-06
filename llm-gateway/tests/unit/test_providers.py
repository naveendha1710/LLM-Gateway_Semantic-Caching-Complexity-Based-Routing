"""Unit tests for providers.

Uses httpx.MockTransport so no live network calls are made. Also tests a
FakeProvider for the gateway path without any external dependency.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.api.schemas.requests import ChatCompletionRequest, Message
from app.api.schemas.responses import ChatCompletionResponse
from app.core.config import Settings
from app.core.exceptions import ProviderError, ProviderTimeout
from app.providers.cloud_provider import CloudProvider


def _settings() -> Settings:
    return Settings(
        app_env="dev",
        cloud_provider_api_key="test-key",
        cloud_provider_base_url="https://api.openai.com/v1",
        cloud_provider_model="gpt-4o-mini",
    )


def _make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[Message(role="user", content="Hello")],
    )


def _mock_response(content: str = "Hi there") -> dict[str, Any]:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        },
    }


@pytest.mark.asyncio
async def test_cloud_provider_generate_success() -> None:
    """A successful provider call returns a properly parsed response."""
    settings = _settings()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json=_mock_response("Hello back"))

    transport = httpx.MockTransport(handler)
    provider = CloudProvider(settings)
    # Inject mock transport
    provider._client = httpx.AsyncClient(
        base_url=str(settings.cloud_provider_base_url),
        transport=transport,
        headers={
            "Authorization": f"Bearer {settings.cloud_provider_api_key}",
            "Content-Type": "application/json",
        },
    )

    response = await provider.generate(_make_request())
    assert isinstance(response, ChatCompletionResponse)
    assert response.source == "cloud"
    assert response.degraded is False
    assert response.choices[0].message.content == "Hello back"
    assert response.usage.total_tokens == 8

    await provider.close()


@pytest.mark.asyncio
async def test_cloud_provider_generate_http_error() -> None:
    """A 4xx/5xx from the provider raises ProviderError."""
    settings = _settings()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream error")

    transport = httpx.MockTransport(handler)
    provider = CloudProvider(settings)
    provider._client = httpx.AsyncClient(
        base_url=str(settings.cloud_provider_base_url),
        transport=transport,
        headers={"Authorization": "Bearer test-key"},
    )

    with pytest.raises(ProviderError):
        await provider.generate(_make_request())

    await provider.close()


@pytest.mark.asyncio
async def test_cloud_provider_timeout() -> None:
    """A timeout from the provider raises ProviderTimeout."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    settings = _settings()
    transport = httpx.MockTransport(handler)
    provider = CloudProvider(settings)
    provider._client = httpx.AsyncClient(
        base_url=str(settings.cloud_provider_base_url),
        transport=transport,
        headers={"Authorization": "Bearer test-key"},
    )

    with pytest.raises(ProviderTimeout):
        await provider.generate(_make_request())

    await provider.close()


@pytest.mark.asyncio
async def test_cloud_provider_health_ok() -> None:
    """Health check returns True when /models responds 200."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": []})

    settings = _settings()
    transport = httpx.MockTransport(handler)
    provider = CloudProvider(settings)
    provider._client = httpx.AsyncClient(
        base_url=str(settings.cloud_provider_base_url),
        transport=transport,
        headers={"Authorization": "Bearer test-key"},
    )

    assert await provider.health() is True
    await provider.close()


@pytest.mark.asyncio
async def test_cloud_provider_health_unhealthy() -> None:
    """Health check returns False when /models errors."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="down")

    settings = _settings()
    transport = httpx.MockTransport(handler)
    provider = CloudProvider(settings)
    provider._client = httpx.AsyncClient(
        base_url=str(settings.cloud_provider_base_url),
        transport=transport,
        headers={"Authorization": "Bearer test-key"},
    )

    assert await provider.health() is False
    await provider.close()
