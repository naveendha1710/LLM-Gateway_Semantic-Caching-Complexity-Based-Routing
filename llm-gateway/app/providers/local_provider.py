"""Local model provider — httpx-based, config-driven.

Talks to an OpenAI-compatible local endpoint (Ollama, llama.cpp, vLLM, etc.).
This provider is used for cost-effective local inference in Phase 3.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.api.schemas.requests import ChatCompletionRequest
from app.api.schemas.responses import (
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Usage,
)
from app.core.config import Settings
from app.core.exceptions import ProviderError, ProviderTimeout
from app.observability.logging_config import request_id_var

logger = logging.getLogger(__name__)


class LocalProvider:
    """OpenAI-compatible local provider (Ollama, llama.cpp, vLLM, etc.)."""

    def __init__(
        self,
        settings: Settings,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        provider_name: str = "local",
    ) -> None:
        self._settings = settings
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._provider_name = provider_name
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def cost_per_1k(self) -> float:
        # Local inference is effectively free (just compute cost)
        return 0.0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            # Provider-specific config MUST be provided by factory; no legacy fallback.
            if not self._base_url:
                raise ProviderError(
                    f"LocalProvider '{self._provider_name}': base_url is required but not configured",
                    url=None,
                    request_id=request_id_var.get(),
                )
            # Ensure base_url doesn't end with /v1 if it's already included
            base_url = self._base_url
            if base_url.endswith("/v1"):
                base_url = base_url[:-3]
            timeout = self._timeout_seconds or self._settings.local_provider_timeout_seconds
            self._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=timeout,
                headers={
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def generate(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Send the request to the local endpoint and parse the response."""
        client = await self._get_client()
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        try:
            resp = await client.post("/v1/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"Local provider timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            request_url = getattr(exc, "request", None)
            url_str = str(request_url.url) if request_url and hasattr(request_url, "url") else None
            raise ProviderError(
                f"Local provider request failed: {exc}",
                url=url_str,
                request_id=request_id_var.get(),
            ) from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"Local provider returned HTTP {resp.status_code}: {resp.text[:200]}",
                url=str(resp.request.url) if resp.request else None,
                response_status=resp.status_code,
                response_body=resp.text[:500],
                request_id=resp.headers.get("x-request-id") or resp.headers.get("request-id"),
            )

        data = resp.json()
        return self._parse_response(data, request.model)

    @staticmethod
    def _parse_response(
        data: dict[str, Any], model: str
    ) -> ChatCompletionResponse:
        """Translate the local response into our internal schema."""
        choices = []
        for i, ch in enumerate(data.get("choices", [])):
            msg = ch.get("message", {})
            choices.append(
                Choice(
                    index=ch.get("index", i),
                    message=ChoiceMessage(
                        role=msg.get("role", "assistant"),
                        content=msg.get("content", ""),
                    ),
                    finish_reason=ch.get("finish_reason", "stop"),
                )
            )

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        # Extract optional reasoning content from the first choice, if present.
        reasoning: str | None = None
        if data.get("choices"):
            first_msg = data["choices"][0].get("message", {})
            reasoning = first_msg.get("reasoning") or first_msg.get("reasoning_content")

        return ChatCompletionResponse(
            id=data.get("id", ""),
            created=data.get("created", int(time.time())),
            model=data.get("model", model),
            choices=choices,
            usage=usage,
            source="local",
            degraded=False,
            reasoning=reasoning,
        )

    async def health(self) -> bool:
        """Lightweight health check — hit the models endpoint."""
        try:
            client = await self._get_client()
            resp = await client.get("/v1/models")
            return resp.status_code < 400
        except Exception:
            return False

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()