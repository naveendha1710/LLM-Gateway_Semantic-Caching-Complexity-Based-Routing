"""Cloud model provider — httpx-based, config-driven.

Talks to an OpenAI-compatible endpoint. This is the only "real" provider in
Phase 1; local and others arrive in later phases.
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

logger = logging.getLogger(__name__)


class CloudProvider:
    """OpenAI-compatible cloud provider."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "cloud"

    @property
    def cost_per_1k(self) -> float:
        # Placeholder — real pricing comes from config in a later phase.
        return 0.002

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=str(self._settings.cloud_provider_base_url),
                timeout=self._settings.cloud_provider_timeout_seconds,
                headers={
                    "Authorization": f"Bearer {self._settings.cloud_provider_api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def generate(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Send the request to the cloud endpoint and parse the response."""
        client = await self._get_client()
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        try:
            resp = await client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"Cloud provider timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Cloud provider request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"Cloud provider returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        return self._parse_response(data, request.model)

    @staticmethod
    def _parse_response(
        data: dict[str, Any], model: str
    ) -> ChatCompletionResponse:
        """Translate the cloud response into our internal schema."""
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

        return ChatCompletionResponse(
            id=data.get("id", ""),
            created=data.get("created", int(time.time())),
            model=data.get("model", model),
            choices=choices,
            usage=usage,
            source="cloud",
            degraded=False,
        )

    async def health(self) -> bool:
        """Lightweight health check — hit the models endpoint."""
        try:
            client = await self._get_client()
            resp = await client.get("/models")
            return resp.status_code < 400
        except Exception:
            return False

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
