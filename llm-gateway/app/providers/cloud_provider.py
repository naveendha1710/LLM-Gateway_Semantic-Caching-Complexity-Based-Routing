"""Cloud model provider — httpx-based, config-driven.

Talks to an OpenAI-compatible endpoint. This is the only "real" provider in
Phase 1; local and others arrive in later phases.
"""

from __future__ import annotations

import logging
import time
import os
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


class CloudProvider:
    """OpenAI-compatible cloud provider."""

    def __init__(
        self,
        settings: Settings,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        provider_name: str = "cloud",
    ) -> None:
        self._settings = settings
        self._base_url = base_url
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._provider_name = provider_name
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def cost_per_1k(self) -> float:
        # Placeholder — real pricing comes from config in a later phase.
        return 0.002

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            # Provider-specific config MUST be provided by factory; no legacy fallback.
            # If base_url/api_key are missing, fail fast with a clear error.
            if not self._base_url:
                raise ProviderError(
                    f"CloudProvider '{self._provider_name}': base_url is required but not configured",
                    url=None,
                    request_id=request_id_var.get(),
                )
            if not self._api_key:
                raise ProviderError(
                    f"CloudProvider '{self._provider_name}': api_key is required but not configured "
                    f"(check api_key_env in provider entry)",
                    url=None,
                    request_id=request_id_var.get(),
                )
            timeout = self._timeout_seconds or self._settings.cloud_provider_timeout_seconds
            
            # Explicitly disable environment proxy settings to avoid unintended proxy usage.
            # This mirrors the behavior of the successful PowerShell request which does not rely on proxy env vars.
            # Determine proxy configuration from environment variables if present.
            # httpx respects the ``proxies`` argument; we pull the HTTPS proxy first,
            # falling back to HTTP. If neither is set, ``proxies`` is left as ``None``.
            proxy_url = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
            # If no proxy is defined we pass ``None`` – httpx treats ``None`` as “no proxy”.
            proxy_arg = proxy_url if proxy_url else None
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                # Enable proxy handling only when a proxy URL is provided.
                trust_env=bool(proxy_arg),
                proxy=proxy_arg,
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
        # NVIDIA-specific extensions to request reasoning separately.
        # Enable chain‑of‑thought generation and set a reasoning budget.
        # Previously the budget was set equal to ``max_tokens`` which left no room for the
        # actual answer content, causing the "reasoning‑leak" where the final answer was
        # missing and the reasoning text was returned as the response content.
        #
        # The budget should be a *fraction* of the total token allowance. We now default
        # to half of ``max_tokens`` (rounded down) when ``max_tokens`` is provided, and keep
        # the historic default of 1024 tokens when it is not.
        if request.max_tokens is not None:
            reasoning_budget = max(request.max_tokens // 2, 1)
        else:
            reasoning_budget = 1024
        payload["chat_template_kwargs"] = {"enable_thinking": True}
        payload["reasoning_budget"] = reasoning_budget

        try:
            resp = await client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"Cloud provider timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            # Include request URL if available for debugging
            request_url = getattr(exc, "request", None)
            url_str = str(request_url.url) if request_url and hasattr(request_url, "url") else None
            raise ProviderError(
                f"Cloud provider request failed: {exc}",
                url=url_str,
                request_id=request_id_var.get(),
            ) from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"Cloud provider returned HTTP {resp.status_code}: {resp.text[:200]}",
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
        # Extract optional reasoning content from the first choice, if present.
        reasoning: str | None = None
        if data.get("choices"):
            first_msg = data["choices"][0].get("message", {})
            # Some providers may use "reasoning" or "reasoning_content".
            reasoning = first_msg.get("reasoning") or first_msg.get("reasoning_content")

        return ChatCompletionResponse(
            id=data.get("id", ""),
            created=data.get("created", int(time.time())),
            model=data.get("model", model),
            choices=choices,
            usage=usage,
            source="cloud",
            degraded=False,
            reasoning=reasoning,
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
