"""Response schemas — OpenAI-compatible with gateway extensions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: Literal["stop", "length"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-style response enriched with gateway metadata."""

    id: str = Field(..., description="Unique response ID")
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)

    # ── Gateway extensions ───────────────────────────────────────
    source: Literal["cache", "local", "cloud"] = Field(
        ..., description="Where the answer came from"
    )
    degraded: bool = Field(
        False, description="True if served under degraded conditions"
    )
    # Optional separate reasoning content provided by the model.
    reasoning: str | None = Field(
        None,
        description="Model‑generated reasoning or chain‑of‑thought, if any",
    )
    # The tier or routing strategy that was selected for this request.
    # Populated by the ProviderSelector based on the tier configuration.
    routing_strategy: str | None = Field(
        None,
        description="Name of the tier (e.g., 'simple', 'complex') used for routing",
    )


class GatewayStatsResponse(BaseModel):
    """Gateway statistics response."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    local_requests: int = 0
    cloud_requests: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    errors: int = 0


class ErrorResponse(BaseModel):
    """Standard error envelope returned for all failures."""

    error: dict  # {"code": str, "message": str}
    request_id: str | None = None
