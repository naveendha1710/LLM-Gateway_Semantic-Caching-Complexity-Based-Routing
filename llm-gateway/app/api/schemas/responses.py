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


class ErrorResponse(BaseModel):
    """Standard error envelope returned for all failures."""

    error: dict  # {"code": str, "message": str}
    request_id: str | None = None
