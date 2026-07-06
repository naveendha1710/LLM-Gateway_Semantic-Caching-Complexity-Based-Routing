"""Request schemas — OpenAI-compatible chat completions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str = Field(..., min_length=1, description="Message text")


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions body — mirrors OpenAI's schema."""

    model: str = Field(..., description="Model or tier alias to use")
    messages: list[Message] = Field(
        ..., min_length=1, description="Conversation turns"
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=1, le=8192)
    stream: bool = False  # Phase 1: non-streaming only

    # Gateway-specific extension
    bypass_cache: bool = Field(
        False, description="Skip cache lookup (still writes on miss)"
    )

    @field_validator("stream")
    @classmethod
    def _reject_streaming(cls, v: bool) -> bool:
        if v:
            raise ValueError("streaming is not yet supported")
        return v
