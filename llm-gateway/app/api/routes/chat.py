"""Chat completions route — POST /v1/chat/completions."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from app.api.schemas.requests import ChatCompletionRequest
from app.api.schemas.responses import ChatCompletionResponse
from app.core.exceptions import GatewayError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest, request: Request
) -> ChatCompletionResponse:
    """Handle a chat completion request through the gateway."""
    gateway = request.app.state.gateway
    response = await gateway.handle(payload)
    logger.info(
        "chat_completion_served",
        extra={
            "source": response.source,
            "degraded": response.degraded,
            "model": response.model,
        },
    )
    return response
