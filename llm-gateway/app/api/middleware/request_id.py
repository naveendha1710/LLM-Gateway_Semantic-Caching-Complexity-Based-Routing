"""Correlation ID middleware.

Generates (or propagates) a request ID for every inbound request and stores
it in a contextvar so structured logs can be correlated end-to-end.
"""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.observability.logging_config import request_id_var

HEADER_NAME = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Ensure every request has a correlation ID."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        # Honour incoming header, else generate
        rid = request.headers.get(HEADER_NAME) or str(uuid.uuid4())

        # Set contextvar for structured logging
        token = request_id_var.set(rid)
        try:
            response: Response = await call_next(request)
        finally:
            request_id_var.reset(token)

        response.headers[HEADER_NAME] = rid
        return response
