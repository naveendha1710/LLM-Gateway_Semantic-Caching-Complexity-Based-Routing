"""Global error handler middleware.

Catches all unhandled exceptions and translates them into clean JSON error
responses using the exception hierarchy in app.core.exceptions. This ensures
clients always receive a consistent error envelope.
"""

from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.exceptions import GatewayError
from app.observability.logging_config import request_id_var

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Convert unhandled exceptions into structured JSON errors."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        try:
            return await call_next(request)
        except GatewayError as exc:
            return self._build_error_response(exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Unhandled exception",
                extra={"event": "unhandled_exception", "error": str(exc)},
            )
            # Wrap unknown errors in a generic 500
            return self._build_error_response(
                GatewayError("Internal server error", status_code=500)
            )

    @staticmethod
    def _build_error_response(exc: GatewayError) -> Response:
        body = {
            "error": {
                "code": exc.error_code,
                "message": exc.message,
            },
            "request_id": request_id_var.get(),
        }
        return JSONResponse(
            status_code=exc.status_code,
            content=body,
            headers={"X-Request-ID": request_id_var.get() or ""},
        )
