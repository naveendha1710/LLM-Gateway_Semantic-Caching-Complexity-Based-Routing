"""Structured JSON logging with correlation ID support.

Uses python-json-logger to emit one JSON object per log line, making logs
easy to ingest into ELK / Loki / CloudWatch. A correlation ID (request_id)
is propagated via a contextvar so every log line within a request is linked.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

from pythonjsonlogger.json import JsonFormatter

# Contextvar set by the request_id middleware
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class CorrelationFilter(logging.Filter):
    """Inject the current request_id into every log record."""

    def __init__(self) -> None:
        super().__init__(name="correlation")

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        return True


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger to emit structured JSON to stdout.

    Idempotent: calling multiple times will not duplicate handlers.
    """
    root = logging.getLogger()
    # Remove any existing handlers (idempotent re-config)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(CorrelationFilter())

    formatter = JsonFormatter(
        json.dumps  # use stdlib json for serialization
    )
    # Custom field mapping for readability
    formatter = JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(request_id)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; configuration is global."""
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    """Emit a structured log event with arbitrary key-value context."""
    logger.log(level, event, extra={"event": event, **fields})
