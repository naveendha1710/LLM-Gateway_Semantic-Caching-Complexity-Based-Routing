"""Retry logic with exponential backoff for provider calls."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from app.core.exceptions import ProviderUnavailable

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 0.5  # seconds
    max_delay: float = 10.0  # seconds
    exponential_base: float = 2.0
    jitter: float = 0.1  # fraction of delay to add as jitter
    retryable_exceptions: tuple[type[Exception], ...] = (
        ProviderUnavailable,
        TimeoutError,
        ConnectionError,
    )


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay with exponential backoff and jitter."""
    delay = min(
        config.base_delay * (config.exponential_base ** attempt),
        config.max_delay,
    )
    jitter_range = delay * config.jitter
    return delay + random.uniform(-jitter_range, jitter_range)


async def retry_async(
    func: Callable[[], Awaitable[T]],
    config: RetryConfig | None = None,
    on_retry: Callable[[Exception, int], Awaitable[None] | None] | None = None,
) -> T:
    """Execute an async function with retry logic.

    Args:
        func: Async function to execute
        config: Retry configuration (uses defaults if None)
        on_retry: Optional callback called before each retry (exception, attempt_number)

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries exhausted
    """
    if config is None:
        config = RetryConfig()

    last_exception: Exception | None = None

    for attempt in range(config.max_attempts):
        try:
            return await func()
        except config.retryable_exceptions as e:
            last_exception = e
            if attempt < config.max_attempts - 1:
                delay = calculate_delay(attempt, config)
                logger.warning(
                    "Retry attempt %d/%d after %.2fs: %s",
                    attempt + 1,
                    config.max_attempts,
                    delay,
                    e,
                )
                if on_retry:
                    await on_retry(e, attempt + 1)
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "All %d retry attempts exhausted", config.max_attempts
                )

    raise last_exception  # type: ignore[misc]


def is_retryable_error(exception: Exception, config: RetryConfig | None = None) -> bool:
    """Check if an exception is retryable based on config."""
    if config is None:
        config = RetryConfig()
    return isinstance(exception, config.retryable_exceptions)