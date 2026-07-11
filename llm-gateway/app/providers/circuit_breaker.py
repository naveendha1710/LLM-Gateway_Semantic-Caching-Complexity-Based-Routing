"""Circuit breaker pattern for provider resilience."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar

from app.core.exceptions import ProviderUnavailable

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open before closing
    timeout: float = 30.0  # Seconds before half-open
    excluded_exceptions: tuple[type[Exception], ...] = ()  # Don't count these as failures


@dataclass
class CircuitBreakerStats:
    """Runtime statistics for a circuit breaker."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreaker:
    """Circuit breaker implementation for provider calls.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests blocked immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        self._last_state_change = time.monotonic()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for timeout transition."""
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_state_change >= self.config.timeout:
                # Transition to half-open after timeout
                return CircuitState.HALF_OPEN
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get current statistics."""
        return self._stats

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            self._last_state_change = time.monotonic()
            self._stats.state_changes += 1
            logger.info(
                "Circuit breaker '%s': %s -> %s",
                self.name,
                old_state.value,
                new_state.value,
            )

    def _record_success(self) -> None:
        """Record a successful call."""
        self._stats.total_calls += 1
        self._stats.successful_calls += 1
        self._stats.consecutive_failures = 0
        self._stats.consecutive_successes += 1
        self._stats.last_success_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            if self._stats.consecutive_successes >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
                self._stats.consecutive_successes = 0

    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        # Check if this exception should be excluded
        if isinstance(exception, self.config.excluded_exceptions):
            return

        self._stats.total_calls += 1
        self._stats.failed_calls += 1
        self._stats.consecutive_successes = 0
        self._stats.consecutive_failures += 1
        self._stats.last_failure_time = time.monotonic()

        if self._state == CircuitState.CLOSED:
            if self._stats.consecutive_failures >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self._transition_to(CircuitState.OPEN)

    def _record_rejected(self) -> None:
        """Record a rejected call (circuit open)."""
        self._stats.rejected_calls += 1

    async def call(
        self,
        func: Callable[[], Awaitable[T]],
    ) -> T:
        """Execute a function with circuit breaker protection.

        Args:
            func: Async function to execute

        Returns:
            Result of the function call

        Raises:
            ProviderUnavailable: If circuit is open
            Exception: Any exception from the function
        """
        async with self._lock:
            current_state = self.state

            if current_state == CircuitState.OPEN:
                self._record_rejected()
                raise ProviderUnavailable(
                    f"Circuit breaker '{self.name}' is OPEN",
                    provider=self.name,
                )

        try:
            result = await func()
            async with self._lock:
                self._record_success()
            return result
        except Exception as e:
            async with self._lock:
                self._record_failure(e)
            raise

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self._transition_to(CircuitState.CLOSED)
        self._stats.consecutive_failures = 0
        self._stats.consecutive_successes = 0

    def force_open(self) -> None:
        """Manually force the circuit breaker open."""
        self._transition_to(CircuitState.OPEN)

    def force_closed(self) -> None:
        """Manually force the circuit breaker closed."""
        self._transition_to(CircuitState.CLOSED)
        self._stats.consecutive_failures = 0
        self._stats.consecutive_successes = 0


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    # Allow dict‑style access in tests and legacy code
    def __getitem__(self, name: str) -> CircuitBreaker | None:  # pragma: no cover
        """Retrieve a circuit breaker by name using ``registry[name]`` syntax.

        This mirrors the behaviour of a plain ``dict`` for backward
        compatibility with existing tests that expect the registry to be
        subscriptable. It returns ``None`` if the breaker does not exist.
        """
        return self._breakers.get(name)

    async def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    async def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name."""
        async with self._lock:
            return self._breakers.get(name)

    async def get_all_stats(self) -> dict[str, CircuitBreakerStats]:
        """Get stats for all circuit breakers."""
        async with self._lock:
            return {name: cb.stats for name, cb in self._breakers.items()}

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        async with self._lock:
            for cb in self._breakers.values():
                cb.reset()