"""Custom exception hierarchy for the gateway.

Each exception maps to a specific failure mode documented in the project
structure. The error_handler middleware translates these into clean JSON
responses with appropriate HTTP status codes.
"""

from __future__ import annotations


class GatewayError(Exception):
    """Base exception for all gateway-specific errors."""

    status_code: int = 500
    error_code: str = "gateway_error"

    def __init__(self, message: str = "", *, status_code: int | None = None):
        super().__init__(message)
        self.message = message or self.__class__.__doc__ or "Gateway error"
        if status_code is not None:
            self.status_code = status_code


# ── Cache failures (fail-open) ──────────────────────────────────
class CacheUnavailable(GatewayError):
    """The cache subsystem is unreachable or degraded."""

    status_code = 200  # fail-open: request still succeeds
    error_code = "cache_unavailable"


# ── Provider failures (fail-closed) ─────────────────────────────
class ProviderError(GatewayError):
    """A single provider returned an error."""

    status_code = 502
    error_code = "provider_error"


class ProviderTimeout(ProviderError):
    """A provider call exceeded its timeout."""

    error_code = "provider_timeout"


class AllProvidersFailed(GatewayError):
    """Every provider in the tier was attempted and failed."""

    status_code = 503
    error_code = "all_providers_failed"


# ── Configuration failures (fail at startup) ───────────────────
class InvalidTierConfig(GatewayError):
    """tiers.yaml is missing, malformed, or semantically invalid."""

    status_code = 500
    error_code = "invalid_tier_config"


class InvalidConfigError(GatewayError):
    """Settings failed validation at startup."""

    status_code = 500
    error_code = "invalid_config"


# ── Request failures ────────────────────────────────────────────
class RequestTimeout(GatewayError):
    """The overall request exceeded the global timeout."""

    status_code = 504
    error_code = "request_timeout"


class InvalidRequest(GatewayError):
    """The client sent a malformed or invalid request."""

    status_code = 400
    error_code = "invalid_request"
