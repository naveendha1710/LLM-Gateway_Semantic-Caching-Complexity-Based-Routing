"""Typed application settings via pydantic-settings.

Loads from environment variables and/or a YAML config file. Fails fast on
invalid configuration at startup — never at request time.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed settings for the entire gateway."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────
    app_name: str = "llm-gateway"
    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "INFO"
    app_request_timeout_seconds: float = 30.0

    # ── Cloud provider ──────────────────────────────────────────
    cloud_provider_api_key: str = ""
    cloud_provider_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    cloud_provider_model: str = "gemini-1.5-flash"
    cloud_provider_timeout_seconds: float = 15.0
    cloud_provider_max_retries: int = 3

    # ── Cache (Phase 2 — stubbed for Phase 1) ────────────────────
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.85
    cache_ttl_seconds: int = 86400

    # ── Validation ───────────────────────────────────────────────
    @field_validator("app_env")
    @classmethod
    def _validate_env(cls, v: str) -> str:
        allowed = {"dev", "staging", "prod"}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(
                f"app_env must be one of {allowed}, got '{v}'"
            )
        return v_lower

    @field_validator("app_log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(
                f"app_log_level must be one of {allowed}, got '{v}'"
            )
        return v_upper

    @field_validator("app_port")
    @classmethod
    def _validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"app_port must be 1-65535, got {v}")
        return v

    @field_validator("app_request_timeout_seconds")
    @classmethod
    def _validate_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("app_request_timeout_seconds must be positive")
        return v

    @field_validator("cloud_provider_timeout_seconds")
    @classmethod
    def _validate_cloud_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("cloud_provider_timeout_seconds must be positive")
        return v

    @field_validator("cloud_provider_max_retries")
    @classmethod
    def _validate_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("cloud_provider_max_retries must be >= 0")
        return v

    @field_validator("cache_similarity_threshold")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                "cache_similarity_threshold must be between 0.0 and 1.0"
            )
        return v

    @model_validator(mode="after")
    def _validate_prod_requires_api_key(self) -> "Settings":
        """In prod, a cloud API key is mandatory."""
        if self.app_env == "prod" and not self.cloud_provider_api_key:
            raise ValueError(
                "cloud_provider_api_key is required when app_env='prod'"
            )
        return self

    # ── YAML overlay ─────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from a YAML file, then overlay env vars on top.

        Env vars take precedence over YAML values, matching 12-factor
        principles.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            yaml_data: dict[str, Any] = yaml.safe_load(f) or {}

        # Flatten nested YAML into the flat key names pydantic-settings expects
        flat: dict[str, Any] = {}
        for key, value in yaml_data.items():
            flat[key.lower()] = value

        # Env vars override YAML
        for key in flat:
            env_val = os.environ.get(key.upper())
            if env_val is not None:
                flat[key] = env_val

        return cls(**flat)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton.

    In tests, call `get_settings.cache_clear()` after setting env vars.
    """
    return Settings()
