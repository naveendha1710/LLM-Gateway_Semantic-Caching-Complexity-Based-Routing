"""Typed application settings via pydantic-settings.

Loads from environment variables and/or a YAML config file. Fails fast on
invalid configuration at startup — never at request time.
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderEntry(BaseModel):
    """Configuration for a single provider within a tier."""

    name: str = Field(..., description="Unique identifier for this provider entry")
    provider: str = Field(..., description="Provider type (e.g., 'nvidia', 'openai', 'anthropic', 'ollama')")
    provider_kind: Literal["local", "cloud"] = Field(..., description="Provider kind: local or cloud")
    base_url: str = Field(default="", description="API base URL for this provider (optional for tests)")
    api_key_env: str | None = Field(default=None, description="Environment variable name for API key (never store raw keys in YAML)")
    model: str = Field(..., description="Model identifier (e.g., 'nvidia/nemotron-3-nano-30b-a3b')")
    enabled: bool = Field(default=True, description="Whether this provider/model is available for routing")
    priority: int = Field(default=1, description="Lower = higher priority (tried first within tier)")
    max_tokens: int | None = Field(default=None, description="Max tokens for this model")
    cost_per_1k: float | None = Field(default=None, description="Cost per 1k tokens (used by cost_optimized strategy)")
    params: dict[str, Any] = Field(default_factory=dict, description="Generation parameters (temperature, top_p, etc.)")
    timeout_seconds: float | None = Field(default=None, description="Request timeout in seconds (overrides legacy settings)")


class TierModelsConfig(BaseModel):
    """Configuration for a complexity-based tier with multiple providers."""

    routing_strategy: Literal["cost_optimized", "quality_optimized", "latency_optimized", "balanced"] = Field(
        default="balanced", description="Routing strategy for this tier"
    )
    enabled: bool = Field(default=True, description="Whether this entire tier is enabled")
    fallback_tier: str | None = Field(default=None, description="Tier name to fall back to if ALL providers in this tier fail")
    providers: list[ProviderEntry] = Field(default_factory=list, description="List of providers in this tier (can mix local and cloud)")


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
    cloud_provider_models: list[str] = ["gemini-1.5-flash", "gemini-1.5-pro"]
    cloud_provider_timeout_seconds: float = 15.0
    cloud_provider_max_retries: int = 3

    # ── Local provider (Phase 3 — local models) ─────────────────
    local_provider_base_url: str = "http://localhost:11434"
    local_provider_model: str = "llama3.2"
    local_provider_models: list[str] = ["llama3.2", "llama3.1"]
    local_provider_timeout_seconds: float = 60.0

    # ── Cache (Phase 2 — semantic cache) ─────────────────────────
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.85
    cache_ttl_seconds: int = 86400
    cache_embedder_kind: str = "hash"
    cache_embedder_dimension: int = 256
    cache_redis_url: str = "redis://localhost:6379/0"

    # ── Multi-model configuration (new schema) ───────────────────
    models: dict[str, TierModelsConfig] = Field(
        default_factory=dict,
        description="Complexity-based tier configurations with mixed providers"
    )

    # ── Backward compatibility fields ────────────────────────
    # The original flat‑config fields are retained as regular list attributes so
    # tests can assign to them directly. The newer nested ``models`` schema is
    # used when present; otherwise these fields provide the legacy defaults.
    _local_provider_models_legacy: list[str] = ["llama3.2", "llama3.1"]
    _cloud_provider_models_legacy: list[str] = ["nvidia/nemotron-3-nano-30b-a3b"]

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
        """In prod, ensure every enabled cloud provider entry has its API key env var set.

        The legacy ``cloud_provider_api_key`` field is deprecated; production
        deployments must provide an environment variable for each cloud provider
        defined in ``models``. If a provider entry is enabled and its
        ``api_key_env`` is defined, we verify that ``os.getenv`` returns a
        non‑empty value. Missing keys raise a clear error indicating which
        provider is mis‑configured.
        """
        if self.app_env != "prod":
            return self

        # Iterate over all tiers and providers in the new schema.
        for tier_name, tier_cfg in self.models.items():
            if not tier_cfg.enabled:
                continue
            for provider in tier_cfg.providers:
                if provider.provider_kind != "cloud" or not provider.enabled:
                    continue
                env_name = provider.api_key_env
                if env_name:
                    if not os.getenv(env_name):
                        raise ValueError(
                            f"Environment variable '{env_name}' required for cloud provider '{provider.name}' in tier '{tier_name}' when app_env='prod'"
                        )
        # Legacy fallback: if no new schema is used, fall back to the old field.
        if not self.models and self.app_env == "prod" and not self.cloud_provider_api_key:
            raise ValueError(
                "cloud_provider_api_key is required when app_env='prod' (legacy config)"
            )
        return self

    # ── YAML overlay ─────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from a YAML file, then overlay env vars on top.

        Env vars take precedence over YAML values, matching 12-factor
        principles.

        Supports both the new nested `models` schema and the legacy flat
        schema (with deprecation warning).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            yaml_data: dict[str, Any] = yaml.safe_load(f) or {}

        # Check if new nested schema is present
        has_new_schema = "models" in yaml_data

        if not has_new_schema:
            # Backward compatibility: synthesize models block from legacy flat keys
            warnings.warn(
                "Legacy flat configuration detected. Please migrate to the new "
                "'models' nested schema. The flat keys (cloud_provider_models, "
                "local_provider_models, etc.) are deprecated and will be removed "
                "in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            yaml_data = cls._synthesize_models_from_legacy(yaml_data)

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

    @classmethod
    def _synthesize_models_from_legacy(cls, yaml_data: dict[str, Any]) -> dict[str, Any]:
        """Synthesize the new 'models' block from legacy flat keys.
        
        Creates 'simple' and 'complex' tiers with mixed providers.
        """
        models: dict[str, Any] = {}

        # Cloud provider models
        cloud_models = yaml_data.get("cloud_provider_models", [])
        cloud_base_url = yaml_data.get("cloud_provider_base_url", "https://integrate.api.nvidia.com/v1")
        cloud_api_key_env = "CLOUD_PROVIDER_API_KEY"

        # Local provider models
        local_models = yaml_data.get("local_provider_models", [])
        local_base_url = yaml_data.get("local_provider_base_url", "http://localhost:11434")

        # Build 'simple' tier: fast/cheap models (both cloud and local)
        simple_providers = []
        
        # Add cloud nano model as first priority in simple tier
        if cloud_models:
            simple_providers.append({
                "name": "nvidia-nano",
                "provider": "nvidia",
                "provider_kind": "cloud",
                "base_url": cloud_base_url,
                "api_key_env": cloud_api_key_env,
                "model": cloud_models[0],
                "enabled": True,
                "priority": 1,
                "max_tokens": 4096,
                "cost_per_1k": 0.0001,
                "params": {"temperature": 0.3}
            })
        
        # Add local model as second priority in simple tier
        if local_models:
            simple_providers.append({
                "name": "ollama-fast",
                "provider": "ollama",
                "provider_kind": "local",
                "base_url": local_base_url,
                "api_key_env": None,
                "model": local_models[0],
                "enabled": True,
                "priority": 2,
                "max_tokens": 4096,
                "cost_per_1k": 0.0,
                "params": {"temperature": 0.3}
            })

        if simple_providers:
            models["simple"] = {
                "routing_strategy": "cost_optimized",
                "enabled": True,
                "fallback_tier": "complex",
                "providers": simple_providers
            }

        # Build 'complex' tier: more capable models (both cloud and local)
        complex_providers = []
        
        # Add cloud larger model as first priority in complex tier
        if len(cloud_models) > 1:
            complex_providers.append({
                "name": "nvidia-large",
                "provider": "nvidia",
                "provider_kind": "cloud",
                "base_url": cloud_base_url,
                "api_key_env": cloud_api_key_env,
                "model": cloud_models[1],
                "enabled": True,
                "priority": 1,
                "max_tokens": 8192,
                "cost_per_1k": 0.0005,
                "params": {"temperature": 0.5}
            })
        elif cloud_models:
            # If only one cloud model, use it in complex tier too
            complex_providers.append({
                "name": "nvidia-large",
                "provider": "nvidia",
                "provider_kind": "cloud",
                "base_url": cloud_base_url,
                "api_key_env": cloud_api_key_env,
                "model": cloud_models[0],
                "enabled": True,
                "priority": 1,
                "max_tokens": 8192,
                "cost_per_1k": 0.0005,
                "params": {"temperature": 0.5}
            })
        
        # Add local larger model as second priority in complex tier
        if len(local_models) > 1:
            complex_providers.append({
                "name": "ollama-large",
                "provider": "ollama",
                "provider_kind": "local",
                "base_url": local_base_url,
                "api_key_env": None,
                "model": local_models[1],
                "enabled": True,
                "priority": 2,
                "max_tokens": 8192,
                "cost_per_1k": 0.0,
                "params": {"temperature": 0.5}
            })
        elif local_models:
            complex_providers.append({
                "name": "ollama-large",
                "provider": "ollama",
                "provider_kind": "local",
                "base_url": local_base_url,
                "api_key_env": None,
                "model": local_models[0],
                "enabled": True,
                "priority": 2,
                "max_tokens": 8192,
                "cost_per_1k": 0.0,
                "params": {"temperature": 0.5}
            })

        if complex_providers:
            models["complex"] = {
                "routing_strategy": "quality_optimized",
                "enabled": True,
                "fallback_tier": None,  # No fallback from complex tier
                "providers": complex_providers
            }

        yaml_data["models"] = models
        return yaml_data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton.

    Loads settings from YAML config file (if exists), then overlays
    environment variables. In tests, call `get_settings.cache_clear()`
    after setting env vars.
    """
    # Check for explicit config file path via env var
    config_path = os.environ.get("CONFIG_FILE")
    
    # Default to settings.dev.yaml in config directory
    if not config_path:
        default_path = Path(__file__).parent.parent.parent / "config" / "settings.dev.yaml"
        if default_path.exists():
            config_path = str(default_path)
    
    # Also check for local override
    local_path = Path(__file__).parent.parent.parent / "config" / "settings.local.yaml"
    if local_path.exists():
        config_path = str(local_path)
    
    if config_path:
        return Settings.from_yaml(config_path)
    
    return Settings()
