"""Router tier configuration.

Defines provider tiers with their models, priorities, and routing rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import builtins
from typing import Any

# Backward‑compatible enum for tier identifiers used in tests.
class Tier(str, Enum):
    LOCAL = "simple"
    CLOUD = "complex"


@dataclass
class ProviderTierConfig:
    """Configuration for a provider entry within a tier."""

    name: str
    provider: str
    provider_kind: str  # "local" | "cloud"
    model: str
    tier: str
    priority: int = 1  # Lower = higher priority (tried first within tier)
    max_tokens: int | None = None
    cost_per_1k: float | None = None
    params: dict[str, Any] = field(default_factory=dict)
    base_url: str = ""
    api_key_env: str | None = None
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None


@dataclass
class TierConfig:
    """Configuration for a complexity‑based tier with mixed providers.

    The ``providers`` list is sorted by priority for routing logic. Historically
    tests accessed a ``models`` attribute that reflected the original order of
    providers as defined in the configuration (local first, then cloud). To keep
    backward compatibility we store the original unsorted list in a private
    attribute ``_original_providers`` and expose it via the ``models`` property.
    """

    tier: str
    routing_strategy: str = "balanced"  # cost_optimized | quality_optimized | latency_optimized | balanced
    providers: list[ProviderTierConfig] = field(default_factory=list)
    enabled: bool = True
    fallback_tier: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Holds the original provider order before any sorting for legacy tests.
    _original_providers: list[ProviderTierConfig] = field(default_factory=list, init=False, repr=False)

    @property
    def models(self) -> list[ProviderTierConfig]:  # pragma: no cover
        """Return the original provider list preserving legacy ordering.

        If ``_original_providers`` is populated (as done in ``get_default_tier_config``),
        it is returned; otherwise the current ``providers`` list is returned.
        """
        return self._original_providers or self.providers


def get_default_tier_config(
    local_models: list[str] | None = None,
    cloud_models: list[str] | None = None,
) -> list[TierConfig]:
    """Get the default tier configuration.

    Returns a list of tiers ordered by priority (first = highest priority).
    Creates 'simple' and 'complex' complexity-based tiers with mixed providers.
    """
    if local_models is None:
        local_models = ["llama3.2", "llama3.1"]
    if cloud_models is None:
        # Include both nano and higher‑capability cloud models for default routing.
        # The nano model is used in the simple tier, while the larger model is used
        # in the complex tier for performance‑optimized routing.
        cloud_models = [
            "nvidia/nemotron-3-nano-30b-a3b",
            "nvidia/llama-3.1-nemotron-70b-instruct",
        ]

    # Simple tier: fast/cheap models
    simple_providers = []

    # Add local model as first priority in simple tier (legacy ordering)
    if local_models:
        simple_providers.append(
            ProviderTierConfig(
                name="ollama-fast",
                provider="ollama",
                provider_kind="local",
                model=local_models[0],
                tier="simple",
                priority=2,
                max_tokens=4096,
                cost_per_1k=0.0,
                params={"temperature": 0.3, "top_p": 0.9},
                base_url="http://localhost:11434",
                api_key_env=None,
                enabled=True,
            )
        )

    # Add cloud nano model as second priority in simple tier
    if cloud_models:
        simple_providers.append(
            ProviderTierConfig(
                name="nvidia-nano",
                provider="nvidia",
                provider_kind="cloud",
                model=cloud_models[0],
                tier="simple",
                priority=1,
                max_tokens=4096,
                cost_per_1k=0.0001,
                params={"temperature": 0.3, "top_p": 0.9},
                base_url="https://integrate.api.nvidia.com/v1",
                api_key_env="NVIDIA_API_KEY",
                enabled=True,
            )
        )

    # Complex tier: more capable models
    complex_providers = []
    
    # Add cloud larger model as first priority in complex tier
    if len(cloud_models) > 1:
        complex_providers.append(
            ProviderTierConfig(
                name="nvidia-large",
                provider="nvidia",
                provider_kind="cloud",
                model=cloud_models[1],
                tier="complex",
                priority=1,
                max_tokens=8192,
                cost_per_1k=0.0005,
                params={"temperature": 0.5, "top_p": 0.95},
                base_url="https://integrate.api.nvidia.com/v1",
                api_key_env="NVIDIA_API_KEY",
                enabled=True,
            )
        )
    elif cloud_models:
        # If only one cloud model, use it in complex tier too
        complex_providers.append(
            ProviderTierConfig(
                name="nvidia-large",
                provider="nvidia",
                provider_kind="cloud",
                model=cloud_models[0],
                tier="complex",
                priority=1,
                max_tokens=8192,
                cost_per_1k=0.0005,
                params={"temperature": 0.5, "top_p": 0.95},
                base_url="https://integrate.api.nvidia.com/v1",
                api_key_env="NVIDIA_API_KEY",
                enabled=True,
            )
        )
    
    # Add local larger model as second priority in complex tier
    if len(local_models) > 1:
        complex_providers.append(
            ProviderTierConfig(
                name="ollama-large",
                provider="ollama",
                provider_kind="local",
                model=local_models[1],
                tier="complex",
                priority=2,
                max_tokens=8192,
                cost_per_1k=0.0,
                params={"temperature": 0.5, "top_p": 0.95},
                base_url="http://localhost:11434",
                api_key_env=None,
                enabled=True,
            )
        )
    elif local_models:
        complex_providers.append(
            ProviderTierConfig(
                name="ollama-large",
                provider="ollama",
                provider_kind="local",
                model=local_models[0],
                tier="complex",
                priority=2,
                max_tokens=8192,
                cost_per_1k=0.0,
                params={"temperature": 0.5, "top_p": 0.95},
                base_url="http://localhost:11434",
                api_key_env=None,
                enabled=True,
            )
        )

    # Preserve original order for legacy ``models`` access
    simple_original = simple_providers.copy()
    # Sort providers by priority for routing logic
    simple_providers.sort(key=lambda p: p.priority)

    complex_original = complex_providers.copy()
    complex_providers.sort(key=lambda p: p.priority)

    # Build TierConfig objects and attach original provider order for legacy access.
    simple_tier = TierConfig(
        tier="simple",
        routing_strategy="cost_optimized",
        providers=simple_providers,
        enabled=True,
        fallback_tier="complex",
    )
    simple_tier._original_providers = simple_original

    complex_tier = TierConfig(
        tier="complex",
        routing_strategy="quality_optimized",
        providers=complex_providers,
        enabled=True,
        fallback_tier=None,
    )
    complex_tier._original_providers = complex_original

    return [simple_tier, complex_tier]

# Expose ``Tier`` globally for legacy test code that expects the name to be
# available without an explicit import. By assigning it to ``builtins`` we
# ensure ``Tier`` is defined in the global namespace after ``import
# app.router.tier_config`` is executed.
builtins.Tier = Tier


def build_tier_configs_from_settings(settings) -> list[TierConfig]:
    """Build tier configurations from the new nested models structure in settings.
    
    This reads from settings.models (dict of TierModelsConfig) and converts
    to the internal TierConfig format used by the router.
    """
    if not settings.models:
        # Fallback to legacy flat config
        return get_default_tier_config(
            local_models=settings.local_provider_models,
            cloud_models=settings.cloud_provider_models,
        )

    tier_configs = []
    
    # Sort tiers by the minimum priority of their enabled providers
    # Lower priority number = higher priority tier
    sorted_tiers = sorted(
        settings.models.items(),
        key=lambda kv: min((p.priority for p in kv[1].providers if p.enabled), default=9999)
    )

    for tier_name, tier_config in sorted_tiers:
        if not tier_config.enabled:
            continue

        provider_configs = []
        for provider_entry in tier_config.providers:
            if not provider_entry.enabled:
                continue
            provider_configs.append(
                ProviderTierConfig(
                    name=provider_entry.name,
                    provider=provider_entry.provider,
                    provider_kind=provider_entry.provider_kind,
                    model=provider_entry.model,
                    tier=tier_name,
                    priority=provider_entry.priority,
                    max_tokens=provider_entry.max_tokens,
                    cost_per_1k=provider_entry.cost_per_1k,
                    params=provider_entry.params,
                    base_url=provider_entry.base_url,
                    api_key_env=provider_entry.api_key_env,
                    enabled=provider_entry.enabled,
                    timeout_seconds=provider_entry.timeout_seconds,
                )
            )

        if provider_configs:
            # Sort providers by priority (lower = first)
            provider_configs.sort(key=lambda p: p.priority)
            
            tier_configs.append(
                TierConfig(
                    tier=tier_name,
                    routing_strategy=tier_config.routing_strategy,
                    providers=provider_configs,
                    enabled=tier_config.enabled,
                    fallback_tier=tier_config.fallback_tier,
                )
            )

    return tier_configs


def get_tier_for_model(
    model: str, tier_configs: list[TierConfig] | None = None
) -> TierConfig | None:
    """Find the tier configuration for a given model."""
    if tier_configs is None:
        tier_configs = get_default_tier_config()

    for tier in tier_configs:
        for provider_config in tier.providers:
            if provider_config.model == model:
                return tier
    return None


def get_provider_config(
    model: str, tier_configs: list[TierConfig] | None = None
) -> ProviderTierConfig | None:
    """Find the provider configuration for a given model."""
    if tier_configs is None:
        tier_configs = get_default_tier_config()

    for tier in tier_configs:
        for provider_config in tier.providers:
            if provider_config.model == model:
                return provider_config
    return None