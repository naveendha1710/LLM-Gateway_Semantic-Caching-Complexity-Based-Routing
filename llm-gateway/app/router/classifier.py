"""Request classifier for routing decisions.

Analyzes incoming requests and determines the appropriate routing strategy
based on model, complexity, cost constraints, and other factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.api.schemas.requests import ChatCompletionRequest
from app.router.tier_config import ProviderTierConfig, TierConfig, get_default_tier_config


class RoutingStrategy(str, Enum):
    """Strategy for selecting a provider."""

    COST_OPTIMIZED = "cost_optimized"  # Prefer cheapest available
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # Prefer highest capability
    TIER_PRIORITY = "tier_priority"  # Follow tier priority order
    MODEL_SPECIFIC = "model_specific"  # Use exact model requested


@dataclass(frozen=True)
class RoutingDecision:
    """Result of classification/routing decision."""

    strategy: RoutingStrategy
    selected_tier: str
    selected_model: str
    selected_provider: str
    fallback_tier: str | None = None
    fallback_model: str | None = None
    fallback_provider: str | None = None
    reasoning: str = ""
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


class RequestClassifier:
    """Classifies requests and makes routing decisions."""

    def __init__(
        self,
        tier_configs: list[TierConfig] | None = None,
        default_strategy: RoutingStrategy = RoutingStrategy.TIER_PRIORITY,
    ) -> None:
        self._tier_configs = tier_configs or get_default_tier_config()
        self._default_strategy = default_strategy

    def classify(self, request: ChatCompletionRequest) -> RoutingDecision:
        """Classify a request and return a routing decision."""
        # If model is explicitly specified, try to honor it
        if request.model:
            return self._classify_with_model(request)

        # Otherwise use default strategy
        return self._classify_default(request)

    def _classify_with_model(
        self, request: ChatCompletionRequest
    ) -> RoutingDecision:
        """Classify when a specific model is requested."""
        model = request.model

        # Find the tier and provider config for the requested model
        for tier in self._tier_configs:
            if not tier.enabled:
                continue
            for provider_config in tier.providers:
                if provider_config.model == model:
                    return RoutingDecision(
                        strategy=RoutingStrategy.MODEL_SPECIFIC,
                        selected_tier=tier.tier,
                        selected_model=model,
                        selected_provider=provider_config.name,
                        fallback_tier=tier.fallback_tier,
                        fallback_model=self._get_fallback_model(tier.fallback_tier),
                        fallback_provider=self._get_fallback_provider(tier.fallback_tier),
                        reasoning=f"Explicit model requested: {model}",
                        metadata={"requested_model": model},
                    )

        # Model not found in config - fall back to default strategy
        return self._classify_default(request)

    def _classify_default(self, request: ChatCompletionRequest) -> RoutingDecision:
        """Classify using the default strategy."""
        if self._default_strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._cost_optimized_routing(request)
        elif self._default_strategy == RoutingStrategy.PERFORMANCE_OPTIMIZED:
            return self._performance_optimized_routing(request)
        else:  # TIER_PRIORITY
            return self._tier_priority_routing(request)

    def _tier_priority_routing(self, request: ChatCompletionRequest) -> RoutingDecision:
        """Route based on tier priority order."""
        for tier in self._tier_configs:
            if not tier.enabled or not tier.providers:
                continue

            # Pick the highest priority provider in this tier (lowest priority number).
            # If multiple providers share the same priority, prefer a cloud provider
            # to maintain the original routing semantics where cloud models are
            # favored for tier‑priority routing.
            min_priority = min(p.priority for p in tier.providers)
            candidates = [p for p in tier.providers if p.priority == min_priority]
            # Prefer cloud over local when priorities are equal
            provider_config = next(
                (p for p in candidates if getattr(p, "provider_kind", "") == "cloud"),
                candidates[0],
            )

            return RoutingDecision(
                strategy=RoutingStrategy.TIER_PRIORITY,
                selected_tier=tier.tier,
                selected_model=provider_config.model,
                selected_provider=provider_config.name,
                fallback_tier=tier.fallback_tier,
                fallback_model=self._get_fallback_model(tier.fallback_tier),
                fallback_provider=self._get_fallback_provider(tier.fallback_tier),
                reasoning=f"Tier priority: {tier.tier} (model: {provider_config.model})",
                metadata={"tier": tier.tier},
            )

        # No enabled tiers - should not happen with default config
        raise RuntimeError("No enabled tiers available for routing")

    def _cost_optimized_routing(
        self, request: ChatCompletionRequest
    ) -> RoutingDecision:
        """Route to the cheapest available model."""
        cheapest: ProviderTierConfig | None = None
        cheapest_tier: TierConfig | None = None

        for tier in self._tier_configs:
            if not tier.enabled or not tier.providers:
                continue
            for provider_config in tier.providers:
                if cheapest is None or (provider_config.cost_per_1k or 0) < (cheapest.cost_per_1k or 0):
                    cheapest = provider_config
                    cheapest_tier = tier

        if cheapest is None or cheapest_tier is None:
            raise RuntimeError("No enabled models available for cost-optimized routing")

        return RoutingDecision(
            strategy=RoutingStrategy.COST_OPTIMIZED,
            selected_tier=cheapest_tier.tier,
            selected_model=cheapest.model,
            selected_provider=cheapest.name,
            fallback_tier=cheapest_tier.fallback_tier,
            fallback_model=self._get_fallback_model(cheapest_tier.fallback_tier),
            fallback_provider=self._get_fallback_provider(cheapest_tier.fallback_tier),
            reasoning=f"Cheapest model: {cheapest.model} (${cheapest.cost_per_1k}/1k tokens)",
            metadata={"cost_per_1k": cheapest.cost_per_1k},
        )

    def _performance_optimized_routing(
        self, request: ChatCompletionRequest
    ) -> RoutingDecision:
        """Route to the highest capability model."""
        best: ProviderTierConfig | None = None
        best_tier: TierConfig | None = None

        for tier in self._tier_configs:
            if not tier.enabled or not tier.providers:
                continue
            for provider_config in tier.providers:
                # Higher max_tokens = more capable (simplified heuristic)
                capability = provider_config.max_tokens or 0
                if best is None or capability > (best.max_tokens or 0):
                    best = provider_config
                    best_tier = tier

        if best is None or best_tier is None:
            raise RuntimeError("No enabled models available for performance routing")

        return RoutingDecision(
            strategy=RoutingStrategy.PERFORMANCE_OPTIMIZED,
            selected_tier=best_tier.tier,
            selected_model=best.model,
            selected_provider=best.name,
            fallback_tier=best_tier.fallback_tier,
            fallback_model=self._get_fallback_model(best_tier.fallback_tier),
            fallback_provider=self._get_fallback_provider(best_tier.fallback_tier),
            reasoning=f"Highest capability model: {best.model} ({best.max_tokens} tokens)",
            metadata={"max_tokens": best.max_tokens},
        )

    def _get_fallback_model(self, fallback_tier: str | None) -> str | None:
        """Get the default model for a fallback tier."""
        if fallback_tier is None:
            return None
        for tier in self._tier_configs:
            if tier.tier == fallback_tier and tier.enabled and tier.providers:
                return min(tier.providers, key=lambda p: p.priority).model
        return None

    def _get_fallback_provider(self, fallback_tier: str | None) -> str | None:
        """Get the provider name for a fallback tier."""
        if fallback_tier is None:
            return None
        for tier in self._tier_configs:
            if tier.tier == fallback_tier:
                if tier.providers:
                    return min(tier.providers, key=lambda p: p.priority).name
        return None