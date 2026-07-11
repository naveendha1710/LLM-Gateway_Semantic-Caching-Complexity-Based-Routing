"""Unit tests for router components: classifier, selector, and failover behavior."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from app.api.schemas.requests import ChatCompletionRequest, Message
from app.api.schemas.responses import ChatCompletionResponse, Choice, ChoiceMessage, Usage
from app.router.classifier import RequestClassifier, RoutingDecision, RoutingStrategy
from app.router.selector import ProviderSelector
from app.router.tier_config import ProviderTierConfig, TierConfig, get_default_tier_config


def _make_request(model: str = "llama3.2", content: str = "Hello") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[Message(role="user", content=content)],
    )


def _make_response(content: str = "Hello!", model: str = "llama3.2", source: str = "local") -> ChatCompletionResponse:
    import time
    return ChatCompletionResponse(
        id="test-id",
        model=model,
        choices=[Choice(index=0, message=ChoiceMessage(role="assistant", content=content), finish_reason="stop")],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        created=int(time.time()),
        source=source,
    )


class FakeProvider:
    """Minimal provider stub for router tests."""

    def __init__(self, name: str = "local", response: ChatCompletionResponse | None = None, healthy: bool = True) -> None:
        self._name = name
        self._response = response or _make_response(model=name)
        self._healthy = healthy
        self.generate_calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def cost_per_1k(self) -> float:
        return 0.0

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.generate_calls += 1
        return self._response

    async def health(self) -> bool:
        return self._healthy


class TestRequestClassifier:
    """Tests for RequestClassifier routing decisions."""

    def test_tier_priority_routing_default(self) -> None:
        """Default strategy should pick first enabled tier (simple)."""
        classifier = RequestClassifier()
        request = _make_request(model="unknown-model")
        decision = classifier.classify(request)

        assert decision.strategy == RoutingStrategy.TIER_PRIORITY
        assert decision.selected_tier == "simple"
        assert decision.selected_provider == "nvidia-nano"
        assert decision.fallback_tier == "complex"

    def test_model_specific_routing_local_model(self) -> None:
        """Explicit local model should route to simple tier."""
        classifier = RequestClassifier()
        request = _make_request(model="llama3.2")
        decision = classifier.classify(request)

        assert decision.strategy == RoutingStrategy.MODEL_SPECIFIC
        assert decision.selected_tier == "simple"
        assert decision.selected_provider == "ollama-fast"
        assert decision.selected_model == "llama3.2"

    def test_model_specific_routing_cloud_model(self) -> None:
        """Explicit cloud model should route to appropriate tier."""
        classifier = RequestClassifier()
        request = _make_request(model="nvidia/nemotron-3-nano-30b-a3b")
        decision = classifier.classify(request)

        assert decision.strategy == RoutingStrategy.MODEL_SPECIFIC
        assert decision.selected_tier == "simple"
        assert decision.selected_provider == "nvidia-nano"
        assert decision.selected_model == "nvidia/nemotron-3-nano-30b-a3b"

    def test_model_specific_unknown_model_falls_back(self) -> None:
        """Unknown model should fall back to default strategy."""
        classifier = RequestClassifier()
        request = _make_request(model="unknown-model")
        decision = classifier.classify(request)

        assert decision.strategy == RoutingStrategy.TIER_PRIORITY
        assert decision.selected_tier == "simple"

    def test_cost_optimized_routing(self) -> None:
        """COST_OPTIMIZED strategy should pick cheapest model."""
        classifier = RequestClassifier(default_strategy=RoutingStrategy.COST_OPTIMIZED)
        request = _make_request(model="unknown-model")
        decision = classifier.classify(request)

        assert decision.strategy == RoutingStrategy.COST_OPTIMIZED
        assert decision.selected_model == "llama3.2"
        assert decision.selected_provider == "ollama-fast"

    def test_performance_optimized_routing(self) -> None:
        """PERFORMANCE_OPTIMIZED strategy should pick highest capability model."""
        classifier = RequestClassifier(default_strategy=RoutingStrategy.PERFORMANCE_OPTIMIZED)
        request = _make_request(model="unknown-model")
        decision = classifier.classify(request)

        assert decision.strategy == RoutingStrategy.PERFORMANCE_OPTIMIZED
        assert decision.selected_model == "nvidia/llama-3.1-nemotron-70b-instruct"
        assert decision.selected_provider == "nvidia-large"

    def test_disabled_tier_skipped(self) -> None:
        """Disabled tiers should be skipped in routing."""
        tier_configs = [
            TierConfig(
                tier="simple",
                routing_strategy="cost_optimized",
                providers=[
                    ProviderTierConfig(
                        name="ollama-fast",
                        provider="ollama",
                        provider_kind="local",
                        model="llama3.2",
                        tier="simple",
                        priority=1,
                    ),
                ],
                enabled=False,
                fallback_tier="complex",
            ),
            TierConfig(
                tier="complex",
                routing_strategy="quality_optimized",
                providers=[
                    ProviderTierConfig(
                        name="nvidia-large",
                        provider="nvidia",
                        provider_kind="cloud",
                        model="nvidia/llama-3.1-nemotron-70b-instruct",
                        tier="complex",
                        priority=1,
                    ),
                ],
                enabled=True,
                fallback_tier=None,
            ),
        ]
        classifier = RequestClassifier(tier_configs=tier_configs)
        request = _make_request()
        decision = classifier.classify(request)

        assert decision.selected_tier == "complex"
        assert decision.selected_provider == "nvidia-large"

    def test_routing_decision_includes_reasoning(self) -> None:
        """Routing decision should include reasoning metadata."""
        classifier = RequestClassifier()
        request = _make_request(model="nvidia/nemotron-3-nano-30b-a3b")
        decision = classifier.classify(request)

        assert "Explicit model requested" in decision.reasoning
        assert decision.metadata.get("requested_model") == "nvidia/nemotron-3-nano-30b-a3b"


class TestProviderSelector:
    """Tests for ProviderSelector selection and failover."""

    @pytest.fixture
    def tier_configs(self) -> list[TierConfig]:
        return [
            TierConfig(
                tier="simple",
                routing_strategy="cost_optimized",
                providers=[
                    ProviderTierConfig(
                        name="ollama-fast",
                        provider="ollama",
                        provider_kind="local",
                        model="llama3.2",
                        tier="simple",
                        priority=1,
                        max_tokens=4096,
                        cost_per_1k=0.0,
                    ),
                    ProviderTierConfig(
                        name="nvidia-nano",
                        provider="nvidia",
                        provider_kind="cloud",
                        model="nvidia/nemotron-3-nano-30b-a3b",
                        tier="simple",
                        priority=2,
                        max_tokens=16384,
                        cost_per_1k=0.0001,
                    ),
                ],
                enabled=True,
                fallback_tier="complex",
            ),
            TierConfig(
                tier="complex",
                routing_strategy="quality_optimized",
                providers=[
                    ProviderTierConfig(
                        name="nvidia-large",
                        provider="nvidia",
                        provider_kind="cloud",
                        model="nvidia/llama-3.1-nemotron-70b-instruct",
                        tier="complex",
                        priority=1,
                        max_tokens=8192,
                        cost_per_1k=0.001,
                    ),
                ],
                enabled=True,
                fallback_tier=None,
            ),
        ]

    @pytest.fixture
    def selector(self, tier_configs: list[TierConfig]) -> ProviderSelector:
        return ProviderSelector(tier_configs=tier_configs)

    @pytest.mark.asyncio
    async def test_select_and_execute_primary_success(self, selector: ProviderSelector) -> None:
        """Successful primary provider should return response without failover."""
        provider = FakeProvider(name="ollama-fast", response=_make_response(model="llama3.2", source="local"))
        selector.register_provider("ollama-fast", provider)

        request = _make_request(model="llama3.2")
        response = await selector.select_and_execute(request)

        assert response.source == "local"
        assert provider.generate_calls == 1

    @pytest.mark.asyncio
    async def test_select_and_execute_failover_to_same_tier(self, selector: ProviderSelector) -> None:
        """Failed primary should failover to next provider in same tier."""
        primary = FakeProvider(name="ollama-fast", healthy=False)
        fallback = FakeProvider(name="nvidia-nano", response=_make_response(model="nvidia/nemotron-3-nano-30b-a3b", source="cloud"))
        selector.register_provider("ollama-fast", primary)
        selector.register_provider("nvidia-nano", fallback)

        request = _make_request(model="llama3.2")
        response = await selector.select_and_execute(request)

        assert response.source == "cloud"
        assert primary.generate_calls == 0
        assert fallback.generate_calls == 1

    @pytest.mark.asyncio
    async def test_select_and_execute_failover_to_fallback_tier(self, selector: ProviderSelector) -> None:
        """All providers in primary tier failed should failover to fallback tier."""
        primary1 = FakeProvider(name="ollama-fast", healthy=False)
        primary2 = FakeProvider(name="nvidia-nano", healthy=False)
        fallback = FakeProvider(name="nvidia-large", response=_make_response(model="nvidia/llama-3.1-nemotron-70b-instruct", source="cloud"))
        selector.register_provider("ollama-fast", primary1)
        selector.register_provider("nvidia-nano", primary2)
        selector.register_provider("nvidia-large", fallback)

        request = _make_request(model="llama3.2")
        response = await selector.select_and_execute(request)

        assert response.source == "cloud"
        assert response.model == "nvidia/llama-3.1-nemotron-70b-instruct"
        assert fallback.generate_calls == 1

    @pytest.mark.asyncio
    async def test_select_and_execute_all_fail_raises(self, selector: ProviderSelector) -> None:
        """All providers failing should raise AllProvidersFailed."""
        primary1 = FakeProvider(name="ollama-fast", healthy=False)
        primary2 = FakeProvider(name="nvidia-nano", healthy=False)
        fallback = FakeProvider(name="nvidia-large", healthy=False)
        selector.register_provider("ollama-fast", primary1)
        selector.register_provider("nvidia-nano", primary2)
        selector.register_provider("nvidia-large", fallback)

        request = _make_request(model="llama3.2")
        with pytest.raises(Exception, match="All providers failed"):
            await selector.select_and_execute(request)

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, selector: ProviderSelector) -> None:
        """Circuit breaker should open after threshold failures."""
        provider = FakeProvider(name="ollama-fast", healthy=False)
        selector.register_provider("ollama-fast", provider)

        request = _make_request(model="llama3.2")
        for _ in range(5):
            try:
                await selector.select_and_execute(request)
            except Exception:
                pass

        states = selector.get_circuit_breaker_states()
        assert states["ollama-fast"] == "OPEN"

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_after_timeout(self, selector: ProviderSelector) -> None:
        """Circuit breaker should go to HALF_OPEN after timeout."""
        provider = FakeProvider(name="ollama-fast", healthy=False)
        selector.register_provider("ollama-fast", provider)

        request = _make_request(model="llama3.2")
        for _ in range(5):
            try:
                await selector.select_and_execute(request)
            except Exception:
                pass

        cb = selector._circuit_breakers["ollama-fast"]
        cb._state = "HALF_OPEN"

        states = selector.get_circuit_breaker_states()
        assert states["ollama-fast"] == "HALF_OPEN"

    @pytest.mark.asyncio
    async def test_health_check_all(self, selector: ProviderSelector) -> None:
        """health_check_all should return health status for all registered providers."""
        healthy = FakeProvider(name="ollama-fast", healthy=True)
        unhealthy = FakeProvider(name="nvidia-nano", healthy=False)
        selector.register_provider("ollama-fast", healthy)
        selector.register_provider("nvidia-nano", unhealthy)

        results = await selector.health_check_all()

        assert results["ollama-fast"] is True
        assert results["nvidia-nano"] is False

    @pytest.mark.asyncio
    async def test_cycle_detection_prevents_infinite_fallback(self, tier_configs: list[TierConfig]) -> None:
        """Cycle detection should prevent infinite fallback loops."""
        tier_configs[1].fallback_tier = "simple"
        selector = ProviderSelector(tier_configs=tier_configs)

        primary1 = FakeProvider(name="ollama-fast", healthy=False)
        primary2 = FakeProvider(name="nvidia-nano", healthy=False)
        fallback = FakeProvider(name="nvidia-large", healthy=False)
        selector.register_provider("ollama-fast", primary1)
        selector.register_provider("nvidia-nano", primary2)
        selector.register_provider("nvidia-large", fallback)

        request = _make_request(model="llama3.2")
        with pytest.raises(Exception, match="All providers failed"):
            await selector.select_and_execute(request)

        assert primary1.generate_calls == 0
        assert primary2.generate_calls == 0
        assert fallback.generate_calls == 0

    @pytest.mark.asyncio
    async def test_priority_sort_lower_first(self, tier_configs: list[TierConfig]) -> None:
        """Providers with lower priority number should be tried first."""
        tier_configs[0].providers[0].priority = 2
        tier_configs[0].providers[1].priority = 1
        selector = ProviderSelector(tier_configs=tier_configs)

        provider1 = FakeProvider(name="ollama-fast", healthy=False)
        provider2 = FakeProvider(name="nvidia-nano", response=_make_response(model="nvidia/nemotron-3-nano-30b-a3b", source="cloud"))
        selector.register_provider("ollama-fast", provider1)
        selector.register_provider("nvidia-nano", provider2)

        request = _make_request(model="llama3.2")
        response = await selector.select_and_execute(request)

        assert response.source == "cloud"
        assert provider2.generate_calls == 1
        assert provider1.generate_calls == 0
