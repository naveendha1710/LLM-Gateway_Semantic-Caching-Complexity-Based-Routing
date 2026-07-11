"""Tests for router tier configuration with complexity-based tiers."""

from __future__ import annotations

import pytest

from app.router.tier_config import (
    ProviderTierConfig,
    TierConfig,
    build_tier_configs_from_settings,
    get_default_tier_config,
    get_provider_config,
    get_tier_for_model,
)


class TestProviderTierConfig:
    """Tests for ProviderTierConfig dataclass."""

    def test_provider_tier_config_creation(self):
        """Test ProviderTierConfig can be created with all fields."""
        config = ProviderTierConfig(
            name="nvidia-primary",
            provider="nvidia",
            provider_kind="cloud",
            model="nvidia/nemotron-3-nano-30b-a3b",
            tier="simple",
            priority=10,
            max_tokens=16384,
            cost_per_1k=0.15,
            params={"temperature": 0.7},
            base_url="https://integrate.api.nvidia.com/v1",
            api_key_env="NVIDIA_API_KEY",
            enabled=True,
        )
        assert config.name == "nvidia-primary"
        assert config.provider == "nvidia"
        assert config.provider_kind == "cloud"
        assert config.model == "nvidia/nemotron-3-nano-30b-a3b"
        assert config.tier == "simple"
        assert config.priority == 10
        assert config.max_tokens == 16384
        assert config.cost_per_1k == 0.15
        assert config.params == {"temperature": 0.7}
        assert config.base_url == "https://integrate.api.nvidia.com/v1"
        assert config.api_key_env == "NVIDIA_API_KEY"
        assert config.enabled is True

    def test_provider_tier_config_defaults(self):
        """Test ProviderTierConfig defaults."""
        config = ProviderTierConfig(
            name="test",
            provider="ollama",
            provider_kind="local",
            model="llama3.2",
            tier="simple",
        )
        assert config.priority == 1  # Default is 1 (not 0)
        assert config.max_tokens is None
        assert config.cost_per_1k is None
        assert config.params == {}
        assert config.base_url == ""
        assert config.api_key_env is None
        assert config.enabled is True
        assert config.metadata == {}

    def test_provider_tier_config_priority_lower_is_higher_priority(self):
        """Test that lower priority number means higher priority (tried first)."""
        config1 = ProviderTierConfig(
            name="high-priority",
            provider="ollama",
            provider_kind="local",
            model="llama3.2",
            tier="simple",
            priority=1,
        )
        config2 = ProviderTierConfig(
            name="low-priority",
            provider="ollama",
            provider_kind="local",
            model="llama3.1",
            tier="simple",
            priority=10,
        )
        # Lower priority number = higher priority
        assert config1.priority < config2.priority


class TestTierConfig:
    """Tests for TierConfig dataclass."""

    def test_tier_config_creation(self):
        """Test TierConfig can be created with all fields."""
        providers = [
            ProviderTierConfig(
                name="ollama-primary",
                provider="ollama",
                provider_kind="local",
                model="llama3.2",
                tier="simple",
                priority=1,
            ),
            ProviderTierConfig(
                name="nvidia-primary",
                provider="nvidia",
                provider_kind="cloud",
                model="nvidia/nemotron-3-nano-30b-a3b",
                tier="simple",
                priority=2,
            ),
        ]
        config = TierConfig(
            tier="simple",
            routing_strategy="cost_optimized",
            providers=providers,
            enabled=True,
            fallback_tier="complex",
            metadata={"region": "us-east"},
        )
        assert config.tier == "simple"
        assert config.routing_strategy == "cost_optimized"
        assert len(config.providers) == 2
        assert config.enabled is True
        assert config.fallback_tier == "complex"
        assert config.metadata == {"region": "us-east"}

    def test_tier_config_defaults(self):
        """Test TierConfig defaults."""
        config = TierConfig(tier="complex", routing_strategy="tier_priority")
        assert config.providers == []
        assert config.enabled is True
        assert config.fallback_tier is None
        assert config.metadata == {}

    def test_tier_config_routing_strategies(self):
        """Test valid routing strategies."""
        strategies = ["cost_optimized", "quality_optimized", "latency_optimized", "balanced"]
        for strategy in strategies:
            config = TierConfig(tier="test", routing_strategy=strategy)
            assert config.routing_strategy == strategy

    def test_tier_config_mixed_providers(self):
        """Test tier can have mixed local and cloud providers."""
        providers = [
            ProviderTierConfig(
                name="nvidia-nano",
                provider="nvidia",
                provider_kind="cloud",
                model="nvidia/nemotron-3-nano-30b-a3b",
                tier="simple",
                priority=1,
            ),
            ProviderTierConfig(
                name="ollama-fast",
                provider="ollama",
                provider_kind="local",
                model="llama3.2",
                tier="simple",
                priority=2,
            ),
        ]
        config = TierConfig(
            tier="simple",
            routing_strategy="cost_optimized",
            providers=providers,
        )
        assert len(config.providers) == 2
        provider_kinds = {p.provider_kind for p in config.providers}
        assert "cloud" in provider_kinds
        assert "local" in provider_kinds


class TestGetDefaultTierConfig:
    """Tests for get_default_tier_config function."""

    def test_returns_two_tiers_simple_and_complex(self):
        """Test default config returns simple and complex tiers."""
        tiers = get_default_tier_config()
        tier_names = [t.tier for t in tiers]
        assert "simple" in tier_names
        assert "complex" in tier_names
        assert len(tiers) == 2

    def test_simple_tier_has_providers(self):
        """Test simple tier has providers configured."""
        tiers = get_default_tier_config()
        simple_tier = next(t for t in tiers if t.tier == "simple")
        assert len(simple_tier.providers) >= 1
        assert simple_tier.enabled is True
        assert simple_tier.fallback_tier == "complex"
        assert simple_tier.routing_strategy == "cost_optimized"

    def test_complex_tier_has_providers(self):
        """Test complex tier has providers configured."""
        tiers = get_default_tier_config()
        complex_tier = next(t for t in tiers if t.tier == "complex")
        assert len(complex_tier.providers) >= 1
        assert complex_tier.enabled is True
        assert complex_tier.fallback_tier is None
        assert complex_tier.routing_strategy == "quality_optimized"

    def test_simple_tier_providers_sorted_by_priority(self):
        """Test simple tier providers are sorted by priority (lower first)."""
        tiers = get_default_tier_config()
        simple_tier = next(t for t in tiers if t.tier == "simple")
        priorities = [p.priority for p in simple_tier.providers]
        assert priorities == sorted(priorities)

    def test_complex_tier_providers_sorted_by_priority(self):
        """Test complex tier providers are sorted by priority (lower first)."""
        tiers = get_default_tier_config()
        complex_tier = next(t for t in tiers if t.tier == "complex")
        priorities = [p.priority for p in complex_tier.providers]
        assert priorities == sorted(priorities)

    def test_custom_local_models(self):
        """Test custom local models list."""
        tiers = get_default_tier_config(local_models=["mistral:7b", "codellama:7b"])
        simple_tier = next(t for t in tiers if t.tier == "simple")
        # Should have at least the local model in simple tier
        local_providers = [p for p in simple_tier.providers if p.provider_kind == "local"]
        assert len(local_providers) >= 1

    def test_custom_cloud_models(self):
        """Test custom cloud models list."""
        tiers = get_default_tier_config(cloud_models=["model-a", "model-b", "model-c"])
        simple_tier = next(t for t in tiers if t.tier == "simple")
        cloud_providers = [p for p in simple_tier.providers if p.provider_kind == "cloud"]
        assert len(cloud_providers) >= 1

    def test_local_model_cost_zero(self):
        """Test local models have zero cost."""
        tiers = get_default_tier_config(local_models=["llama3.2"])
        simple_tier = next(t for t in tiers if t.tier == "simple")
        local_providers = [p for p in simple_tier.providers if p.provider_kind == "local"]
        for p in local_providers:
            assert p.cost_per_1k == 0.0

    def test_cloud_model_cost_positive(self):
        """Test cloud models have positive cost."""
        tiers = get_default_tier_config(cloud_models=["model1"])
        simple_tier = next(t for t in tiers if t.tier == "simple")
        cloud_providers = [p for p in simple_tier.providers if p.provider_kind == "cloud"]
        for p in cloud_providers:
            assert p.cost_per_1k is not None and p.cost_per_1k > 0.0

    def test_provider_has_base_url_and_api_key_env(self):
        """Test providers have base_url and api_key_env configured."""
        tiers = get_default_tier_config()
        for tier in tiers:
            for provider in tier.providers:
                assert provider.base_url is not None
                assert provider.base_url != ""
                if provider.provider_kind == "cloud":
                    assert provider.api_key_env is not None
                    assert provider.api_key_env != ""
                else:
                    # Local providers may have None api_key_env
                    pass

    def test_tier_order_simple_first(self):
        """Test tiers are ordered with simple first (higher priority)."""
        tiers = get_default_tier_config()
        assert tiers[0].tier == "simple"
        assert tiers[1].tier == "complex"

    def test_empty_model_lists(self):
        """Test empty model lists work."""
        tiers = get_default_tier_config(local_models=[], cloud_models=[])
        # Should still return two tiers but with no providers
        assert len(tiers) == 2
        for tier in tiers:
            assert tier.providers == []


class TestGetTierForModel:
    """Tests for get_tier_for_model function."""

    def test_finds_tier_for_model(self):
        """Test finding tier for a known model."""
        tiers = get_default_tier_config()
        tier = get_tier_for_model("llama3.2", tiers)
        assert tier is not None
        assert tier.tier == "simple"

    def test_returns_none_for_unknown_model(self):
        """Test returns None for unknown model."""
        tiers = get_default_tier_config()
        tier = get_tier_for_model("unknown-model", tiers)
        assert tier is None

    def test_finds_complex_tier_model(self):
        """Test finding model in complex tier."""
        tiers = get_default_tier_config()
        # The complex tier should have llama3.1
        tier = get_tier_for_model("llama3.1", tiers)
        assert tier is not None
        assert tier.tier == "complex"


class TestGetProviderConfig:
    """Tests for get_provider_config function."""

    def test_finds_provider_for_model(self):
        """Test finding provider config for a known model."""
        tiers = get_default_tier_config()
        provider = get_provider_config("llama3.2", tiers)
        assert provider is not None
        assert provider.model == "llama3.2"
        assert provider.provider_kind == "local"

    def test_returns_none_for_unknown_model(self):
        """Test returns None for unknown model."""
        tiers = get_default_tier_config()
        provider = get_provider_config("unknown-model", tiers)
        assert provider is None

    def test_returns_correct_provider_details(self):
        """Test provider config has correct details."""
        tiers = get_default_tier_config()
        provider = get_provider_config("nvidia/nemotron-3-nano-30b-a3b", tiers)
        assert provider is not None
        assert provider.provider == "nvidia"
        assert provider.provider_kind == "cloud"
        assert provider.api_key_env == "NVIDIA_API_KEY"


class TestBuildTierConfigsFromSettings:
    """Tests for build_tier_configs_from_settings function."""

    def test_builds_from_settings_models_dict(self):
        """Test building tier configs from settings.models dict."""
        from app.core.config import Settings, ProviderEntry, TierModelsConfig
        
        # Create a mock settings object with models dict
        settings = Settings()
        settings.models = {
            "simple": TierModelsConfig(
                enabled=True,
                routing_strategy="cost_optimized",
                fallback_tier="complex",
                providers=[
                    ProviderEntry(
                        name="nvidia-nano",
                        provider="nvidia",
                        provider_kind="cloud",
                        model="nvidia/nemotron-3-nano-30b-a3b",
                        priority=1,
                        max_tokens=4096,
                        cost_per_1k=0.0001,
                        params={"temperature": 0.3},
                        base_url="https://integrate.api.nvidia.com/v1",
                        api_key_env="NVIDIA_API_KEY",
                        enabled=True,
                    ),
                    ProviderEntry(
                        name="ollama-fast",
                        provider="ollama",
                        provider_kind="local",
                        model="llama3.2",
                        priority=2,
                        max_tokens=4096,
                        cost_per_1k=0.0,
                        params={"temperature": 0.3},
                        base_url="http://localhost:11434",
                        api_key_env=None,
                        enabled=True,
                    ),
                ],
            ),
            "complex": TierModelsConfig(
                enabled=True,
                routing_strategy="quality_optimized",
                fallback_tier=None,
                providers=[
                    ProviderEntry(
                        name="nvidia-large",
                        provider="nvidia",
                        provider_kind="cloud",
                        model="nvidia/llama-3.1-nemotron-70b-instruct",
                        priority=1,
                        max_tokens=8192,
                        cost_per_1k=0.0005,
                        params={"temperature": 0.5},
                        base_url="https://integrate.api.nvidia.com/v1",
                        api_key_env="NVIDIA_API_KEY",
                        enabled=True,
                    ),
                ],
            ),
        }
        
        tier_configs = build_tier_configs_from_settings(settings)
        
        assert len(tier_configs) == 2
        assert tier_configs[0].tier == "simple"
        assert tier_configs[1].tier == "complex"
        assert len(tier_configs[0].providers) == 2
        assert len(tier_configs[1].providers) == 1

    def test_filters_disabled_tiers(self):
        """Test disabled tiers are filtered out."""
        from app.core.config import Settings, ProviderEntry, TierModelsConfig
        
        settings = Settings()
        settings.models = {
            "simple": TierModelsConfig(
                enabled=True,
                routing_strategy="cost_optimized",
                providers=[
                    ProviderEntry(
                        name="test",
                        provider="ollama",
                        provider_kind="local",
                        model="llama3.2",
                        priority=1,
                        enabled=True,
                    ),
                ],
            ),
            "complex": TierModelsConfig(
                enabled=False,  # Disabled
                routing_strategy="quality_optimized",
                providers=[
                    ProviderEntry(
                        name="test2",
                        provider="ollama",
                        provider_kind="local",
                        model="llama3.1",
                        priority=1,
                        enabled=True,
                    ),
                ],
            ),
        }
        
        tier_configs = build_tier_configs_from_settings(settings)
        
        assert len(tier_configs) == 1
        assert tier_configs[0].tier == "simple"

    def test_filters_disabled_providers(self):
        """Test disabled providers are filtered out."""
        from app.core.config import Settings, ProviderEntry, TierModelsConfig
        
        settings = Settings()
        settings.models = {
            "simple": TierModelsConfig(
                enabled=True,
                routing_strategy="cost_optimized",
                providers=[
                    ProviderEntry(
                        name="enabled",
                        provider="ollama",
                        provider_kind="local",
                        model="llama3.2",
                        priority=1,
                        enabled=True,
                    ),
                    ProviderEntry(
                        name="disabled",
                        provider="ollama",
                        provider_kind="local",
                        model="llama3.1",
                        priority=2,
                        enabled=False,
                    ),
                ],
            ),
        }
        
        tier_configs = build_tier_configs_from_settings(settings)
        
        assert len(tier_configs) == 1
        assert len(tier_configs[0].providers) == 1
        assert tier_configs[0].providers[0].name == "enabled"

    def test_sorts_tiers_by_min_provider_priority(self):
        """Test tiers are sorted by minimum provider priority."""
        from app.core.config import Settings, ProviderEntry, TierModelsConfig
        
        settings = Settings()
        settings.models = {
            "complex": TierModelsConfig(
                enabled=True,
                routing_strategy="quality_optimized",
                providers=[
                    ProviderEntry(
                        name="complex-provider",
                        provider="ollama",
                        provider_kind="local",
                        model="llama3.1",
                        priority=10,  # Higher priority number = lower priority
                        enabled=True,
                    ),
                ],
            ),
            "simple": TierModelsConfig(
                enabled=True,
                routing_strategy="cost_optimized",
                providers=[
                    ProviderEntry(
                        name="simple-provider",
                        provider="ollama",
                        provider_kind="local",
                        model="llama3.2",
                        priority=1,  # Lower priority number = higher priority
                        enabled=True,
                    ),
                ],
            ),
        }
        
        tier_configs = build_tier_configs_from_settings(settings)
        
        # simple should come first because its provider has priority 1
        assert tier_configs[0].tier == "simple"
        assert tier_configs[1].tier == "complex"

    def test_sorts_providers_by_priority_within_tier(self):
        """Test providers are sorted by priority within each tier."""
        from app.core.config import Settings, ProviderEntry, TierModelsConfig
        
        settings = Settings()
        settings.models = {
            "simple": TierModelsConfig(
                enabled=True,
                routing_strategy="cost_optimized",
                providers=[
                    ProviderEntry(
                        name="low-priority",
                        provider="ollama",
                        provider_kind="local",
                        model="llama3.1",
                        priority=10,
                        enabled=True,
                    ),
                    ProviderEntry(
                        name="high-priority",
                        provider="ollama",
                        provider_kind="local",
                        model="llama3.2",
                        priority=1,
                        enabled=True,
                    ),
                ],
            ),
        }
        
        tier_configs = build_tier_configs_from_settings(settings)
        
        providers = tier_configs[0].providers
        assert providers[0].name == "high-priority"
        assert providers[1].name == "low-priority"

    def test_fallback_tier_preserved(self):
        """Test fallback_tier is preserved from settings."""
        from app.core.config import Settings, ProviderEntry, TierModelsConfig
        
        settings = Settings()
        settings.models = {
            "simple": TierModelsConfig(
                enabled=True,
                routing_strategy="cost_optimized",
                fallback_tier="complex",
                providers=[
                    ProviderEntry(
                        name="test",
                        provider="ollama",
                        provider_kind="local",
                        model="llama3.2",
                        priority=1,
                        enabled=True,
                    ),
                ],
            ),
            "complex": TierModelsConfig(
                enabled=True,
                routing_strategy="quality_optimized",
                fallback_tier=None,
                providers=[
                    ProviderEntry(
                        name="test2",
                        provider="ollama",
                        provider_kind="local",
                        model="llama3.1",
                        priority=1,
                        enabled=True,
                    ),
                ],
            ),
        }
        
        tier_configs = build_tier_configs_from_settings(settings)
        
        simple_tier = next(t for t in tier_configs if t.tier == "simple")
        complex_tier = next(t for t in tier_configs if t.tier == "complex")
        
        assert simple_tier.fallback_tier == "complex"
        assert complex_tier.fallback_tier is None

    def test_fallback_to_legacy_when_no_models(self):
        """Test falls back to legacy config when settings.models is empty."""
        from app.core.config import Settings
        
        settings = Settings()
        settings.models = {}
        settings.local_provider_models = ["llama3.2"]
        settings.cloud_provider_models = ["nvidia/nemotron-3-nano-30b-a3b"]
        
        tier_configs = build_tier_configs_from_settings(settings)
        
        # Should still return two tiers from legacy config
        assert len(tier_configs) == 2
        tier_names = [t.tier for t in tier_configs]
        assert "simple" in tier_names
        assert "complex" in tier_names


class TestTierConfigIntegration:
    """Integration tests for tier config with classifier/selector."""

    def test_multiple_providers_per_tier(self):
        """Test multiple providers can be configured per tier."""
        tiers = get_default_tier_config(
            local_models=["llama3.2", "llama3.1", "mistral:7b"],
            cloud_models=["nvidia/nemotron-3-nano-30b-a3b", "nvidia/llama-3.1-nemotron-70b-instruct"],
        )
        simple_tier = next(t for t in tiers if t.tier == "simple")
        complex_tier = next(t for t in tiers if t.tier == "complex")
        # Simple tier should have at least 2 providers (cloud + local)
        assert len(simple_tier.providers) >= 2
        # Complex tier should have at least 1 provider
        assert len(complex_tier.providers) >= 1

    def test_provider_metadata_preserved(self):
        """Test provider metadata can be stored."""
        config = ProviderTierConfig(
            name="test",
            provider="ollama",
            provider_kind="local",
            model="test",
            tier="simple",
            metadata={"capability": "code", "size": "7b"},
        )
        assert config.metadata["capability"] == "code"
        assert config.metadata["size"] == "7b"

    def test_tier_metadata_preserved(self):
        """Test tier metadata can be stored."""
        config = TierConfig(
            tier="simple",
            routing_strategy="cost_optimized",
            metadata={"region": "us-east", "gpu": True},
        )
        assert config.metadata["region"] == "us-east"
        assert config.metadata["gpu"] is True


class TestCycleDetection:
    """Tests for cycle detection in tier fallback chains."""

    def test_fallback_chain_no_cycle(self):
        """Test fallback chain without cycles works."""
        tiers = get_default_tier_config()
        simple_tier = next(t for t in tiers if t.tier == "simple")
        complex_tier = next(t for t in tiers if t.tier == "complex")
        
        # simple -> complex -> None (no cycle)
        assert simple_tier.fallback_tier == "complex"
        assert complex_tier.fallback_tier is None

    def test_cycle_detection_in_selector(self):
        """Test cycle detection logic (tested in selector tests)."""
        # This is tested in test_router.py with ProviderSelector._failover
        pass

    def test_legacy_model_names_preserved(self):
        """Test legacy model names are preserved in config."""
        tiers = get_default_tier_config(
            local_models=["llama3.2"],
            cloud_models=["gemini-1.5-flash"],
        )
        local_tier = next(t for t in tiers if t.tier == Tier.LOCAL)
        cloud_tier = next(t for t in tiers if t.tier == Tier.CLOUD)
        assert local_tier.models[0].model == "llama3.2"
        assert cloud_tier.models[0].model == "gemini-1.5-flash"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])