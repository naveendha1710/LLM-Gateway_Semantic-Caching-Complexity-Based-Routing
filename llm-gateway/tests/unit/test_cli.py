"""Unit tests for CLI command parsing and output formatting."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from app.cli.main import (
    _render_benchmark_results,
    _render_cache_inspect_table,
    _render_load_test_results,
    _render_stats_table,
    app,
    format_metadata_line,
)


class TestCLICommands:
    """Test CLI command parsing and basic structure."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_chat_command_help(self):
        """Test chat command help output."""
        result = self.runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "Send a chat message" in result.output
        assert "--url" in result.output
        assert "--model" in result.output
        assert "--temperature" in result.output
        assert "--max-tokens" in result.output
        assert "--bypass-cache" in result.output

    def test_stats_command_help(self):
        """Test stats command help output."""
        result = self.runner.invoke(app, ["stats", "--help"])
        assert result.exit_code == 0
        assert "Show aggregate gateway statistics" in result.output
        assert "--url" in result.output

    def test_cache_inspect_command_help(self):
        """Test cache-inspect command help output."""
        result = self.runner.invoke(app, ["cache-inspect", "--help"])
        assert result.exit_code == 0
        assert "Inspect cache" in result.output
        assert "--url" in result.output
        assert "--top" in result.output

    def test_health_command_help(self):
        """Test health command help output."""
        result = self.runner.invoke(app, ["health", "--help"])
        assert result.exit_code == 0
        assert "Show gateway and per-provider health" in result.output
        assert "--url" in result.output

    def test_benchmark_command_help(self):
        """Test benchmark command help output."""
        result = self.runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "Run latency/throughput/cache benchmark" in result.output
        assert "--requests" in result.output
        assert "--concurrency" in result.output
        assert "--warmup" in result.output
        assert "--model" in result.output
        assert "--output" in result.output

    def test_load_test_command_help(self):
        """Test load-test command help output."""
        result = self.runner.invoke(app, ["load-test", "--help"])
        assert result.exit_code == 0
        assert "Run sustained load test" in result.output
        assert "--mode" in result.output
        assert "--rate" in result.output
        assert "--duration" in result.output

    def test_chaos_command_help(self):
        """Test chaos command help output."""
        result = self.runner.invoke(app, ["chaos", "--help"])
        assert result.exit_code == 0
        assert "Run chaos/resilience tests" in result.output
        assert "--scenario" in result.output


class TestFormatMetadataLine:
    """Test the format_metadata_line function."""

    def test_format_cache_hit(self):
        """Test formatting a cache hit response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "source": "cache",
            "degraded": False,
            "routing_strategy": "cache_hit",
            "model": "cached-model",
            "usage": {"total_tokens": 50},
        }

        result = format_metadata_line(mock_response, 15.5)
        assert "source=cache" in str(result)
        assert "tier=cache_hit" in str(result)
        assert "latency=16ms" in str(result)
        assert "model=cached-model" in str(result)
        assert "tokens=50" in str(result)

    def test_format_local_provider(self):
        """Test formatting a local provider response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "source": "local",
            "degraded": False,
            "routing_strategy": "local",
            "model": "local-llama",
            "usage": {"total_tokens": 100},
        }

        result = format_metadata_line(mock_response, 123.7)
        assert "source=local" in str(result)
        assert "tier=local" in str(result)
        assert "latency=124ms" in str(result)
        assert "model=local-llama" in str(result)

    def test_format_cloud_provider(self):
        """Test formatting a cloud provider response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "source": "cloud",
            "degraded": False,
            "routing_strategy": "cloud",
            "model": "gpt-4",
            "usage": {"total_tokens": 200},
        }

        result = format_metadata_line(mock_response, 500.2)
        assert "source=cloud" in str(result)
        assert "tier=cloud" in str(result)
        assert "latency=500ms" in str(result)
        assert "model=gpt-4" in str(result)

    def test_format_degraded_response(self):
        """Test formatting a degraded response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "source": "local",
            "degraded": True,
            "routing_strategy": "local",
            "model": "local-llama",
            "usage": {"total_tokens": 50},
        }

        result = format_metadata_line(mock_response, 100.0)
        assert "DEGRADED" in str(result)

    def test_format_unknown_source(self):
        """Test formatting with unknown source."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "source": "unknown",
            "degraded": False,
            "routing_strategy": "unknown",
            "model": "unknown",
            "usage": {},
        }

        result = format_metadata_line(mock_response, 50.0)
        assert "source=unknown" in str(result)


class TestRenderStatsTable:
    """Test the _render_stats_table function."""

    def test_render_stats_table_complete(self):
        """Test rendering stats table with all fields."""
        data = {
            "cache": {"hit_rate": 75.5, "hits": 75, "misses": 25},
            "latency": {
                "avg_ms": 150.5,
                "p50_ms": 120.0,
                "p95_ms": 300.0,
                "p99_ms": 500.0,
            },
            "cost": {"total_usd": 0.025, "per_1k_tokens_usd": 0.002},
            "providers": {
                "local": {"requests": 50, "errors": 2},
                "cloud": {"requests": 50, "errors": 1},
            },
        }

        console = Console(record=True)
        _render_stats_table(data)
        # Just verify it doesn't crash - output is captured by console

    def test_render_stats_table_empty(self):
        """Test rendering stats table with minimal data."""
        data = {}

        console = Console(record=True)
        _render_stats_table(data)
        # Just verify it doesn't crash


class TestRenderCacheInspectTable:
    """Test the _render_cache_inspect_table function."""

    def test_render_cache_inspect_with_results(self):
        """Test rendering cache inspect with results."""
        data = {
            "results": [
                {
                    "score": 0.95,
                    "prompt": "What is the capital of France?",
                    "age_seconds": 3600,
                    "source": "local",
                },
                {
                    "score": 0.87,
                    "prompt": "What is the capital of Germany?",
                    "age_seconds": 7200,
                    "source": "cloud",
                },
            ]
        }

        console = Console(record=True)
        _render_cache_inspect_table(data, "capital of France")
        # Just verify it doesn't crash

    def test_render_cache_inspect_empty(self):
        """Test rendering cache inspect with no results."""
        data = {"results": []}

        console = Console(record=True)
        _render_cache_inspect_table(data, "nonexistent query")
        # Just verify it doesn't crash


class TestRenderBenchmarkResults:
    """Test the _render_benchmark_results function."""

    def test_render_benchmark_results(self):
        """Test rendering benchmark results."""
        results = {
            "summary": {
                "total_requests": 100,
                "successful": 95,
                "failed": 5,
                "cache_hit_rate": 60.0,
                "avg_latency_ms": 150.5,
                "p50_latency_ms": 120.0,
                "p95_latency_ms": 300.0,
                "p99_latency_ms": 500.0,
                "throughput_rps": 50.0,
            },
            "benchmarks": [
                {
                    "name": "warmup",
                    "requests": 10,
                    "cache_hit_rate": 0.0,
                    "avg_latency_ms": 200.0,
                    "p95_latency_ms": 400.0,
                    "throughput_rps": 25.0,
                },
                {
                    "name": "benchmark",
                    "requests": 90,
                    "cache_hit_rate": 66.7,
                    "avg_latency_ms": 145.0,
                    "p95_latency_ms": 280.0,
                    "throughput_rps": 55.0,
                },
            ],
        }

        console = Console(record=True)
        _render_benchmark_results(results)
        # Just verify it doesn't crash


class TestRenderLoadTestResults:
    """Test the _render_load_test_results function."""

    def test_render_load_test_results(self):
        """Test rendering load test results."""
        results = {
            "total_requests": 1000,
            "successful": 980,
            "failed": 20,
            "avg_latency_ms": 125.5,
            "p95_latency_ms": 250.0,
            "throughput_rps": 80.0,
            "error_rate": 0.02,
        }

        console = Console(record=True)
        _render_load_test_results(results)
        # Just verify it doesn't crash


class TestRenderChaosResults:
    """Test the _render_chaos_results function."""

    def test_render_chaos_results(self):
        """Test rendering chaos test results."""
        from app.cli.main import _render_chaos_results

        results = {
            "provider_failure": {"passed": True, "details": "Failover worked"},
            "cache_fail_open": {"passed": True, "details": "Cache failed open"},
            "network_latency": {"passed": False, "details": "Timeout exceeded"},
        }

        console = Console(record=True)
        _render_chaos_results(results)
        # Just verify it doesn't crash


class TestGatewayClient:
    """Test the GatewayClient class."""

    @pytest.mark.asyncio
    async def test_chat_request(self):
        """Test chat request method."""
        from app.cli.main import GatewayClient

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Hello!"}}],
                "source": "local",
                "degraded": False,
                "model": "test-model",
                "usage": {"total_tokens": 10},
            }
            mock_client.request.return_value = mock_response

            async with GatewayClient("http://localhost:8000") as client:
                response = await client.chat("Hello", model="test-model")

            assert response.is_success
            mock_client.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_request(self):
        """Test health request method."""
        from app.cli.main import GatewayClient

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_client.request.return_value = mock_response

            async with GatewayClient("http://localhost:8000") as client:
                response = await client.health()

            assert response.is_success
            mock_client.request.assert_called_with("GET", "http://localhost:8000/health")

    @pytest.mark.asyncio
    async def test_stats_request(self):
        """Test stats request method."""
        from app.cli.main import GatewayClient

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_client.request.return_value = mock_response

            async with GatewayClient("http://localhost:8000") as client:
                response = await client.stats()

            assert response.is_success
            mock_client.request.assert_called_with("GET", "http://localhost:8000/v1/stats")

    @pytest.mark.asyncio
    async def test_cache_inspect_request(self):
        """Test cache inspect request method."""
        from app.cli.main import GatewayClient

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_client.request.return_value = mock_response

            async with GatewayClient("http://localhost:8000") as client:
                response = await client.cache_inspect("test query", top_n=3)

            assert response.is_success
            mock_client.request.assert_called_with(
                "POST", "http://localhost:8000/v1/cache/inspect", json={"query": "test query", "top_n": 3}
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])