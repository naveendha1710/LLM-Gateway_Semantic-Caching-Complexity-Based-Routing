#!/usr/bin/env python3
"""
Benchmark script for LLM Gateway.

This script runs performance benchmarks against the LLM Gateway,
measuring latency, throughput, and cache hit rates.
"""

import asyncio
import argparse
import json
import statistics
import time
from dataclasses import dataclass, asdict
from typing import List, Optional
import httpx
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import get_settings


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_seconds: float
    requests_per_second: float
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    cache_hit_rate: float
    errors: List[str]


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    latency_ms: float
    success: bool
    cache_hit: bool
    error: Optional[str] = None


class BenchmarkRunner:
    """Runs benchmarks against the LLM Gateway."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    async def health_check(self) -> bool:
        """Check if the gateway is healthy."""
        try:
            response = await self.client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def send_request(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 100,
    ) -> RequestMetrics:
        """Send a single chat completion request."""
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model:
            payload["model"] = model

        start = time.perf_counter()
        try:
            response = await self.client.post("/v1/chat/completions", json=payload)
            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                # Check for cache hit header
                cache_hit = response.headers.get("X-Cache-Hit", "false").lower() == "true"
                return RequestMetrics(
                    latency_ms=latency_ms,
                    success=True,
                    cache_hit=cache_hit,
                )
            else:
                return RequestMetrics(
                    latency_ms=latency_ms,
                    success=False,
                    cache_hit=False,
                    error=f"HTTP {response.status_code}: {response.text}",
                )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return RequestMetrics(
                latency_ms=latency_ms,
                success=False,
                cache_hit=False,
                error=str(e),
            )

    async def run_benchmark(
        self,
        name: str,
        num_requests: int,
        concurrency: int,
        messages: List[dict],
        model: Optional[str] = None,
        warmup_requests: int = 0,
    ) -> BenchmarkResult:
        """Run a benchmark with specified parameters."""
        print(f"\n{'='*60}")
        print(f"Running benchmark: {name}")
        print(f"Requests: {num_requests}, Concurrency: {concurrency}")
        if warmup_requests:
            print(f"Warmup requests: {warmup_requests}")
        print(f"{'='*60}")

        # Warmup
        if warmup_requests > 0:
            print("Warming up...")
            semaphore = asyncio.Semaphore(concurrency)

            async def warmup_request():
                async with semaphore:
                    await self.send_request(messages, model)

            await asyncio.gather(*[warmup_request() for _ in range(warmup_requests)])
            print("Warmup complete.")

        # Actual benchmark
        print("Running benchmark...")
        semaphore = asyncio.Semaphore(concurrency)
        metrics: List[RequestMetrics] = []

        async def benchmark_request():
            async with semaphore:
                metric = await self.send_request(messages, model)
                metrics.append(metric)
                return metric

        start_time = time.perf_counter()
        await asyncio.gather(*[benchmark_request() for _ in range(num_requests)])
        total_time = time.perf_counter() - start_time

        # Calculate results
        successful = [m for m in metrics if m.success]
        failed = [m for m in metrics if not m.success]
        latencies = [m.latency_ms for m in successful]
        cache_hits = sum(1 for m in successful if m.cache_hit)

        if latencies:
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
            p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
        else:
            avg_latency = median_latency = p95_latency = p99_latency = min_latency = max_latency = 0.0

        cache_hit_rate = cache_hits / len(successful) if successful else 0.0
        rps = num_requests / total_time if total_time > 0 else 0.0

        errors = [m.error for m in failed if m.error]

        result = BenchmarkResult(
            name=name,
            total_requests=num_requests,
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_time_seconds=total_time,
            requests_per_second=rps,
            avg_latency_ms=avg_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            cache_hit_rate=cache_hit_rate,
            errors=errors[:10],  # Limit errors shown
        )

        self._print_result(result)
        return result

    def _print_result(self, result: BenchmarkResult):
        """Print benchmark results."""
        print(f"\nResults for {result.name}:")
        print(f"  Total Requests:     {result.total_requests}")
        print(f"  Successful:         {result.successful_requests}")
        print(f"  Failed:             {result.failed_requests}")
        print(f"  Total Time:         {result.total_time_seconds:.2f}s")
        print(f"  Throughput:         {result.requests_per_second:.2f} req/s")
        print(f"  Avg Latency:        {result.avg_latency_ms:.2f}ms")
        print(f"  Median Latency:     {result.median_latency_ms:.2f}ms")
        print(f"  P95 Latency:        {result.p95_latency_ms:.2f}ms")
        print(f"  P99 Latency:        {result.p99_latency_ms:.2f}ms")
        print(f"  Min Latency:        {result.min_latency_ms:.2f}ms")
        print(f"  Max Latency:        {result.max_latency_ms:.2f}ms")
        print(f"  Cache Hit Rate:     {result.cache_hit_rate:.2%}")
        if result.errors:
            print(f"  Errors (first 10):  {result.errors}")


async def main():
    parser = argparse.ArgumentParser(description="LLM Gateway Benchmark")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the LLM Gateway (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for authentication",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of requests per benchmark (default: 100)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup requests (default: 10)",
    )
    parser.add_argument(
        "--model",
        help="Model to use for requests",
    )
    parser.add_argument(
        "--output",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (fewer requests)",
    )
    args = parser.parse_args()

    if args.quick:
        args.requests = 20
        args.concurrency = 5
        args.warmup = 5

    # Test messages
    test_messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "user", "content": "Explain quantum computing in simple terms."},
        {"role": "user", "content": "Write a short poem about coding."},
        {"role": "user", "content": "How does a neural network work?"},
        {"role": "user", "content": "What are the benefits of caching?"},
    ]

    async with BenchmarkRunner(args.url, args.api_key) as runner:
        # Health check
        print("Checking gateway health...")
        healthy = await runner.health_check()
        if not healthy:
            print("ERROR: Gateway is not healthy!")
            sys.exit(1)
        print("Gateway is healthy.")

        results = []

        # Benchmark 1: Basic latency test
        result = await runner.run_benchmark(
            name="Basic Latency Test",
            num_requests=args.requests,
            concurrency=args.concurrency,
            messages=[test_messages[0]],
            model=args.model,
            warmup_requests=args.warmup,
        )
        results.append(result)

        # Benchmark 2: Cache effectiveness test (repeated same request)
        result = await runner.run_benchmark(
            name="Cache Effectiveness Test",
            num_requests=args.requests,
            concurrency=args.concurrency,
            messages=[test_messages[1]],
            model=args.model,
            warmup_requests=0,  # No warmup to test cold cache
        )
        results.append(result)

        # Benchmark 3: Concurrent load test
        result = await runner.run_benchmark(
            name="Concurrent Load Test",
            num_requests=args.requests * 2,
            concurrency=args.concurrency * 2,
            messages=[test_messages[2]],
            model=args.model,
            warmup_requests=args.warmup,
        )
        results.append(result)

        # Benchmark 4: Mixed workload test
        async def mixed_workload():
            semaphore = asyncio.Semaphore(args.concurrency)
            metrics: List[RequestMetrics] = []

            async def send_mixed():
                async with semaphore:
                    msg = test_messages[len(metrics) % len(test_messages)]
                    metric = await runner.send_request([msg], args.model)
                    metrics.append(metric)

            start = time.perf_counter()
            await asyncio.gather(*[send_mixed() for _ in range(args.requests)])
            total_time = time.perf_counter() - start

            successful = [m for m in metrics if m.success]
            latencies = [m.latency_ms for m in successful]
            cache_hits = sum(1 for m in successful if m.cache_hit)

            return BenchmarkResult(
                name="Mixed Workload Test",
                total_requests=args.requests,
                successful_requests=len(successful),
                failed_requests=len(metrics) - len(successful),
                total_time_seconds=total_time,
                requests_per_second=args.requests / total_time if total_time > 0 else 0,
                avg_latency_ms=statistics.mean(latencies) if latencies else 0,
                median_latency_ms=statistics.median(latencies) if latencies else 0,
                p95_latency_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (max(latencies) if latencies else 0),
                p99_latency_ms=statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else (max(latencies) if latencies else 0),
                min_latency_ms=min(latencies) if latencies else 0,
                max_latency_ms=max(latencies) if latencies else 0,
                cache_hit_rate=cache_hits / len(successful) if successful else 0,
                errors=[m.error for m in metrics if not m.success and m.error][:10],
            )

        result = await mixed_workload()
        runner._print_result(result)
        results.append(result)

        # Summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        for r in results:
            print(f"{r.name:30s} | {r.requests_per_second:8.2f} req/s | "
                  f"P95: {r.p95_latency_ms:7.2f}ms | Cache: {r.cache_hit_rate:.1%}")

        # Save results
        if args.output:
            output_data = {
                "timestamp": time.time(),
                "config": {
                    "url": args.url,
                    "requests": args.requests,
                    "concurrency": args.concurrency,
                    "warmup": args.warmup,
                    "model": args.model,
                },
                "results": [asdict(r) for r in results],
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())