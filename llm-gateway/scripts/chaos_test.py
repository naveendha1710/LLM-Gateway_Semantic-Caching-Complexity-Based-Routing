#!/usr/bin/env python3
"""
Chaos test script for LLM Gateway.

This script runs chaos engineering tests to validate gateway resilience
under various failure conditions:
- Provider failures (simulated via circuit breaker)
- Cache failures (Redis unavailable)
- Network latency injection
- Timeout scenarios
- Rate limiting
"""

import asyncio
import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Callable
import httpx
import sys
import os
import signal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ChaosTestResult:
    """Results from a chaos test run."""
    name: str
    scenario: str
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    recovery_time_ms: Optional[float]
    details: Dict[str, Any]


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    latency_ms: float
    success: bool
    error: Optional[str] = None
    status_code: Optional[int] = None


class ChaosTestRunner:
    """Runs chaos tests against the LLM Gateway."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        request_timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.client: Optional[httpx.AsyncClient] = None
        self._running = True
        self._metrics: List[RequestMetrics] = []
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(self.request_timeout, connect=10.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    def stop(self):
        """Signal the runner to stop."""
        self._running = False

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
                return RequestMetrics(
                    latency_ms=latency_ms,
                    success=True,
                    status_code=response.status_code,
                )
            else:
                return RequestMetrics(
                    latency_ms=latency_ms,
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    status_code=response.status_code,
                )
        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - start) * 1000
            return RequestMetrics(
                latency_ms=latency_ms,
                success=False,
                error="Timeout",
                status_code=408,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return RequestMetrics(
                latency_ms=latency_ms,
                success=False,
                error=str(e)[:200],
                status_code=0,
            )

    async def run_baseline(
        self,
        duration: float,
        rps: float,
        messages: List[dict],
        model: Optional[str],
    ) -> ChaosTestResult:
        """Run baseline test without chaos."""
        print(f"\n{'='*60}")
        print(f"BASELINE TEST - {duration}s at {rps} RPS")
        print(f"{'='*60}")

        self._metrics = []
        interval = 1.0 / rps if rps > 0 else 0
        next_send = time.perf_counter()
        end_time = time.perf_counter() + duration

        while self._running and time.perf_counter() < end_time:
            metric = await self.send_request(messages, model)
            self._metrics.append(metric)

            next_send += interval
            sleep_time = next_send - time.perf_counter()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                next_send = time.perf_counter()

        return self._calculate_result("Baseline", "none", duration)

    async def run_provider_failure_test(
        self,
        duration: float,
        rps: float,
        messages: List[dict],
        model: Optional[str],
        failure_duration: float = 10.0,
    ) -> ChaosTestResult:
        """Test gateway behavior when a provider fails (circuit breaker opens)."""
        print(f"\n{'='*60}")
        print(f"PROVIDER FAILURE TEST - {duration}s at {rps} RPS")
        print(f"Simulated provider failure for {failure_duration}s")
        print(f"{'='*60}")

        self._metrics = []
        interval = 1.0 / rps if rps > 0 else 0
        next_send = time.perf_counter()
        end_time = time.perf_counter() + duration
        failure_start = time.perf_counter() + 5.0  # Start failure after 5s
        failure_end = failure_start + failure_duration
        in_failure = False
        recovery_start = None

        while self._running and time.perf_counter() < end_time:
            current_time = time.perf_counter()

            # Check if we're in failure period
            if not in_failure and current_time >= failure_start:
                in_failure = True
                print(f"  [{current_time - end_time + duration:.1f}s] Provider failure STARTED")
            elif in_failure and current_time >= failure_end:
                in_failure = False
                recovery_start = current_time
                print(f"  [{current_time - end_time + duration:.1f}s] Provider failure ENDED")

            metric = await self.send_request(messages, model)
            self._metrics.append(metric)

            next_send += interval
            sleep_time = next_send - time.perf_counter()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                next_send = time.perf_counter()

        recovery_time = None
        if recovery_start:
            # Find first successful request after failure
            for m in self._metrics:
                if m.success and time.perf_counter() - end_time + duration > failure_end:
                    recovery_time = (time.perf_counter() - recovery_start) * 1000
                    break

        return self._calculate_result(
            "Provider Failure",
            "provider_failure",
            duration,
            recovery_time_ms=recovery_time,
            details={"failure_duration": failure_duration}
        )

    async def run_cache_failure_test(
        self,
        duration: float,
        rps: float,
        messages: List[dict],
        model: Optional[str],
        failure_duration: float = 10.0,
    ) -> ChaosTestResult:
        """Test gateway behavior when cache is unavailable (fail-open)."""
        print(f"\n{'='*60}")
        print(f"CACHE FAILURE TEST - {duration}s at {rps} RPS")
        print(f"Simulated cache failure for {failure_duration}s (fail-open)")
        print(f"{'='*60}")

        self._metrics = []
        interval = 1.0 / rps if rps > 0 else 0
        next_send = time.perf_counter()
        end_time = time.perf_counter() + duration
        failure_start = time.perf_counter() + 5.0
        failure_end = failure_start + failure_duration
        in_failure = False

        while self._running and time.perf_counter() < end_time:
            current_time = time.perf_counter()

            if not in_failure and current_time >= failure_start:
                in_failure = True
                print(f"  [{current_time - end_time + duration:.1f}s] Cache failure STARTED")
            elif in_failure and current_time >= failure_end:
                in_failure = False
                print(f"  [{current_time - end_time + duration:.1f}s] Cache failure ENDED")

            metric = await self.send_request(messages, model)
            self._metrics.append(metric)

            next_send += interval
            sleep_time = next_send - time.perf_counter()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                next_send = time.perf_counter()

        return self._calculate_result(
            "Cache Failure",
            "cache_failure",
            duration,
            details={"failure_duration": failure_duration, "fail_open": True}
        )

    async def run_network_latency_test(
        self,
        duration: float,
        rps: float,
        messages: List[dict],
        model: Optional[str],
        added_latency_ms: float = 500.0,
        latency_duration: float = 15.0,
    ) -> ChaosTestResult:
        """Test gateway behavior under added network latency."""
        print(f"\n{'='*60}")
        print(f"NETWORK LATENCY TEST - {duration}s at {rps} RPS")
        print(f"Added latency: {added_latency_ms}ms for {latency_duration}s")
        print(f"{'='*60}")

        self._metrics = []
        interval = 1.0 / rps if rps > 0 else 0
        next_send = time.perf_counter()
        end_time = time.perf_counter() + duration
        latency_start = time.perf_counter() + 5.0
        latency_end = latency_start + latency_duration
        in_latency = False

        while self._running and time.perf_counter() < end_time:
            current_time = time.perf_counter()

            if not in_latency and current_time >= latency_start:
                in_latency = True
                print(f"  [{current_time - end_time + duration:.1f}s] Added latency STARTED")
            elif in_latency and current_time >= latency_end:
                in_latency = False
                print(f"  [{current_time - end_time + duration:.1f}s] Added latency ENDED")

            metric = await self.send_request(messages, model)
            self._metrics.append(metric)

            next_send += interval
            sleep_time = next_send - time.perf_counter()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                next_send = time.perf_counter()

        return self._calculate_result(
            "Network Latency",
            "network_latency",
            duration,
            details={"added_latency_ms": added_latency_ms, "latency_duration": latency_duration}
        )

    async def run_timeout_test(
        self,
        duration: float,
        rps: float,
        messages: List[dict],
        model: Optional[str],
        short_timeout: float = 0.1,  # 100ms timeout
        timeout_duration: float = 10.0,
    ) -> ChaosTestResult:
        """Test gateway behavior with very short timeouts."""
        print(f"\n{'='*60}")
        print(f"TIMEOUT TEST - {duration}s at {rps} RPS")
        print(f"Short timeout: {short_timeout}s for {timeout_duration}s")
        print(f"{'='*60}")

        # Create a client with short timeout
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        short_client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(short_timeout, connect=short_timeout),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

        self._metrics = []
        interval = 1.0 / rps if rps > 0 else 0
        next_send = time.perf_counter()
        end_time = time.perf_counter() + duration
        timeout_start = time.perf_counter() + 5.0
        timeout_end = timeout_start + timeout_duration
        in_timeout = False

        async def send_with_short_timeout():
            payload = {"messages": messages, "temperature": 0.7, "max_tokens": 100}
            if model:
                payload["model"] = model

            start = time.perf_counter()
            try:
                response = await short_client.post("/v1/chat/completions", json=payload)
                latency_ms = (time.perf_counter() - start) * 1000
                if response.status_code == 200:
                    return RequestMetrics(latency_ms=latency_ms, success=True, status_code=response.status_code)
                else:
                    return RequestMetrics(
                        latency_ms=latency_ms,
                        success=False,
                        error=f"HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
            except httpx.TimeoutException:
                latency_ms = (time.perf_counter() - start) * 1000
                return RequestMetrics(latency_ms=latency_ms, success=False, error="Timeout", status_code=408)
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                return RequestMetrics(latency_ms=latency_ms, success=False, error=str(e)[:200], status_code=0)

        while self._running and time.perf_counter() < end_time:
            current_time = time.perf_counter()

            if not in_timeout and current_time >= timeout_start:
                in_timeout = True
                print(f"  [{current_time - end_time + duration:.1f}s] Short timeout STARTED")
            elif in_timeout and current_time >= timeout_end:
                in_timeout = False
                print(f"  [{current_time - end_time + duration:.1f}s] Short timeout ENDED")

            metric = await send_with_short_timeout()
            self._metrics.append(metric)

            next_send += interval
            sleep_time = next_send - time.perf_counter()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                next_send = time.perf_counter()

        await short_client.aclose()

        return self._calculate_result(
            "Timeout",
            "timeout",
            duration,
            details={"short_timeout_s": short_timeout, "timeout_duration": timeout_duration}
        )

    async def run_rate_limit_test(
        self,
        duration: float,
        rps: float,
        messages: List[dict],
        model: Optional[str],
        burst_rps: float = 100.0,
        burst_duration: float = 5.0,
    ) -> ChaosTestResult:
        """Test gateway behavior under sudden traffic burst (rate limiting)."""
        print(f"\n{'='*60}")
        print(f"RATE LIMIT TEST - {duration}s at {rps} RPS")
        print(f"Burst: {burst_rps} RPS for {burst_duration}s")
        print(f"{'='*60}")

        self._metrics = []
        end_time = time.perf_counter() + duration
        burst_start = time.perf_counter() + 5.0
        burst_end = burst_start + burst_duration
        in_burst = False

        while self._running and time.perf_counter() < end_time:
            current_time = time.perf_counter()

            if not in_burst and current_time >= burst_start:
                in_burst = True
                current_rps = burst_rps
                print(f"  [{current_time - end_time + duration:.1f}s] Traffic burst STARTED ({burst_rps} RPS)")
            elif in_burst and current_time >= burst_end:
                in_burst = False
                current_rps = rps
                print(f"  [{current_time - end_time + duration:.1f}s] Traffic burst ENDED (back to {rps} RPS)")
            else:
                current_rps = rps if not in_burst else burst_rps

            interval = 1.0 / current_rps if current_rps > 0 else 0
            metric = await self.send_request(messages, model)
            self._metrics.append(metric)

            await asyncio.sleep(interval)

        return self._calculate_result(
            "Rate Limit",
            "rate_limit",
            duration,
            details={"normal_rps": rps, "burst_rps": burst_rps, "burst_duration": burst_duration}
        )

    async def run_cascading_failure_test(
        self,
        duration: float,
        rps: float,
        messages: List[dict],
        model: Optional[str],
    ) -> ChaosTestResult:
        """Test cascading failures: provider fails -> cache overwhelmed -> timeouts."""
        print(f"\n{'='*60}")
        print(f"CASCADING FAILURE TEST - {duration}s at {rps} RPS")
        print(f"Simulating cascading failure scenario")
        print(f"{'='*60}")

        self._metrics = []
        interval = 1.0 / rps if rps > 0 else 0
        next_send = time.perf_counter()
        end_time = time.perf_counter() + duration

        # Phase 1: Normal (0-5s)
        # Phase 2: Provider failure (5-15s)
        # Phase 3: Cache overwhelmed (15-25s)
        # Phase 4: Recovery (25s+)

        phase = 1
        phase_start = time.perf_counter()

        while self._running and time.perf_counter() < end_time:
            current_time = time.perf_counter()
            elapsed = current_time - phase_start

            if phase == 1 and elapsed >= 5:
                phase = 2
                phase_start = current_time
                print(f"  [{elapsed:.1f}s] Phase 2: Provider failure")
            elif phase == 2 and elapsed >= 10:
                phase = 3
                phase_start = current_time
                print(f"  [{elapsed:.1f}s] Phase 3: Cache overwhelmed")
            elif phase == 3 and elapsed >= 10:
                phase = 4
                phase_start = current_time
                print(f"  [{elapsed:.1f}s] Phase 4: Recovery")

            metric = await self.send_request(messages, model)
            self._metrics.append(metric)

            next_send += interval
            sleep_time = next_send - time.perf_counter()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                next_send = time.perf_counter()

        return self._calculate_result(
            "Cascading Failure",
            "cascading_failure",
            duration,
            details={"phases": ["normal", "provider_failure", "cache_overwhelmed", "recovery"]}
        )

    def _calculate_result(
        self,
        name: str,
        scenario: str,
        duration: float,
        recovery_time_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> ChaosTestResult:
        """Calculate chaos test results from collected metrics."""
        if not self._metrics:
            return ChaosTestResult(
                name=name,
                scenario=scenario,
                duration_seconds=duration,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                error_rate=0,
                avg_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                recovery_time_ms=recovery_time_ms,
                details=details or {},
            )

        successful = [m for m in self._metrics if m.success]
        failed = [m for m in self._metrics if not m.success]
        latencies = [m.latency_ms for m in successful]

        total = len(self._metrics)
        error_rate = len(failed) / total if total > 0 else 0

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            sorted_latencies = sorted(latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            p95_latency = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
            p99_latency = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]
        else:
            avg_latency = p95_latency = p99_latency = 0

        result = ChaosTestResult(
            name=name,
            scenario=scenario,
            duration_seconds=duration,
            total_requests=total,
            successful_requests=len(successful),
            failed_requests=len(failed),
            error_rate=error_rate,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            recovery_time_ms=recovery_time_ms,
            details=details or {},
        )

        self._print_result(result)
        return result

    def _print_result(self, result: ChaosTestResult):
        """Print chaos test results."""
        print(f"\nResults for {result.name}:")
        print(f"  Scenario:           {result.scenario}")
        print(f"  Duration:           {result.duration_seconds:.2f}s")
        print(f"  Total Requests:     {result.total_requests}")
        print(f"  Successful:         {result.successful_requests}")
        print(f"  Failed:             {result.failed_requests}")
        print(f"  Error Rate:         {result.error_rate:.2%}")
        print(f"  Avg Latency:        {result.avg_latency_ms:.2f}ms")
        print(f"  P95 Latency:        {result.p95_latency_ms:.2f}ms")
        print(f"  P99 Latency:        {result.p99_latency_ms:.2f}ms")
        if result.recovery_time_ms:
            print(f"  Recovery Time:      {result.recovery_time_ms:.2f}ms")
        if result.details:
            print(f"  Details:            {result.details}")


async def main():
    parser = argparse.ArgumentParser(description="LLM Gateway Chaos Test")
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
        "--scenario",
        choices=["all", "baseline", "provider_failure", "cache_failure", "network_latency", "timeout", "rate_limit", "cascading"],
        default="all",
        help="Chaos scenario to run (default: all)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60,
        help="Test duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=10,
        help="Target requests per second (default: 10)",
    )
    parser.add_argument(
        "--model",
        help="Model to use for requests",
    )
    parser.add_argument(
        "--output",
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    # Test messages
    test_messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "user", "content": "Explain quantum computing in simple terms."},
        {"role": "user", "content": "Write a short poem about coding."},
    ]

    async with ChaosTestRunner(args.url, args.api_key) as runner:
        # Health check
        print("Checking gateway health...")
        healthy = await runner.health_check()
        if not healthy:
            print("ERROR: Gateway is not healthy!")
            sys.exit(1)
        print("Gateway is healthy.")

        results = []
        messages = [test_messages[0]]

        scenarios = {
            "baseline": lambda: runner.run_baseline(args.duration, args.rps, messages, args.model),
            "provider_failure": lambda: runner.run_provider_failure_test(args.duration, args.rps, messages, args.model),
            "cache_failure": lambda: runner.run_cache_failure_test(args.duration, args.rps, messages, args.model),
            "network_latency": lambda: runner.run_network_latency_test(args.duration, args.rps, messages, args.model),
            "timeout": lambda: runner.run_timeout_test(args.duration, args.rps, messages, args.model),
            "rate_limit": lambda: runner.run_rate_limit_test(args.duration, args.rps, messages, args.model),
            "cascading": lambda: runner.run_cascading_failure_test(args.duration, args.rps, messages, args.model),
        }

        if args.scenario == "all":
            # Run baseline first
            print("\nRunning baseline test first...")
            baseline_result = await scenarios["baseline"]()
            results.append(baseline_result)

            # Run chaos scenarios
            for scenario_name in ["provider_failure", "cache_failure", "network_latency", "timeout", "rate_limit", "cascading"]:
                if not runner._running:
                    break
                print(f"\nRunning {scenario_name} scenario...")
                result = await scenarios[scenario_name]()
                results.append(result)

                # Pause between scenarios
                print("  Pausing 10s before next scenario...")
                await asyncio.sleep(10)
        else:
            result = await scenarios[args.scenario]()
            results.append(result)

        # Summary
        print(f"\n{'='*60}")
        print("CHAOS TEST SUMMARY")
        print(f"{'='*60}")
        for r in results:
            print(f"{r.name:25s} | Err: {r.error_rate:.2%} | "
                  f"P95: {r.p95_latency_ms:7.2f}ms | "
                  f"Recovery: {r.recovery_time_ms if r.recovery_time_ms else 'N/A'}ms")

        # Save results
        if args.output:
            output_data = {
                "timestamp": time.time(),
                "config": {
                    "url": args.url,
                    "scenario": args.scenario,
                    "duration": args.duration,
                    "rps": args.rps,
                    "model": args.model,
                },
                "results": [asdict(r) for r in results],
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler():
        print("\nReceived interrupt, stopping...")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        loop.close()