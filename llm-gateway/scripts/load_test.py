#!/usr/bin/env python3
"""
Load test script for LLM Gateway.

This script runs sustained load tests to evaluate gateway behavior
under continuous traffic, including stress testing and soak testing.
"""

import asyncio
import argparse
import json
import statistics
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import httpx
import sys
import os
import signal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class LoadTestResult:
    """Results from a load test run."""
    name: str
    duration_seconds: float
    target_rps: float
    actual_rps: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    latency_stddev: float
    cache_hit_rate: float
    timeouts: int
    errors_by_type: Dict[str, int]


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    latency_ms: float
    success: bool
    cache_hit: bool
    error: Optional[str] = None
    status_code: Optional[int] = None


class LoadTestRunner:
    """Runs load tests against the LLM Gateway."""

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
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
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

            cache_hit = response.headers.get("X-Cache-Hit", "false").lower() == "true"

            if response.status_code == 200:
                return RequestMetrics(
                    latency_ms=latency_ms,
                    success=True,
                    cache_hit=cache_hit,
                    status_code=response.status_code,
                )
            else:
                return RequestMetrics(
                    latency_ms=latency_ms,
                    success=False,
                    cache_hit=cache_hit,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    status_code=response.status_code,
                )
        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - start) * 1000
            return RequestMetrics(
                latency_ms=latency_ms,
                success=False,
                cache_hit=False,
                error="Timeout",
                status_code=408,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return RequestMetrics(
                latency_ms=latency_ms,
                success=False,
                cache_hit=False,
                error=str(e)[:200],
                status_code=0,
            )

    async def _worker(
        self,
        worker_id: int,
        messages: List[dict],
        model: Optional[str],
        target_rps: float,
        duration: float,
        results_queue: asyncio.Queue,
    ):
        """Worker that sends requests at target RPS."""
        interval = 1.0 / target_rps if target_rps > 0 else 0
        next_send = time.perf_counter()
        end_time = time.perf_counter() + duration
        request_count = 0

        while self._running and time.perf_counter() < end_time:
            # Send request
            metric = await self.send_request(messages, model)
            await results_queue.put(metric)
            request_count += 1

            # Rate limiting
            next_send += interval
            sleep_time = next_send - time.perf_counter()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                # We're behind schedule, reset next_send
                next_send = time.perf_counter()

        await results_queue.put(None)  # Signal completion

    async def run_load_test(
        self,
        name: str,
        duration_seconds: float,
        target_rps: float,
        num_workers: int,
        messages: List[dict],
        model: Optional[str] = None,
    ) -> LoadTestResult:
        """Run a load test with specified parameters."""
        print(f"\n{'='*60}")
        print(f"Load Test: {name}")
        print(f"Duration: {duration_seconds}s, Target RPS: {target_rps}")
        print(f"Workers: {num_workers}")
        print(f"{'='*60}")

        self._metrics = []
        results_queue: asyncio.Queue = asyncio.Queue()

        # Start workers
        workers = [
            asyncio.create_task(
                self._worker(i, messages, model, target_rps / num_workers, duration_seconds, results_queue)
            )
            for i in range(num_workers)
        ]

        # Collect results
        completed_workers = 0
        start_time = time.perf_counter()

        print("Running load test...")
        last_report = start_time
        request_count = 0

        while completed_workers < num_workers:
            metric = await results_queue.get()
            if metric is None:
                completed_workers += 1
            else:
                self._metrics.append(metric)
                request_count += 1

            # Progress report every 5 seconds
            if time.perf_counter() - last_report >= 5:
                elapsed = time.perf_counter() - start_time
                current_rps = request_count / elapsed if elapsed > 0 else 0
                success_count = sum(1 for m in self._metrics if m.success)
                print(f"  [{elapsed:.0f}s] Requests: {request_count}, "
                      f"RPS: {current_rps:.1f}, Success: {success_count}/{request_count}")
                last_report = time.perf_counter()

        actual_duration = time.perf_counter() - start_time

        # Wait for workers to finish
        await asyncio.gather(*workers, return_exceptions=True)

        return self._calculate_results(name, target_rps, actual_duration)

    def _calculate_results(
        self,
        name: str,
        target_rps: float,
        actual_duration: float,
    ) -> LoadTestResult:
        """Calculate load test results from collected metrics."""
        if not self._metrics:
            return LoadTestResult(
                name=name,
                duration_seconds=actual_duration,
                target_rps=target_rps,
                actual_rps=0,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                error_rate=0,
                avg_latency_ms=0,
                median_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                max_latency_ms=0,
                latency_stddev=0,
                cache_hit_rate=0,
                timeouts=0,
                errors_by_type={},
            )

        successful = [m for m in self._metrics if m.success]
        failed = [m for m in self._metrics if not m.success]
        latencies = [m.latency_ms for m in successful]
        cache_hits = sum(1 for m in successful if m.cache_hit)
        timeouts = sum(1 for m in failed if m.error == "Timeout")

        # Count errors by type
        errors_by_type: Dict[str, int] = {}
        for m in failed:
            error_type = m.error.split(":")[0] if m.error else "Unknown"
            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1

        total = len(self._metrics)
        actual_rps = total / actual_duration if actual_duration > 0 else 0

        if latencies:
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
            p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
            max_latency = max(latencies)
            latency_stddev = statistics.stdev(latencies) if len(latencies) > 1 else 0
        else:
            avg_latency = median_latency = p95_latency = p99_latency = max_latency = latency_stddev = 0

        cache_hit_rate = cache_hits / len(successful) if successful else 0
        error_rate = len(failed) / total if total > 0 else 0

        result = LoadTestResult(
            name=name,
            duration_seconds=actual_duration,
            target_rps=target_rps,
            actual_rps=actual_rps,
            total_requests=total,
            successful_requests=len(successful),
            failed_requests=len(failed),
            error_rate=error_rate,
            avg_latency_ms=avg_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            max_latency_ms=max_latency,
            latency_stddev=latency_stddev,
            cache_hit_rate=cache_hit_rate,
            timeouts=timeouts,
            errors_by_type=errors_by_type,
        )

        self._print_result(result)
        return result

    def _print_result(self, result: LoadTestResult):
        """Print load test results."""
        print(f"\nResults for {result.name}:")
        print(f"  Duration:           {result.duration_seconds:.2f}s")
        print(f"  Target RPS:         {result.target_rps:.2f}")
        print(f"  Actual RPS:         {result.actual_rps:.2f}")
        print(f"  Total Requests:     {result.total_requests}")
        print(f"  Successful:         {result.successful_requests}")
        print(f"  Failed:             {result.failed_requests}")
        print(f"  Error Rate:         {result.error_rate:.2%}")
        print(f"  Timeouts:           {result.timeouts}")
        print(f"  Avg Latency:        {result.avg_latency_ms:.2f}ms")
        print(f"  Median Latency:     {result.median_latency_ms:.2f}ms")
        print(f"  P95 Latency:        {result.p95_latency_ms:.2f}ms")
        print(f"  P99 Latency:        {result.p99_latency_ms:.2f}ms")
        print(f"  Max Latency:        {result.max_latency_ms:.2f}ms")
        print(f"  Latency StdDev:     {result.latency_stddev:.2f}ms")
        print(f"  Cache Hit Rate:     {result.cache_hit_rate:.2%}")
        if result.errors_by_type:
            print(f"  Errors by Type:     {result.errors_by_type}")


async def run_step_load_test(
    runner: LoadTestRunner,
    messages: List[dict],
    model: Optional[str],
    steps: List[tuple],  # (duration, target_rps, num_workers)
) -> List[LoadTestResult]:
    """Run a step load test with increasing RPS."""
    results = []
    for i, (duration, target_rps, num_workers) in enumerate(steps):
        result = await runner.run_load_test(
            name=f"Step Load Test - Step {i+1} ({target_rps} RPS)",
            duration_seconds=duration,
            target_rps=target_rps,
            num_workers=num_workers,
            messages=messages,
            model=model,
        )
        results.append(result)

        # Brief pause between steps
        if i < len(steps) - 1:
            print("  Pausing 5s before next step...")
            await asyncio.sleep(5)

    return results


async def run_soak_test(
    runner: LoadTestRunner,
    messages: List[dict],
    model: Optional[str],
    duration_seconds: float,
    target_rps: float,
    num_workers: int,
) -> LoadTestResult:
    """Run a soak test (sustained load for extended period)."""
    return await runner.run_load_test(
        name=f"Soak Test ({duration_seconds}s at {target_rps} RPS)",
        duration_seconds=duration_seconds,
        target_rps=target_rps,
        num_workers=num_workers,
        messages=messages,
        model=model,
    )


async def run_stress_test(
    runner: LoadTestRunner,
    messages: List[dict],
    model: Optional[str],
    max_rps: float,
    step_duration: float,
    step_increment: float,
    num_workers: int,
) -> List[LoadTestResult]:
    """Run a stress test (gradually increase load until breaking point)."""
    results = []
    current_rps = step_increment

    print(f"\n{'='*60}")
    print(f"STRESS TEST - Finding breaking point")
    print(f"Max RPS: {max_rps}, Step: {step_increment} RPS every {step_duration}s")
    print(f"{'='*60}")

    while current_rps <= max_rps and runner._running:
        result = await runner.run_load_test(
            name=f"Stress Test - {current_rps} RPS",
            duration_seconds=step_duration,
            target_rps=current_rps,
            num_workers=num_workers,
            messages=messages,
            model=model,
        )
        results.append(result)

        # Check if error rate is too high (breaking point)
        if result.error_rate > 0.1:  # 10% error rate
            print(f"\n  Breaking point reached at {current_rps} RPS "
                  f"(error rate: {result.error_rate:.1%})")
            break

        current_rps += step_increment

        # Brief pause
        if current_rps <= max_rps:
            print("  Pausing 3s before next step...")
            await asyncio.sleep(3)

    return results


async def main():
    parser = argparse.ArgumentParser(description="LLM Gateway Load Test")
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
        "--mode",
        choices=["load", "step", "soak", "stress"],
        default="load",
        help="Test mode (default: load)",
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
        "--workers",
        type=int,
        default=5,
        help="Number of worker tasks (default: 5)",
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
        "--max-rps",
        type=float,
        default=100,
        help="Max RPS for stress test (default: 100)",
    )
    parser.add_argument(
        "--step-duration",
        type=float,
        default=30,
        help="Duration per step in seconds (default: 30)",
    )
    parser.add_argument(
        "--step-increment",
        type=float,
        default=10,
        help="RPS increment per step (default: 10)",
    )
    args = parser.parse_args()

    # Test messages
    test_messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "user", "content": "Explain quantum computing in simple terms."},
        {"role": "user", "content": "Write a short poem about coding."},
        {"role": "user", "content": "How does a neural network work?"},
        {"role": "user", "content": "What are the benefits of caching?"},
    ]

    async with LoadTestRunner(args.url, args.api_key) as runner:
        # Health check
        print("Checking gateway health...")
        healthy = await runner.health_check()
        if not healthy:
            print("ERROR: Gateway is not healthy!")
            sys.exit(1)
        print("Gateway is healthy.")

        results = []

        if args.mode == "load":
            result = await runner.run_load_test(
                name="Constant Load Test",
                duration_seconds=args.duration,
                target_rps=args.rps,
                num_workers=args.workers,
                messages=[test_messages[0]],
                model=args.model,
            )
            results.append(result)

        elif args.mode == "step":
            steps = [
                (args.step_duration, args.rps, args.workers),
                (args.step_duration, args.rps * 2, args.workers * 2),
                (args.step_duration, args.rps * 4, args.workers * 2),
                (args.step_duration, args.rps * 2, args.workers),
                (args.step_duration, args.rps, args.workers),
            ]
            results = await run_step_load_test(runner, [test_messages[0]], args.model, steps)

        elif args.mode == "soak":
            result = await run_soak_test(
                runner,
                [test_messages[0]],
                args.model,
                args.duration,
                args.rps,
                args.workers,
            )
            results.append(result)

        elif args.mode == "stress":
            results = await run_stress_test(
                runner,
                [test_messages[0]],
                args.model,
                args.max_rps,
                args.step_duration,
                args.step_increment,
                args.workers,
            )

        # Summary
        print(f"\n{'='*60}")
        print("LOAD TEST SUMMARY")
        print(f"{'='*60}")
        for r in results:
            print(f"{r.name:40s} | {r.actual_rps:8.2f} RPS | "
                  f"Err: {r.error_rate:.2%} | P95: {r.p95_latency_ms:7.2f}ms")

        # Save results
        if args.output:
            output_data = {
                "timestamp": time.time(),
                "config": {
                    "url": args.url,
                    "mode": args.mode,
                    "duration": args.duration,
                    "rps": args.rps,
                    "workers": args.workers,
                    "model": args.model,
                },
                "results": [asdict(r) for r in results],
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler():
        print("\nReceived interrupt, stopping...")
        # The runner will be stopped via the context manager

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        loop.close()