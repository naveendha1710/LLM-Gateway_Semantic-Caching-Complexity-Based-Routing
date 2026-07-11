"""LLM Gateway CLI - makes gateway behavior visible."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="llmgw",
    help="LLM Gateway CLI - makes routing, caching, and costs visible",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()

# Global settings
DEFAULT_URL = "http://localhost:8000"
# Default model from simple tier (highest priority)
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"


class GatewayClient:
    """Async HTTP client for gateway API."""

    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "GatewayClient":
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        if not self._client:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        url = f"{self.base_url}{path}"
        return await self._client.request(method, url, **kwargs)

    async def chat(
        self,
        message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 100,
        bypass_cache: bool = False,
    ) -> httpx.Response:
        payload = {
            "messages": [{"role": "user", "content": message}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "bypass_cache": bypass_cache,
        }
        if model:
            payload["model"] = model
        return await self._request("POST", "/v1/chat/completions", json=payload)

    async def health(self) -> httpx.Response:
        return await self._request("GET", "/health")

    async def readyz(self) -> httpx.Response:
        return await self._request("GET", "/readyz")

    async def providers_health(self) -> httpx.Response:
        return await self._request("GET", "/health/providers")

    async def stats(self) -> httpx.Response:
        return await self._request("GET", "/v1/stats")

    async def cache_inspect(self, query: str, top_n: int = 5) -> httpx.Response:
        return await self._request("POST", "/v1/cache/inspect", json={"query": query, "top_n": top_n})


def format_metadata_line(response: httpx.Response, latency_ms: float) -> Text:
    """Format a rich metadata line from response headers and body."""
    try:
        data = response.json()
    except Exception:
        data = {}

    source = data.get("source", "unknown")
    degraded = data.get("degraded", False)
    routing_strategy = data.get("routing_strategy", "unknown")
    model = data.get("model", "unknown")
    usage = data.get("usage", {})

    # Color by source
    source_colors = {
        "cache": "green",
        "local": "blue",
        "cloud": "yellow",
        "unknown": "white",
    }
    source_color = source_colors.get(source, "white")

    # Build metadata parts
    parts = []

    # Source with color
    source_text = Text(f"source={source}", style=source_color)
    parts.append(source_text)

    # Tier/routing strategy
    parts.append(Text(f" tier={routing_strategy}", style="cyan"))

    # Latency
    parts.append(Text(f" latency={latency_ms:.0f}ms", style="magenta"))

    # Model
    parts.append(Text(f" model={model}", style="dim"))

    # Usage tokens
    if usage:
        total = usage.get("total_tokens", 0)
        parts.append(Text(f" tokens={total}", style="dim"))

    # Degraded flag
    if degraded:
        parts.append(Text(" ⚠ DEGRADED", style="bold red"))

    # Join with separators
    result = Text()
    for i, part in enumerate(parts):
        if i > 0:
            result.append(" |")
        result.append(part)

    return result


def print_response_content(response: httpx.Response) -> None:
    """Print the response content nicely."""
    try:
        data = response.json()
        # Extract content and optional reasoning.
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        # Reasoning is provided as a top‑level field in the response schema.
        reasoning = data.get("reasoning")

        # Render reasoning first, if present.
        if reasoning:
            console.print(
                Panel(
                    reasoning,
                    title="Reasoning",
                    border_style="dim",
                    style="dim italic",
                )
            )

        if content:
            console.print(Panel(content, title="Response", border_style="green"))
        else:
            console.print("[dim]No content in response[/dim]")
    except Exception:
        console.print(f"[dim]Raw: {response.text}[/dim]")


# Default model from the simple tier (highest priority)
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"


@app.command()
def chat(
    message: Optional[str] = typer.Argument(None, help="Message to send (omit for REPL)"),
    url: str = typer.Option(DEFAULT_URL, "--url", "-u", help="Gateway base URL"),
    model: Optional[str] = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Model override"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Temperature"),
    max_tokens: int = typer.Option(100, "--max-tokens", help="Max tokens"),
    bypass_cache: bool = typer.Option(False, "--bypass-cache", help="Skip cache lookup"),
):
    """Send a chat message (or start REPL if no message given)."""
    if message is not None:
        asyncio.run(_chat_once(message, url, model, temperature, max_tokens, bypass_cache))
    else:
        asyncio.run(_chat_repl(url, model, temperature, max_tokens, bypass_cache))


async def _chat_once(
    message: str,
    url: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
    bypass_cache: bool,
) -> None:
    async with GatewayClient(url) as client:
        start = time.perf_counter()
        response = await client.chat(message, model, temperature, max_tokens, bypass_cache)
        latency_ms = (time.perf_counter() - start) * 1000

        if response.is_success:
            print_response_content(response)
            meta = format_metadata_line(response, latency_ms)
            console.print(meta)
        else:
            console.print(f"[red]Error {response.status_code}: {response.text}[/red]")
            raise typer.Exit(1)


async def _chat_repl(
    url: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
    bypass_cache: bool,
) -> None:
    """Interactive REPL with running session summary."""
    console.print(Panel("LLM Gateway REPL — type 'exit' or 'quit' to leave", border_style="blue"))
    console.print("[dim]Commands: /stats (show session summary), /clear (reset summary)[/dim]\n")

    session_stats = {
        "total": 0,
        "cache_hits": 0,
        "total_latency_ms": 0.0,
        "total_tokens": 0,
        "by_source": {"cache": 0, "local": 0, "cloud": 0, "unknown": 0},
    }

    async with GatewayClient(url) as client:
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if user_input.lower() in ("exit", "quit"):
                break

            if user_input == "/stats":
                _print_session_summary(session_stats)
                continue

            if user_input == "/clear":
                session_stats = {k: 0 if isinstance(v, int) else 0.0 for k, v in session_stats.items()}
                session_stats["by_source"] = {"cache": 0, "local": 0, "cloud": 0, "unknown": 0}
                console.print("[dim]Session summary cleared[/dim]")
                continue

            start = time.perf_counter()
            response = await client.chat(user_input, model, temperature, max_tokens, bypass_cache)
            latency_ms = (time.perf_counter() - start) * 1000

            if response.is_success:
                print_response_content(response)
                meta = format_metadata_line(response, latency_ms)
                console.print(meta)

                # Update session stats
                try:
                    data = response.json()
                    source = data.get("source", "unknown")
                    usage = data.get("usage", {})
                    session_stats["total"] += 1
                    session_stats["total_latency_ms"] += latency_ms
                    session_stats["total_tokens"] += usage.get("total_tokens", 0)
                    session_stats["by_source"][source] = session_stats["by_source"].get(source, 0) + 1
                    if source == "cache":
                        session_stats["cache_hits"] += 1
                except Exception:
                    pass

                # Show summary every 5 turns
                if session_stats["total"] % 5 == 0:
                    _print_session_summary(session_stats)
            else:
                console.print(f"[red]Error {response.status_code}: {response.text}[/red]")


def _print_session_summary(stats: dict) -> None:
    """Print running session summary."""
    total = stats["total"]
    if total == 0:
        return

    hit_rate = (stats["cache_hits"] / total) * 100
    avg_latency = stats["total_latency_ms"] / total

    table = Table(title="Session Summary", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Requests", str(total))
    table.add_row("Cache Hit Rate", f"{hit_rate:.1f}%")
    table.add_row("Avg Latency", f"{avg_latency:.0f}ms")
    table.add_row("Total Tokens", str(stats["total_tokens"]))

    for source, count in stats["by_source"].items():
        if count > 0:
            table.add_row(f"  {source}", str(count))

    console.print(table)


@app.command()
def stats(
    url: str = typer.Option(DEFAULT_URL, "--url", "-u", help="Gateway base URL"),
):
    """Show aggregate gateway statistics."""
    asyncio.run(_stats(url))


async def _stats(url: str) -> None:
    async with GatewayClient(url) as client:
        response = await client.stats()
        if not response.is_success:
            console.print(f"[red]Error {response.status_code}: {response.text}[/red]")
            raise typer.Exit(1)

        data = response.json()
        _render_stats_table(data)


def _render_stats_table(data: dict) -> None:
    """Render stats as a rich table."""
    table = Table(title="Gateway Statistics", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Cache stats
    cache = data.get("cache", {})
    table.add_row("Cache Hit Rate", f"{cache.get('hit_rate', 0):.1f}%")
    table.add_row("Cache Hits", str(cache.get("hits", 0)))
    table.add_row("Cache Misses", str(cache.get("misses", 0)))

    # Latency stats
    latency = data.get("latency", {})
    table.add_row("Avg Latency (ms)", f"{latency.get('avg_ms', 0):.0f}")
    table.add_row("P50 Latency (ms)", f"{latency.get('p50_ms', 0):.0f}")
    table.add_row("P95 Latency (ms)", f"{latency.get('p95_ms', 0):.0f}")
    table.add_row("P99 Latency (ms)", f"{latency.get('p99_ms', 0):.0f}")

    # Cost stats
    cost = data.get("cost", {})
    table.add_row("Total Cost (USD)", f"${cost.get('total_usd', 0):.4f}")
    table.add_row("Cost per 1k tokens", f"${cost.get('per_1k_tokens_usd', 0):.4f}")

    # Provider stats
    providers = data.get("providers", {})
    for provider, stats in providers.items():
        table.add_row(f"  {provider} requests", str(stats.get("requests", 0)))
        table.add_row(f"  {provider} errors", str(stats.get("errors", 0)))

    console.print(table)


@app.command()
def cache_inspect(
    query: str = typer.Argument(..., help="Query to search for in cache"),
    url: str = typer.Option(DEFAULT_URL, "--url", "-u", help="Gateway base URL"),
    top_n: int = typer.Option(5, "--top", "-n", help="Number of results to show"),
):
    """Inspect cache - show top-N nearest cached entries with similarity scores."""
    asyncio.run(_cache_inspect(query, url, top_n))


async def _cache_inspect(query: str, url: str, top_n: int) -> None:
    async with GatewayClient(url) as client:
        response = await client.cache_inspect(query, top_n)
        if not response.is_success:
            console.print(f"[red]Error {response.status_code}: {response.text}[/red]")
            raise typer.Exit(1)

        data = response.json()
        _render_cache_inspect_table(data, query)


def _render_cache_inspect_table(data: dict, query: str) -> None:
    """Render cache inspect results as a table."""
    results = data.get("results", [])

    if not results:
        console.print(f"[yellow]No cached entries found for query: {query}[/yellow]")
        return

    table = Table(title=f"Cache Inspect: '{query}'", show_header=True, header_style="bold")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Score", style="green", width=8)
    table.add_column("Cached Prompt", style="white", min_width=40)
    table.add_column("Age", style="dim", width=12)
    table.add_column("Source", style="magenta", width=8)

    for i, result in enumerate(results, 1):
        score = result.get("score", 0)
        prompt = result.get("prompt", "")[:80]
        age = result.get("age_seconds", 0)
        source = result.get("source", "unknown")

        # Format age
        if age < 60:
            age_str = f"{age:.0f}s"
        elif age < 3600:
            age_str = f"{age/60:.0f}m"
        else:
            age_str = f"{age/3600:.1f}h"

        table.add_row(str(i), f"{score:.3f}", prompt, age_str, source)

    console.print(table)


@app.command()
def health(
    url: str = typer.Option(DEFAULT_URL, "--url", "-u", help="Gateway base URL"),
):
    """Show gateway and per-provider health with circuit breaker state."""
    asyncio.run(_health(url))


async def _health(url: str) -> None:
    async with GatewayClient(url) as client:
        # Get overall health
        health_resp = await client.health()
        readyz_resp = await client.readyz()
        providers_resp = await client.providers_health()

        # Overall status
        overall_healthy = health_resp.is_success and readyz_resp.is_success
        status_text = "HEALTHY" if overall_healthy else "UNHEALTHY"
        status_style = "green" if overall_healthy else "red"

        console.print(Panel(f"Gateway: [{status_style}]{status_text}[/{status_style}]", border_style=status_style))

        # Providers table
        if providers_resp.is_success:
            data = providers_resp.json()
            providers = data.get("providers", {})

            table = Table(title="Provider Health", show_header=True, header_style="bold")
            table.add_column("Provider", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Circuit Breaker", style="yellow")
            table.add_column("Failures", style="red")
            table.add_column("Last Check", style="dim")

            for name, info in providers.items():
                healthy = info.get("healthy", False)
                status = "✓ Healthy" if healthy else "✗ Unhealthy"
                status_style = "green" if healthy else "red"

                cb_state = info.get("circuit_breaker", "unknown")
                cb_style = {"closed": "green", "half_open": "yellow", "open": "red"}.get(cb_state, "white")

                failures = info.get("consecutive_failures", 0)
                last_check = info.get("last_check", "never")

                table.add_row(
                    name,
                    Text(status, style=status_style),
                    Text(cb_state.upper(), style=cb_style),
                    str(failures),
                    last_check,
                )

            console.print(table)
        else:
            console.print(f"[red]Failed to get provider health: {providers_resp.text}[/red]")


# Benchmark commands - import and wrap existing scripts
@app.command()
def benchmark(
    url: str = typer.Option(DEFAULT_URL, "--url", "-u", help="Gateway base URL"),
    requests: int = typer.Option(100, "--requests", "-r", help="Number of requests"),
    concurrency: int = typer.Option(10, "--concurrency", "-c", help="Concurrent requests"),
    warmup: int = typer.Option(10, "--warmup", "-w", help="Warmup requests"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSON file"),
):
    """Run latency/throughput/cache benchmark."""
    asyncio.run(_run_benchmark(url, requests, concurrency, warmup, model, output))


async def _run_benchmark(
    url: str,
    requests: int,
    concurrency: int,
    warmup: int,
    model: Optional[str],
    output: Optional[str],
) -> None:
    # Import the benchmark module
    import sys
    sys.path.insert(0, "scripts")
    from benchmark import BenchmarkRunner

    console.print(Panel(f"Running Benchmark: {requests} requests, {concurrency} concurrent", border_style="blue"))

    runner = BenchmarkRunner(base_url=url, model=model)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running benchmarks...", total=None)
        results = await runner.run_benchmark(
            num_requests=requests,
            concurrency=concurrency,
            warmup=warmup,
        )
        progress.update(task, completed=True)

    _render_benchmark_results(results)

    if output:
        import json
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"[green]Results saved to {output}[/green]")


def _render_benchmark_results(results: dict) -> None:
    """Render benchmark results as rich tables."""
    # Summary table
    summary = results.get("summary", {})
    table = Table(title="Benchmark Summary", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Requests", str(summary.get("total_requests", 0)))
    table.add_row("Successful", str(summary.get("successful", 0)))
    table.add_row("Failed", str(summary.get("failed", 0)))
    table.add_row("Cache Hit Rate", f"{summary.get('cache_hit_rate', 0):.1f}%")
    table.add_row("Avg Latency (ms)", f"{summary.get('avg_latency_ms', 0):.0f}")
    table.add_row("P50 Latency (ms)", f"{summary.get('p50_latency_ms', 0):.0f}")
    table.add_row("P95 Latency (ms)", f"{summary.get('p95_latency_ms', 0):.0f}")
    table.add_row("P99 Latency (ms)", f"{summary.get('p99_latency_ms', 0):.0f}")
    table.add_row("Throughput (req/s)", f"{summary.get('throughput_rps', 0):.1f}")

    console.print(table)

    # Per-benchmark breakdown
    benchmarks = results.get("benchmarks", [])
    if benchmarks:
        table2 = Table(title="Per-Benchmark Results", show_header=True, header_style="bold")
        table2.add_column("Benchmark", style="cyan")
        table2.add_column("Requests", style="green")
        table2.add_column("Cache Hit Rate", style="yellow")
        table2.add_column("Avg Latency (ms)", style="magenta")
        table2.add_column("P95 Latency (ms)", style="magenta")
        table2.add_column("Throughput (req/s)", style="blue")

        for b in benchmarks:
            table2.add_row(
                b.get("name", "unknown"),
                str(b.get("requests", 0)),
                f"{b.get('cache_hit_rate', 0):.1f}%",
                f"{b.get('avg_latency_ms', 0):.0f}",
                f"{b.get('p95_latency_ms', 0):.0f}",
                f"{b.get('throughput_rps', 0):.1f}",
            )

        console.print(table2)


@app.command()
def load_test(
    url: str = typer.Option(DEFAULT_URL, "--url", "-u", help="Gateway base URL"),
    mode: str = typer.Option("constant", "--mode", help="Load test mode: constant, step, soak, stress"),
    rate: int = typer.Option(50, "--rate", help="Request rate (req/s) for constant/soak"),
    duration: int = typer.Option(60, "--duration", help="Duration in seconds"),
    start_rate: int = typer.Option(10, "--start-rate", help="Start rate for step mode"),
    end_rate: int = typer.Option(100, "--end-rate", help="End rate for step mode"),
    steps: int = typer.Option(5, "--steps", help="Number of steps for step mode"),
    step_duration: int = typer.Option(30, "--step-duration", help="Duration per step (seconds)"),
    max_rate: int = typer.Option(500, "--max-rate", help="Max rate for stress mode"),
    step: int = typer.Option(50, "--step", help="Rate step for stress mode"),
):
    """Run sustained load test."""
    asyncio.run(_run_load_test(
        url, mode, rate, duration, start_rate, end_rate, steps, step_duration, max_rate, step
    ))


async def _run_load_test(
    url: str,
    mode: str,
    rate: int,
    duration: int,
    start_rate: int,
    end_rate: int,
    steps: int,
    step_duration: int,
    max_rate: int,
    step: int,
) -> None:
    import sys
    sys.path.insert(0, "scripts")
    from load_test import LoadTestRunner

    console.print(Panel(f"Load Test: {mode} mode", border_style="blue"))

    runner = LoadTestRunner(base_url=url)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running load test...", total=None)
        results = await runner.run(
            mode=mode,
            rate=rate,
            duration=duration,
            start_rate=start_rate,
            end_rate=end_rate,
            steps=steps,
            step_duration=step_duration,
            max_rate=max_rate,
            step=step,
        )
        progress.update(task, completed=True)

    _render_load_test_results(results)


def _render_load_test_results(results: dict) -> None:
    """Render load test results."""
    table = Table(title="Load Test Results", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in results.items():
        if isinstance(value, float):
            table.add_row(key.replace("_", " ").title(), f"{value:.2f}")
        else:
            table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)


@app.command()
def chaos(
    url: str = typer.Option(DEFAULT_URL, "--url", "-u", help="Gateway base URL"),
    scenario: str = typer.Option("all", "--scenario", "-s", help="Chaos scenario to run"),
):
    """Run chaos/resilience tests."""
    asyncio.run(_run_chaos(url, scenario))


async def _run_chaos(url: str, scenario: str) -> None:
    import sys
    sys.path.insert(0, "scripts")
    from chaos_test import ChaosTestRunner

    console.print(Panel(f"Chaos Test: {scenario}", border_style="red"))

    runner = ChaosTestRunner(base_url=url)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running chaos scenario...", total=None)
        results = await runner.run(scenario=scenario)
        progress.update(task, completed=True)

    _render_chaos_results(results)


def _render_chaos_results(results: dict) -> None:
    """Render chaos test results."""
    table = Table(title="Chaos Test Results", show_header=True, header_style="bold")
    table.add_column("Scenario", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")

    for scenario, result in results.items():
        status = "PASS" if result.get("passed", False) else "FAIL"
        status_style = "green" if result.get("passed", False) else "red"
        details = result.get("details", "")
        table.add_row(scenario, Text(status, style=status_style), details)

    console.print(table)


if __name__ == "__main__":
    app()