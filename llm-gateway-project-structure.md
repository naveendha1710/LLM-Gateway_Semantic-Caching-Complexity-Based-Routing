# Semantic Router & LLM Caching Gateway — Project Structure

Production-grade layout covering the happy path (cache hit/miss → route → generate) and the unhappy paths (provider failures, cache degradation, bad classifications, timeouts).

## Folder structure

```
llm-gateway/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Lint, type-check, unit + integration tests on every PR
│       └── deploy.yml                # Build image, push, deploy on merge to main
│
├── app/                              # Application source (importable package; each folder below is its own subpackage)
│   ├── __init__.py
│   ├── main.py                       # FastAPI app factory, startup/shutdown lifecycle, global request timeout
│   │
│   ├── api/                          # HTTP boundary — knows nothing about caching/routing internals
│   │   ├── routes/
│   │   │   ├── chat.py               # POST /v1/chat/completions — main happy-path entrypoint
│   │   │   ├── health.py             # GET /healthz (liveness), /readyz (checks cache + provider health)
│   │   │   └── admin.py              # Cache purge, provider enable/disable, config reload
│   │   ├── middleware/
│   │   │   ├── request_id.py         # Correlation ID on every request, threaded through logs/traces
│   │   │   ├── rate_limit.py         # Per-API-key limiting, returns 429 + Retry-After
│   │   │   ├── auth.py               # API key validation
│   │   │   └── error_handler.py      # Catches unhandled exceptions → clean JSON error + correct status code
│   │   └── schemas/
│   │       ├── requests.py           # Pydantic request models + validation
│   │       └── responses.py          # Includes `source` (cache/local/cloud) and `degraded` flag
│   │
│   ├── core/
│   │   ├── gateway.py                # Orchestrator: cache check → route → generate → quality gate → cache write
│   │   ├── config.py                 # Typed settings (pydantic-settings), fails fast on bad env vars
│   │   └── exceptions.py             # CacheUnavailable, ProviderTimeout, AllProvidersFailed, InvalidTierConfig
│   │
│   ├── cache/                        # Semantic caching subsystem
│   │   ├── embedder.py               # Local embedding model wrapper, own timeout, independent of generation path
│   │   ├── vector_store.py           # Qdrant/Redis client behind an interface, exposes health_check()
│   │   ├── similarity.py             # Threshold logic; logs near-miss scores for later tuning
│   │   ├── quality_gate.py           # is_cacheable() — blocks short/error/low-confidence responses from being cached
│   │   └── invalidation.py           # TTL sweep + manual purge-by-key
│   │
│   ├── router/                       # Complexity classification + provider selection
│   │   ├── classifier.py             # Simple vs complex tier decision
│   │   ├── tier_config.py            # Loads + validates config/tiers.yaml at startup, not at request time
│   │   └── selector.py               # Picks a provider within a tier; owns the fallback order
│   │
│   ├── providers/                    # Model abstraction — add a model without touching any other module
│   │   ├── base.py                   # ModelProvider protocol: generate(), cost_per_1k, health()
│   │   ├── local_provider.py         # Wraps any local model server (your quantized Qwen, vLLM, etc.)
│   │   ├── cloud_provider.py         # Wraps any cloud API (Gemini, OpenAI, Anthropic) — one class, config-driven
│   │   ├── retry.py                  # Exponential backoff + jitter around provider calls
│   │   └── circuit_breaker.py        # Opens after N failures, stops hammering a dead provider, half-opens to probe
│   │
│   ├── observability/                # First-class module, not an afterthought
│   │   ├── logging_config.py         # Structured JSON logs, correlation ID on every line
│   │   ├── metrics.py                # Prometheus: cache_hit_rate, provider_latency, error_rate, cost_per_request
│   │   └── tracing.py                # OpenTelemetry spans across cache → router → provider
│   │
│   └── utils/
│       ├── tokenizer.py              # Token counting for cost estimation and context-length checks
│       └── cost_calculator.py        # Per-request $ tracking, rolled up into benchmark reports
│
├── config/
│   ├── tiers.yaml                    # Tier → ordered provider list + per-model generation params
│   ├── settings.dev.yaml
│   ├── settings.prod.yaml
│   └── logging.yaml
│
├── scripts/                          # Operational — never imported by the running app
│   ├── seed_cache.py                 # Pre-warm cache with known Q&A pairs before launch
│   ├── benchmark.py                  # Latency/cost harness — this is what generates your resume numbers
│   ├── load_test.py                  # Simulated concurrent traffic (locust/k6 style)
│   └── chaos_test.py                 # Deliberately kills a provider/cache mid-run, confirms graceful degradation
│
├── tests/
│   ├── unit/
│   │   ├── test_cache.py
│   │   ├── test_router.py
│   │   ├── test_providers.py
│   │   └── test_quality_gate.py
│   ├── integration/
│   │   ├── test_end_to_end.py        # Cache miss → generate → cache hit on the same query repeated
│   │   └── test_failover.py          # Kills primary provider mid-test, asserts fallback + honest error on total failure
│   └── fixtures/
│       └── sample_prompts.json
│
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml        # app + Qdrant/Redis + local model server, one command to run locally
│   └── k8s/                          # Optional — only needed past a portfolio demo
│       ├── deployment.yaml
│       └── hpa.yaml                  # Autoscaling on queue depth
│
├── docs/
│   ├── architecture.md               # Diagrams + the "why" behind design decisions
│   ├── runbook.md                    # On-call reference: what each alert means, how to recover
│   └── benchmark_results.md          # Your actual measured numbers — doubles as resume evidence
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── LICENSE
```

*Each subpackage under `app/` includes its own `__init__.py`, omitted above for brevity.*

## Failure mode reference

| Scenario | Caught in | Behavior |
|---|---|---|
| Embedding model times out | `cache/embedder.py` | Fail open — treat as cache miss, log warning, request still succeeds |
| Vector store unreachable | `cache/vector_store.py` (`health_check()`) | Fail open for this request; alert if sustained across N requests |
| Local model not loaded / OOM | `providers/circuit_breaker.py` | Marked unhealthy, selector skips to next provider in the tier |
| Cloud API returns 429 / 5xx | `providers/retry.py` | Backoff retry (2–3 attempts), then fall through to next provider |
| All providers in a tier fail | `router/selector.py` → `AllProvidersFailed` | Return 503 + Retry-After, log as critical, response never gets cached |
| Classifier misjudges difficulty | not fatal | Logged for later review; not worth hard-failing on |
| Low-quality response generated | `cache/quality_gate.py` | Blocked before cache write so it can't pollute future hits |
| Cached answer goes stale | `cache/invalidation.py` | TTL expiry (default 24–72h), configurable per use case |
| Malformed `tiers.yaml` | `router/tier_config.py` | Fails at startup, not at first request |
| Request runs long end-to-end | timeout in `main.py` | Returns 504 before the client gives up waiting |
| One provider degraded, not fully down | `circuit_breaker.py` + `metrics.py` | System stays up in a degraded state; response carries `degraded: true` |

## Key rationale

- **`providers/` is the extension point.** Adding a model means writing one new file implementing `ModelProvider` — nothing else changes. That's the open/closed principle actually earning its keep, not just a term on a resume.
- **Fail open on cache, fail closed on generation.** If caching breaks, the safe default is to skip it and generate fresh — slower and pricier, but still correct. If every provider fails, the safe default is an honest error, never a fabricated answer.
- **Config lives outside code.** Swapping a model or reordering fallback in `tiers.yaml` shouldn't need a redeploy or a Python code review.
- **Observability is a module, not a side effect.** `observability/` is what actually produces your benchmark numbers — without correlation IDs and structured logs, "cut latency by X%" is a guess, not a measurement.
- **`test_failover.py` is the most interview-relevant file in the repo.** A test that kills a provider and asserts correct fallback (and honest degradation, not silent failure) is what separates "I called an LLM API" from "I built infrastructure."
- **`docs/runbook.md` exists because a demo that only works when everything's healthy isn't demonstrating LLMOps.** Writing the runbook forces you to think through recovery, not just the happy path.

## Suggested libraries

`tenacity` (retry/backoff) · a small custom breaker or `purgatory` (circuit breaking) · stdlib `logging` + `python-json-logger` (structured logs) · `prometheus-client` (metrics) · `opentelemetry-sdk` (tracing) · `pydantic-settings` (typed config)

## Mapping to the build phases

Phase 1 (bare gateway) only needs `api/`, `core/gateway.py` with the cache branch stubbed out, and one cloud provider. Phase 2 fills in `cache/` completely. Phase 3 fills in `router/` and `providers/`, including the fallback logic. Resilience pieces — `retry.py`, `circuit_breaker.py`, `observability/` — layer in once the happy path works end to end; they don't need to exist on day one, but the folder structure has a home for them from the start so you're not restructuring mid-build.
