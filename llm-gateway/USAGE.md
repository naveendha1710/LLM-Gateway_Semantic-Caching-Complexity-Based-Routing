# LLM Gateway — Usage Guide

This guide covers everything you need to run, configure, test, and benchmark the LLM Gateway.

---

## 📦 Quick Start

```bash
# 1. Install dependencies
cd z:\Projects\Semantic search and cache memory\llm-gateway
pip install -e ".[dev]"

# 2. Start Redis (required for caching)
docker run -d -p 6379:6379 redis:7-alpine

# 3. Configure your API keys (see Configuration below)
cp config/settings.dev.yaml config/settings.local.yaml
# Edit config/settings.local.yaml with your keys

# 4. Run the gateway
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 5. Test it works
curl http://localhost:8000/health
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

---

## 🖥️ CLI — `llmgw`

The `llmgw` CLI makes the gateway's routing, caching, latency, and cost behavior visible without raw curl/JSON.

```bash
# Install (already included in pip install -e ".[dev]")
pip install -e ".[dev]"

# Send a chat message (or start REPL if no message given)
llmgw chat "What is the capital of France?"

# Show aggregate gateway statistics (requests, cache hits, latency, cost)
llmgw stats

# Inspect cache — show top-N nearest cached entries with similarity scores
llmgw cache-inspect --top 5

# Show gateway and per-provider health with circuit breaker state
llmgw health

# Run latency/throughput/cache benchmark
llmgw benchmark --requests 100

# Run sustained load test (constant/step/soak/stress modes)
llmgw load-test --mode constant --rate 10 --duration 30

# Run chaos/resilience tests
llmgw chaos
```

### CLI Command Reference

| Command | Description | Key Options |
|---------|-------------|-------------|
| `chat` | Send chat message or start REPL | `--url`, `--model`, `--temperature`, `--max-tokens`, `--stream` |
| `stats` | Show aggregate gateway statistics | `--url` |
| `cache-inspect` | Show top-N cached entries with similarity | `--url`, `--top N` |
| `health` | Gateway + per-provider health + circuit breakers | `--url` |
| `benchmark` | Latency/throughput/cache benchmark | `--url`, `--requests`, `--concurrency`, `--warmup`, `--output` |
| `load-test` | Sustained load (constant/step/soak/stress) | `--mode`, `--rate`, `--duration`, `--url` |
| `chaos` | Resilience/chaos scenarios | `--scenario`, `--url` |

### Global Options

All commands accept:
- `--url` — Gateway base URL (default: `http://localhost:8000`)
- `--help` — Show command-specific help

```bash
llmgw chat --help
llmgw benchmark --help
# etc.
```

---

## ⚙️ Configuration

The gateway uses **YAML config + environment variables** (12-factor style). Environment variables override YAML.

### Config Files

| File | Purpose |
|------|---------|
| `config/settings.dev.yaml` | Development defaults (committed) |
| `config/settings.local.yaml` | Your local overrides (gitignored) |
| `config/settings.prod.yaml` | Production template |

### New Multi-Model Configuration Schema (Complexity-Based Tiers)

The gateway now supports **complexity-based tiers** (e.g., `simple`, `complex`) where each tier can contain **mixed providers** (both cloud and local). This replaces the old provider-based tier structure.

```yaml
# New multi-model configuration schema - COMPLEXITY-BASED TIERS
models:
  simple:                          # Tier name (complexity-based, NOT provider-based)
    routing_strategy: "cost_optimized"  # "cost_optimized" | "quality_optimized" | "latency_optimized" | "balanced"
    enabled: true                  # Enable/disable entire tier
    fallback_tier: "complex"       # Optional: tier to fall back to if ALL providers in this tier fail
    providers:                     # List of providers (can mix local AND cloud in same tier)
      - name: "nvidia-nano"        # Unique identifier for this provider entry
        provider: "nvidia"         # e.g., "nvidia", "openai", "anthropic", "ollama"
        provider_kind: "cloud"     # "local" | "cloud"
        base_url: "https://integrate.api.nvidia.com/v1"
        api_key_env: "NVIDIA_API_KEY"  # Environment variable name for API key (never store raw keys in YAML)
        model: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        enabled: true
        priority: 1                # Lower = higher priority (tried first within tier)
        max_tokens: 4096           # Optional: max tokens for this model
        cost_per_1k: 0.0001        # Optional: cost per 1k tokens (used by cost_optimized strategy)
        params:                    # Optional: generation parameters
          temperature: 0.3
          top_p: 0.9
      
      - name: "ollama-fast"
        provider: "ollama"
        provider_kind: "local"
        base_url: "http://localhost:11434"
        api_key_env: null          # No API key needed for local
        model: "llama3.2"
        enabled: false
        priority: 2
        max_tokens: 4096
        cost_per_1k: 0.0
        params:
          temperature: 0.3
          top_p: 0.9

  complex:
    routing_strategy: "quality_optimized"
    enabled: true
    fallback_tier: null            # No fallback (last resort)
    providers:
      - name: "nvidia-large"
        provider: "nvidia"
        provider_kind: "cloud"
        base_url: "https://integrate.api.nvidia.com/v1/chat/completions"
        api_key_env: "NVIDIA_API_KEY"
        model: "google/gemma-4-31b-it"
        enabled: true
        priority: 1
        max_tokens: 8192
        cost_per_1k: 0.0005
        params:
          temperature: 0.5
          top_p: 0.95
      
      - name: "ollama-large"
        provider: "ollama"
        provider_kind: "local"
        base_url: "http://localhost:11434"
        api_key_env: null
        model: "llama3.1"
        enabled: false
        priority: 2
        max_tokens: 8192
        cost_per_1k: 0.0
        params:
          temperature: 0.5
          top_p: 0.95
```

### Legacy Flat Keys (Deprecated but Supported)

The old flat key format is still supported for backward compatibility but will show a deprecation warning. They are auto-synthesized into the new `models` block.

```yaml
# Legacy flat keys (deprecated but still supported for backward compatibility)
cloud_provider_api_key: "nvapi-xxxxxxxxxxxxxxxxxxxxxxxx"
cloud_provider_base_url: "https://integrate.api.nvidia.com/v1"
cloud_provider_model: "nvidia/nemotron-3-nano-30b-a3b"
cloud_provider_models:
  - "nvidia/nemotron-3-nano-30b-a3b"
  - "nvidia/llama-3.1-nemotron-70b-instruct"
cloud_provider_timeout_seconds: 15
cloud_provider_max_retries: 3

local_provider_base_url: "http://localhost:11434"
local_provider_model: "llama3.2"
local_provider_models:
  - "llama3.2"
  - "llama3.1"
  - "mistral:7b"
local_provider_timeout_seconds: 60
```

### Environment Variable Overrides

```bash
export NVIDIA_API_KEY="nvapi-xxx"
export CACHE_ENABLED="true"
export CACHE_REDIS_URL="redis://localhost:6379/0"
export APP_LOG_LEVEL="DEBUG"
```

### Key Settings Reference

| Setting | Default | Description |
|---------|---------|-------------|
| `app_env` | `dev` | `dev`, `staging`, or `prod` |
| `app_port` | `8000` | HTTP port |
| `app_log_level` | `INFO` | Log level |
| `cloud_provider_timeout_seconds` | `15` | Cloud API timeout (legacy) |
| `local_provider_timeout_seconds` | `60` | Local model timeout (legacy) |
| `cache_enabled` | `false` | Enable semantic cache |
| `cache_similarity_threshold` | `0.85` | Cosine similarity for cache hits (0-1) |
| `cache_embedder_kind` | `hash` | `hash` or `sentence_transformer` |
| `cache_embedder_dimension` | `256` | Embedding dimension |

### Routing Strategies

| Strategy | Description |
|----------|-------------|
| `cost_optimized` | Select cheapest available model (lowest `cost_per_1k`) |
| `quality_optimized` | Select highest capability model (highest `max_tokens`) |
| `latency_optimized` | Select fastest model (lowest latency - future) |
| `balanced` | Balance cost/quality/latency (future) |
| `tier_priority` | Follow tier priority order (default) |
| `model_specific` | Use exact model requested in request |

### Tier Fallback with Cycle Detection

- Each tier can specify a `fallback_tier` to use when ALL providers in the tier fail
- Cycle detection prevents infinite fallback loops (e.g., simple → complex → simple)
- If a cycle is detected, the request fails with a clear error message

---

## 🏗️ Architecture Overview

```
Request → Middleware → Router (Classifier + Selector) → Provider (with failover)
                ↓
            Cache (fail-open) ← Quality Gate
```

### Request Flow

1. **Middleware**: Request ID + Error handling
2. **Router**: Classifies request → Selects tier/provider → Executes with failover
3. **Cache Check** (fail-open): Embed query → Search vector store → Return hit if score ≥ threshold
4. **Provider Call**: Cloud (NVIDIA) or Local (Ollama) with circuit breaker + retry
5. **Quality Gate**: Validate response quality before caching
6. **Cache Write** (fail-open): Store embedding + response if quality passes
7. **Response**: Returns `ChatCompletionResponse` with metadata headers

### Response Headers

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Unique request identifier |
| `X-Cache-Hit` | `true`/`false` — cache hit status |
| `X-Provider` | `cloud` or `local` — which provider responded |
| `X-Degraded` | `true` if failover occurred |
| `X-Routing-Strategy` | `cost_optimized`, `quality_optimized`, `local_only`, `cloud_only` |

---

## 🚀 Running the Gateway

### Development

```bash
# With auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or directly
python -m app.main
```

### Production

```bash
# Using gunicorn with uvicorn workers
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or Docker
docker build -t llm-gateway -f deploy/docker/Dockerfile .
docker run -p 8000:8000 --env-file .env llm-gateway
```

### With Docker Compose (includes Redis)

```bash
cd deploy/docker
docker-compose up -d
```

---

## 🧪 Testing

### Run All Tests

```bash
# All 90 tests
pytest -v

# Specific test suites
pytest tests/unit/test_cache.py -v
pytest tests/unit/test_providers.py -v
pytest tests/unit/test_router.py -v
pytest tests/integration/test_end_to_end.py -v
```

### Test with Coverage

```bash
pytest --cov=app --cov-report=term-missing
```

---

## 📊 Benchmarking & Load Testing

The `scripts/` directory contains three benchmarking tools.

### 1. `benchmark.py` — Latency, Throughput, Cache Hit Rate

```bash
# Quick test (20 requests)
python scripts/benchmark.py --quick

# Full benchmark (100 requests, 10 concurrent)
python scripts/benchmark.py --requests 100 --concurrency 10 --warmup 10

# Custom URL, model, output
python scripts/benchmark.py \
  --url http://localhost:8000 \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --output results.json

# Help
python scripts/benchmark.py --help
```

**Outputs:**
- Per-benchmark: avg/median/p95/p99 latency, throughput (req/s), cache hit rate
- Summary table across all benchmarks
- JSON export with `--output`

**Benchmarks run:**
1. **Basic Latency** — Single message, measures raw latency
2. **Cache Effectiveness** — Repeated identical requests to measure cache hits
3. **Concurrent Load** — Higher concurrency to test throughput
4. **Mixed Workload** — Rotating through different prompts

### 2. `load_test.py` — Sustained Load Patterns

```bash
# Constant load: 50 req/s for 60s
python scripts/load_test.py --mode constant --rate 50 --duration 60

# Step load: ramp 10→100 req/s over 5 steps
python scripts/load_test.py --mode step --start-rate 10 --end-rate 100 --steps 5 --step-duration 30

# Soak test: 20 req/s for 10 minutes
python scripts/load_test.py --mode soak --rate 20 --duration 600

# Stress test: find breaking point
python scripts/load_test.py --mode stress --max-rate 500 --step 50 --step-duration 10

# Help
python scripts/load_test.py --help
```

**Modes:**
| Mode | Use Case |
|------|----------|
| `constant` | Baseline performance at fixed rate |
| `step` | Find saturation point |
| `soak` | Long-running stability, memory leaks |
| `stress` | Breaking point, error rates under overload |

### 3. `chaos_test.py` — Resilience Testing

```bash
# Run all chaos scenarios
python scripts/chaos_test.py --all

# Specific scenarios
python scripts/chaos_test.py --scenario provider_failure
python scripts/chaos_test.py --scenario cache_failure
python scripts/chaos_test.py --scenario network_latency
python scripts/chaos_test.py --scenario timeout
python scripts/chaos_test.py --scenario rate_limit
python scripts/chaos_test.py --scenario cascading_failure

# Help
python scripts/chaos_test.py --help
```

**Scenarios:**
| Scenario | What It Tests |
|----------|---------------|
| `provider_failure` | Primary provider down → failover to backup |
| `cache_failure` | Redis down → fail-open (no cache, no errors) |
| `network_latency` | Injected latency → timeout handling |
| `timeout` | Request timeout → retry/circuit breaker |
| `rate_limit` | 429 responses → backoff behavior |
| `cascading_failure` | Multiple simultaneous failures |

---

## 🔍 Observability

### Health Endpoints

```bash
# Liveness (process alive)
curl http://localhost:8000/health

# Readiness (providers + cache healthy)
curl http://localhost:8000/readyz

# Detailed provider health
curl http://localhost:8000/health/providers
```

### Logging

Structured JSON logs (configured in `config/logging.yaml`):

```bash
# Pretty print logs
uvicorn app.main:app --log-config config/logging.yaml | jq .
```

### Metrics (Prometheus)

```bash
# Metrics endpoint
curl http://localhost:8000/metrics
```

Key metrics:
- `gateway_requests_total` — Counter by status, provider, cache_hit
- `gateway_request_duration_seconds` — Histogram
- `gateway_cache_hits_total` / `gateway_cache_misses_total`
- `gateway_provider_failures_total` — By provider, error type
- `gateway_circuit_breaker_state` — Gauge (0=closed, 1=half-open, 2=open)

---

## 🛠️ Development

### Project Structure

```
app/
├── api/
│   ├── middleware/       # Request ID, error handling
│   ├── routes/           # chat.py, health.py
│   └── schemas/          # Pydantic request/response models
├── cache/
│   ├── embedder.py       # HashEmbedder, SentenceTransformerEmbedder
│   ├── vector_store.py   # RedisVectorStore
│   ├── similarity.py     # Cosine similarity search
│   ├── quality_gate.py   # Response quality validation
│   └── invalidation.py   # Cache invalidation
├── core/
│   ├── config.py         # Pydantic Settings
│   ├── gateway.py        # Main orchestrator
│   └── exceptions.py     # Custom exceptions
├── observability/
│   └── logging_config.py # Structured logging
├── providers/
│   ├── base.py           # ModelProvider abstract base
│   ├── cloud_provider.py # NVIDIA API
│   ├── local_provider.py # Ollama
│   ├── circuit_breaker.py
│   └── retry.py          # Tenacity-based retry
├── router/
│   ├── classifier.py     # Request → RoutingDecision
│   ├── selector.py       # ProviderSelector with failover
│   └── tier_config.py    # Tier definitions
└── utils/
```

### Adding a New Provider

1. Implement `ModelProvider` in `app/providers/`
2. Add to `app/main.py` lifespan
3. Update `app/router/tier_config.py` with tier config
4. Add tests in `tests/unit/test_providers.py`

### Adding a New Embedder

1. Implement `Embedder` protocol in `app/cache/embedder.py`
2. Add config option in `app/core/config.py`
3. Update `get_embedder()` factory

---

## 🐛 Troubleshooting

### Gateway Won't Start

| Error | Fix |
|-------|-----|
| `cloud_provider_api_key is required when app_env='prod'` | Set `CLOUD_PROVIDER_API_KEY` or use `dev` env |
| `Connection refused: redis://localhost:6379` | Start Redis or set `cache_enabled: false` |
| `ModuleNotFoundError: app` | Run `pip install -e .` from project root |

### Cache Not Working

```bash
# Check cache is enabled
curl http://localhost:8000/health | jq .cache_enabled

# Check Redis connection
redis-cli ping

# Check similarity threshold (too high = no hits)
# Lower threshold in config: cache_similarity_threshold: 0.85
```

### High Latency / Timeouts

```bash
# Increase timeouts in config
cloud_provider_timeout_seconds: 30
local_provider_timeout_seconds: 120

# Check circuit breaker state
curl http://localhost:8000/health/providers
```

### Failover Not Working

```bash
# Check both providers configured
curl http://localhost:8000/health/providers

# Check tier config has fallback
# app/router/tier_config.py → TierConfig.fallback_provider
```

---

## 📝 API Reference

### Chat Completions

```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello!"}
  ],
  "model": "nvidia/nemotron-3-nano-30b-a3b",  # optional, uses router default
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "nvidia/nemotron-3-nano-30b-a3b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help?"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
  "source": "cloud",           # "cloud", "local", or "cache"
  "degraded": false,           # true if failover occurred
  "routing_strategy": "cost_optimized"
}
```

### Health

```bash
GET /health          # Liveness
GET /readyz          # Readiness
GET /health/providers # Per-provider health
```

---

## 🔗 Useful Commands

```bash
# View config (with env overrides)
python -c "from app.core.config import get_settings; import json; print(json.dumps(get_settings().model_dump(), indent=2, default=str))"

# Test embedder
python -c "from app.cache.embedder import get_embedder; e = get_embedder('hash', 256); print(e.embed('test'))"

# Test vector store
python -c "
import asyncio
from app.cache.vector_store import VectorStore
from app.cache.embedder import get_embedder
import fakeredis.aioredis
async def test():
    r = fakeredis.aioredis.FakeRedis()
    e = get_embedder('hash', 256)
    vs = VectorStore(r, e.dimension, 3600)
    await vs.initialize()
    await vs.put('k1', e.embed('hello'), {'data': 'test'})
    hit = await vs.search(e.embed('hello'), 0.5)
    print(hit)
asyncio.run(test())
"

# Run a single test with debug
pytest tests/unit/test_cache.py::test_cache_hit -v -s --log-cli-level=DEBUG
```

---

## 📚 Further Reading

- `PROGRESS.md` — Project progress tracking
- `docs/benchmark_results.md` — Benchmark results template
- `config/logging.yaml` — Logging configuration
- `deploy/k8s/` — Kubernetes manifests