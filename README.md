# LLM Gateway — Semantic Caching & Complexity-Based Routing

A Prototype LLM gateway that sits between your application and one or more
model providers. It provides:

- **Semantic caching** — repeated or near-duplicate requests are served from cache,
  cutting latency and cost. Uses vector embeddings (hash or sentence-transformer)
  with cosine similarity search.
- **Complexity-based routing** — simple queries go to a cheap/fast model, complex
  queries to a capable one. Tiers are config-driven and can mix cloud + local providers.
- **Quality-gated caching** — responses with `finish_reason != "stop"` or mid-sentence
  truncation are **rejected from cache** (prevents serving truncated answers).
- **Resilience** — retry with exponential backoff, circuit breakers, and automatic
  failover across providers within and across tiers.
- **Observability** — structured JSON logs with correlation IDs, response headers
  for cache hit/provider/degraded status, Prometheus-ready metrics.

---

## 📦 Quick Start

```bash
# 1. Install dependencies
cd z:\Projects\Semantic search and cache memory\llm-gateway
pip install -e ".[dev]"

# 2. Start Redis (required for caching)
docker run -d -p 6379:6379 redis:7-alpine

# 3. Configure your API keys
cp config/settings.dev.yaml config/settings.local.yaml
# Edit config/settings.local.yaml with your keys (or use env vars)

# 4. Run the gateway
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 5. Test it works
curl http://localhost:8000/healthz
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

---

## 🖥️ CLI — `llmgw` (or `python -m app.cli.main`)

The `llmgw` CLI makes the gateway's routing, caching, latency, and cost behavior
visible without raw curl/JSON. You can also run it directly as a Python module.

```bash
# Install (already included in pip install -e ".[dev]")
pip install -e ".[dev]"

# Send a chat message (or start REPL if no message given)
llmgw chat "What is the capital of France?"
# or equivalently:
python -m app.cli.main chat "What is the capital of France?"

# Show aggregate gateway statistics (requests, cache hits, latency, cost)
llmgw stats
python -m app.cli.main stats

# Inspect cache — show top-N nearest cached entries with similarity scores
llmgw cache-inspect --top 5
python -m app.cli.main cache-inspect --top 5

# Show gateway and per-provider health with circuit breaker state
llmgw health
python -m app.cli.main health

# Run latency/throughput/cache benchmark
llmgw benchmark --requests 100
python -m app.cli.main benchmark --requests 100

# Run sustained load test (constant/step/soak/stress modes)
llmgw load-test --mode constant --rate 10 --duration 30
python -m app.cli.main load-test --mode constant --rate 10 --duration 30

# Run chaos/resilience tests
llmgw chaos
python -m app.cli.main chaos
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

The gateway uses **YAML config + environment variables** (12-factor style).
Environment variables override YAML.

### Config Files

| File | Purpose |
|------|---------|
| `config/settings.dev.yaml` | Development defaults (committed) |
| `config/settings.local.yaml` | Your local overrides (gitignored) |
| `config/settings.prod.yaml` | Production template |

### New Multi-Model Configuration Schema (Complexity-Based Tiers)

The gateway supports **complexity-based tiers** (e.g., `simple`, `complex`) where
each tier can contain **mixed providers** (both cloud and local). This replaces
the old provider-based tier structure.

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

The old flat key format is still supported for backward compatibility but will
show a deprecation warning. They are auto-synthesized into the new `models` block.

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
| `latency_optimized` | Select fastest model (lowest latency — future) |
| `balanced` | Balance cost/quality/latency (future) |
| `tier_priority` | Follow tier priority order (default) |
| `model_specific` | Use exact model requested in request |

### Tier Fallback with Cycle Detection

- Each tier can specify a `fallback_tier` to use when ALL providers in the tier fail
- Cycle detection prevents infinite fallback loops (e.g., simple → complex → simple)
- If a cycle is detected, the request fails with a clear error message

---

## 🏗️ Architecture Overview

See [ARCHITECTURE.md](ARCHITECTURE.md) for:
- Request flow diagram (ASCII)
- Fail-open rationale
- Quality gate rationale (with incident history)
- Provider abstraction details
- Current folder structure

### Request Flow (High-Level)

```
Request → Middleware → Router (Classifier + Selector) → Provider (with failover)
                ↓
            Cache (fail-open) ← Quality Gate
```

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

```bash
# Run all tests (unit + integration)
pytest

# Specific test suites
pytest tests/unit/test_cache.py -v
pytest tests/unit/test_providers.py -v
pytest tests/unit/test_router.py -v
pytest tests/integration/test_end_to_end.py -v

# Test with coverage
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
```

---

## 📁 Project Structure

```
llm-gateway/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── api/
│   │   ├── middleware/         # Request ID, error handling
│   │   ├── routes/             # chat, health, stats, cache-inspect
│   │   └── schemas/            # Pydantic request/response models
│   ├── cache/
│   │   ├── embedder.py         # Hash / sentence-transformer embedders
│   │   ├── quality_gate.py     # Rejects truncated/incomplete responses from cache
│   │   ├── similarity.py       # Cosine similarity search
│   │   └── vector_store.py     # Redis-backed vector store (fakeredis in tests)
│   ├── cli/
│   │   └── main.py             # Typer CLI (llmgw)
│   ├── core/
│   │   ├── config.py           # Pydantic Settings + YAML overlay
│   │   ├── exceptions.py       # Custom exceptions
│   │   └── gateway.py          # Orchestrates: cache → router → provider → quality gate → cache
│   ├── observability/
│   │   └── logging_config.py   # Structured logging
│   ├── providers/
│   │   ├── base.py             # ModelProvider abstract interface
│   │   ├── cloud_provider.py   # NVIDIA/OpenAI/Anthropic via OpenAI-compatible API
│   │   ├── local_provider.py   # Ollama
│   │   ├── circuit_breaker.py  # Fail-fast per provider
│   │   └── retry.py            # Exponential backoff with jitter
│   ├── router/
│   │   ├── classifier.py       # Simple/complex query classification
│   │   ├── selector.py         # Tier + provider selection
│   │   └── tier_config.py      # Tier configuration models
│   └── utils/
├── config/
│   ├── logging.yaml
│   ├── settings.dev.yaml       # Committed dev defaults
│   └── settings.local.yaml     # Your overrides (gitignored)
├── deploy/
│   ├── docker/
│   └── k8s/
├── docs/
│   └── benchmark_results.md
├── scripts/
│   ├── benchmark.py
│   ├── chaos_test.py
│   ├── diagnose_provider.py    # Bypass gateway, hit provider directly
│   └── load_test.py
├── tests/
│   ├── unit/                   # 139 tests
│   └── integration/            # 6 tests
├── pyproject.toml
└── README.md
```

---

## �️ Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Web Framework** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance async API with automatic OpenAPI docs |
| **Async HTTP** | [httpx](https://www.python-httpx.org/) | Async HTTP client for provider calls with retries/timeout |
| **Validation** | [Pydantic](https://docs.pydantic.dev/) / [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) | Request/response validation, config management with YAML/env overlay |
| **Vector Cache** | [Redis](https://redis.io/) + [redis-py](https://redis-py.readthedocs.io/) | Semantic cache storage with vector similarity search |
| **Embeddings** | [sentence-transformers](https://www.sbert.net/) / custom hash | Text embeddings for semantic similarity (configurable) |
| **CLI** | [Typer](https://typer.tiangolo.com/) | Rich CLI (`llmgw`) with auto-completion |
| **Testing** | [pytest](https://docs.pytest.org/) / [pytest-asyncio](https://pytest-asyncio.readthedocs.io/) | 145 tests (139 unit + 6 integration) |
| **Observability** | Structured JSON logging, `X-Request-ID` correlation, Prometheus metrics | Production-grade observability |
| **Resilience** | Custom circuit breaker, exponential backoff with jitter, provider failover | Fault tolerance across cloud/local providers |
| **Containerization** | Docker, Docker Compose, Kubernetes manifests | Dev/prod deployment |
| **Type Checking** | [mypy](https://mypy-lang.org/) | Static type safety |
| **Linting/Formatting** | [ruff](https://docs.astral.sh/ruff/) | Fast Python linter/formatter |

---

## �📜 License

MIT License — see [LICENSE](LICENSE) for details.
