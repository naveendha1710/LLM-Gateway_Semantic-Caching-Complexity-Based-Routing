# LLM Gateway — Architecture Documentation

This document describes the internal architecture, design decisions, and data flows of the LLM Gateway.

---

## 1. Request Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT REQUEST                                  │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MIDDLEWARE LAYER                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Request ID      │  │ Structured      │  │ Global Exception            │  │
│  │ Generation      │  │ Logging         │  │ Handler                     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            GATEWAY CORE (app/core/gateway.py)                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1. CACHE LOOKUP (FAIL-OPEN)                                         │    │
│  │    ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐   │    │
│  │    │ Embed Query  │───▶│ Vector Search│───▶│ Score ≥ Threshold? │   │    │
│  │    │ (hash or     │    │ (Redis/      │    │     │              │   │    │
│  │    │  sentence-   │    │  fakeredis)  │    │  YES  NO           │   │    │
│  │    │  transformer)│    │              │    │   │    │           │   │    │
│  │    └──────────────┘    └──────────────┘    │   ▼    ▼           │   │    │
│  │                                            │  HIT  MISS          │   │    │
│  │                                            └─────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                    ┌───────────────┴───────────────┐                         │
│                    ▼                               ▼                         │
│         ┌─────────────────────┐           ┌─────────────────────┐            │
│         │ RETURN CACHED       │           │ 2. ROUTER           │            │
│         │ RESPONSE            │           │    (app/router/)    │            │
│         │ X-Cache-Hit: true   │           │                     │            │
│         └─────────────────────┘           │  ┌───────────────┐  │            │
│                                           │  │ CLASSIFIER    │  │            │
│                                           │  │ (simple vs    │  │            │
│                                           │  │  complex)     │  │            │
│                                           │  └───────┬───────┘  │            │
│                                           │          │          │            │
│                                           │          ▼          │            │
│                                           │  ┌───────────────┐  │            │
│                                           │  │ SELECTOR      │  │            │
│                                           │  │ (tier +       │  │            │
│                                           │  │  provider)    │  │            │
│                                           │  └───────┬───────┘  │            │
│                                           └──────────┬──────────┘            │
│                                                      │                        │
└──────────────────────────────────────────────────────┼────────────────────────┘
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. PROVIDER EXECUTION (app/providers/)                                       │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ FOR EACH PROVIDER IN TIER (by priority):                             │   │
│  │   ┌─────────────┐    ┌─────────────┐    ┌────────────────────────┐   │   │
│  │   │ Circuit     │───▶│ Retry with  │───▶│ Execute Request        │   │   │
│  │   │ Breaker     │    │ Backoff     │    │ (Cloud: NVIDIA/OpenAI  │   │   │
│  │   │ Check       │    │ (exp. backoff│    │  Local: Ollama)       │   │   │
│  │   └─────────────┘    │  + jitter)  │    └────────────────────────┘   │   │
│  │                      └─────────────┘                                  │   │
│  │         │                    │                                        │   │
│  │    OPEN                  EXHAUSTED                                   │   │
│  │    │                      │                                          │   │
│  │    ▼                      ▼                                          │   │
│  │ NEXT PROVIDER         NEXT PROVIDER                                  │   │
│  │                                                                       │   │
│  │ IF ALL PROVIDERS IN TIER FAIL:                                       │   │
│  │   → Check fallback_tier                                              │   │
│  │   → Cycle detection (prevent infinite loops)                         │   │
│  │   → Execute fallback tier                                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. QUALITY GATE (app/cache/quality_gate.py)                                 │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ passes_quality_gate(response)                                        │   │
│  │                                                                      │   │
│  │ CHECK 1: finish_reason == "stop"                                     │   │
│  │   • Rejects "length" (token limit hit)                               │   │
│  │   • Rejects "content_filter", "tool_calls", etc.                     │   │
│  │                                                                      │   │
│  │ CHECK 2: Terminal punctuation present                                │   │
│  │   • Regex: r'[.!?]["\']?\s*$'                                        │   │
│  │   • Catches mid-sentence truncation even if finish_reason="stop"    │   │
│  │                                                                      │   │
│  │ RESULT: True → cacheable, False → NOT cacheable                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
         ┌─────────────────────┐       ┌─────────────────────┐
         │ QUALITY PASSES      │       │ QUALITY FAILS       │
         │                     │       │                     │
         │ 5. CACHE WRITE      │       │ RETRY LOGIC         │
         │ (fail-open)         │       │ (Option A)          │
         │                     │       │                     │
         │ • Embed response    │       │ 1. Double max_tokens│
         │ • Store in vector   │       │    (capped at 8192) │
         │   store             │       │ 2. Retry ONCE       │
         │ • Fire-and-forget   │       │                     │
         │   (non-blocking)    │       │ IF RETRY PASSES:    │
         │                     │       │   → Cache & return  │
         │ X-Cache-Hit: false  │       │ IF RETRY FAILS:     │
         │ X-Provider: cloud   │       │   → degraded=true   │
         │ X-Degraded: false   │       │   → Return response │
         └─────────────────────┘       │     with degraded   │
                                       │     flag            │
                                       └─────────────────────┘
                    │                           │
                    └───────────────┬───────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. RESPONSE                                                                  │
│                                                                              │
│  ChatCompletionResponse {                                                    │
│    choices: [...],                                                           │
│    source: "cache" | "cloud" | "local",                                     │
│    degraded: boolean,                                                        │
│    model: string,                                                            │
│    usage: {...}                                                              │
│  }                                                                           │
│                                                                              │
│  Headers:                                                                    │
│  X-Request-ID: <uuid>                                                        │
│  X-Cache-Hit: true | false                                                   │
│  X-Provider: cloud | local                                                   │
│  X-Degraded: true | false                                                    │
│  X-Routing-Strategy: cost_optimized | quality_optimized | ...               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Fail-Open Rationale

### What "Fail-Open" Means Here

The cache layer is designed to **never block the request path**. If Redis is down,
the embedder fails, or the vector store throws — the request proceeds directly to
the provider. The user gets an answer (possibly slower, possibly more expensive),
but never an error caused by the cache.

### Why This Matters

| Scenario | Fail-Closed Behavior | Fail-Open Behavior |
|----------|---------------------|-------------------|
| Redis OOM / crash | All requests 500 | Requests route to provider |
| Embedder model load fails | All requests 500 | Requests route to provider |
| Vector store corruption | All requests 500 | Requests route to provider |
| Network partition to Redis | All requests 500 | Requests route to provider |

### Implementation

```python
# app/core/gateway.py - simplified
async def handle(request):
    # Cache lookup — wrapped in try/except, never raises
    cached = await self._cache_lookup(request)
    if cached:
        return cached
    
    # Provider call — this CAN raise, but that's a real error
    response = await self._call_provider(request)
    
    # Quality gate — pure function, no I/O
    if passes_quality_gate(response):
        # Cache write — fire-and-forget, errors logged not raised
        asyncio.create_task(self._cache_write(request, response))
    
    return response
```

### Trade-offs Accepted

- **Stale reads possible**: If cache write fails silently, next request misses cache
- **Duplicate work**: Two simultaneous identical requests may both hit provider
- **Metrics drift**: Cache hit rate may be slightly under-reported

These are acceptable because **availability > cache consistency** for an LLM gateway.

---

## 3. Quality Gate Rationale

### The Incident That Motivated This

**Date**: During development testing  
**Trigger**: Query "what is the capital of india" with low `max_tokens`  
**Provider Response**: `finish_reason="length"`, content `"The capital of India is **New`  
**Result**: Truncated response was cached and served to subsequent users

### Root Cause

The original quality gate only checked:
```python
# OLD (buggy)
_GOOD_FINISH_REASONS = {"stop", "length"}  # ← "length" was accepted!
```

A response with `finish_reason="length"` means **the model hit the token limit mid-generation**. The content is by definition incomplete.

### The Fix

```python
# NEW (strict)
_GOOD_FINISH_REASONS = {"stop"}  # Only "stop" is acceptable

_TERMINAL_PUNCTUATION = re.compile(r'[.!?]["\']?\s*$')

def passes_quality_gate(response: ChatCompletionResponse) -> bool:
    # Check 1: finish_reason must be "stop"
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        return False
    
    # Check 2: Content must end with terminal punctuation
    content = choice.message.content or ""
    if not _TERMINAL_PUNCTUATION.search(content.strip()):
        return False
    
    return True
```

### Why Both Checks?

| Check | Catches | Example |
|-------|---------|---------|
| `finish_reason == "stop"` | Token limit truncation | `"The capital is New` (finish_reason="length") |
| Terminal punctuation | Mid-sentence stop without token limit | `"The answer is 42` (finish_reason="stop" but no period) |

The second check handles edge cases where the model stops early (e.g., EOS token emitted prematurely) but still reports `finish_reason="stop"`.

### Recovery Behavior (Option A: Retry with Higher Token Budget)

When quality gate rejects a response:

1. **Double `max_tokens`** (capped at 8192)
2. **Retry ONCE** with the same provider
3. **If retry passes**: Cache and return clean response
4. **If retry fails**: Return response with `degraded=true` flag, **do not cache**

This ensures:
- Users get complete answers when possible (automatic recovery)
- Callers can detect degraded responses via `degraded` field / `X-Degraded` header
- Incomplete responses never pollute the cache

---

## 4. Provider Abstraction

### Interface: `ModelProvider` (app/providers/base.py)

```python
class ModelProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @property
    @abstractmethod
    def kind(self) -> ProviderKind: ...  # LOCAL | CLOUD
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs
    ) -> ChatCompletionResponse: ...
    
    @abstractmethod
    async def health_check(self) -> ProviderHealth: ...
```

### Adding a New Provider = One Config Entry

No code changes needed. Just add to `config/settings.local.yaml`:

```yaml
models:
  simple:
    providers:
      - name: "my-new-provider"
        provider: "openai"           # or "anthropic", "ollama", "custom"
        provider_kind: "cloud"
        base_url: "https://api.openai.com/v1"
        api_key_env: "OPENAI_API_KEY"
        model: "gpt-4o-mini"
        enabled: true
        priority: 1
        max_tokens: 4096
        cost_per_1k: 0.00015
```

The gateway:
1. Reads config at startup
2. Instantiates provider via factory (`app/providers/__init__.py`)
3. Wraps with circuit breaker + retry automatically
4. Includes in tier routing

### Built-in Providers

| Provider | Kind | Base Class | Notes |
|----------|------|------------|-------|
| NVIDIA | Cloud | `CloudProvider` | OpenAI-compatible API |
| OpenAI | Cloud | `CloudProvider` | OpenAI-compatible API |
| Anthropic | Cloud | `CloudProvider` | Via OpenAI-compatible proxy |
| Ollama | Local | `LocalProvider` | Native Ollama API |

### Resilience Per Provider

Each provider instance gets:
- **Circuit Breaker**: 5 failures → open for 30s → half-open → close
- **Retry**: Exponential backoff (1s, 2s, 4s) + jitter, max 3 attempts
- **Timeout**: Configurable per provider (default: cloud=15s, local=60s)

---

## 5. Current Folder Structure (Actual)

```
llm-gateway/
├── app/
│   ├── main.py                      # FastAPI app factory, lifespan, routes
│   ├── api/
│   │   ├── middleware/
│   │   │   ├── error_handler.py     # Global exception handler → JSON error
│   │   │   └── request_id.py        # X-Request-ID generation + propagation
│   │   ├── routes/
│   │   │   ├── cache_inspect.py     # POST /v1/cache/inspect
│   │   │   ├── chat.py              # POST /v1/chat/completions
│   │   │   ├── health.py            # GET /healthz, /readyz
│   │   │   └── stats.py             # GET /v1/stats
│   │   └── schemas/
│   │       ├── requests.py          # ChatCompletionRequest, etc.
│   │       └── responses.py         # ChatCompletionResponse, Choice, etc.
│   ├── cache/
│   │   ├── embedder.py              # HashEmbedder, SentenceTransformerEmbedder
│   │   ├── invalidation.py          # TTL-based + manual invalidation
│   │   ├── quality_gate.py          # passes_quality_gate() — THE GATE
│   │   ├── similarity.py            # Cosine similarity search
│   │   └── vector_store.py          # RedisVectorStore (fakeredis in tests)
│   ├── cli/
│   │   └── main.py                  # Typer CLI: chat, stats, cache-inspect, health, benchmark, load-test, chaos
│   ├── core/
│   │   ├── config.py                # Settings (Pydantic Settings + YAML)
│   │   ├── exceptions.py            # GatewayError, ProviderError, CacheError, etc.
│   │   └── gateway.py               # Core orchestration: handle() method
│   ├── observability/
│   │   └── logging_config.py        # Structured JSON logging, correlation IDs
│   ├── providers/
│   │   ├── __init__.py              # Provider factory
│   │   ├── base.py                  # ModelProvider ABC
│   │   ├── cloud_provider.py        # OpenAI-compatible cloud providers
│   │   ├── local_provider.py        # Ollama provider
│   │   ├── circuit_breaker.py       # CircuitBreaker class
│   │   └── retry.py                 # retry_with_backoff decorator
│   ├── router/
│   │   ├── __init__.py
│   │   ├── classifier.py            # Simple/complex query classification
│   │   ├── selector.py              # Tier + provider selection logic
│   │   └── tier_config.py           # TierConfig, ProviderConfig models
│   └── utils/
├── config/
│   ├── logging.yaml                 # Logging configuration
│   ├── settings.dev.yaml            # Committed dev defaults
│   └── settings.local.yaml          # Local overrides (gitignored)
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml       # Gateway + Redis
│   └── k8s/
│       ├── deployment.yaml
│       ├── service.yaml
│       └── configmap.yaml
├── docs/
│   └── benchmark_results.md
├── scripts/
│   ├── benchmark.py                 # Latency/throughput/cache benchmark
│   ├── chaos_test.py                # Resilience scenarios
│   ├── diagnose_provider.py         # Bypass gateway, hit provider directly
│   └── load_test.py                 # Sustained load patterns
├── tests/
│   ├── unit/
│   │   ├── test_cache.py            # 56 tests (quality gate, embedder, vector store)
│   │   ├── test_providers.py        # Provider mocking, circuit breaker, retry
│   │   ├── test_router.py           # Classifier, selector, tier fallback
│   │   └── test_tier_config.py      # Config parsing, legacy synthesis
│   └── integration/
│       └── test_end_to_end.py       # 6 tests (full request flow)
├── pyproject.toml                   # Dependencies, pytest config, entry points
└── README.md
```

---

## 6. Key Data Models

### Request (app/api/schemas/requests.py)

```python
class ChatCompletionRequest(BaseModel):
    model: str = "auto"              # "auto" → router selects
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    stream: bool = False
```

### Response (app/api/schemas/responses.py)

```python
class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    source: Literal["cache", "cloud", "local"] = "cloud"
    degraded: bool = False
```

### Tier Config (app/router/tier_config.py)

```python
class ProviderConfig(BaseModel):
    name: str
    provider: str                    # "nvidia", "openai", "ollama", ...
    provider_kind: ProviderKind      # LOCAL | CLOUD
    base_url: str
    api_key_env: Optional[str]
    model: str
    enabled: bool = True
    priority: int = 1
    max_tokens: Optional[int] = None
    cost_per_1k: float = 0.0
    params: Dict[str, Any] = {}

class TierConfig(BaseModel):
    name: str
    routing_strategy: RoutingStrategy
    enabled: bool = True
    fallback_tier: Optional[str] = None
    providers: List[ProviderConfig] = []
```

---

## 7. Configuration System

### Layered Config (app/core/config.py)

```
Settings (Pydantic BaseSettings)
├── YAML file (settings.dev.yaml / settings.local.yaml)
├── Environment variables (prefix: APP_)
└── Defaults (in code)
```

### Legacy → New Schema Synthesis

The `Settings` class accepts **both** schemas:

```yaml
# NEW (preferred)
models:
  simple:
    providers: [...]

# LEGACY (still works, emits DeprecationWarning)
cloud_provider_api_key: "xxx"
cloud_provider_model: "xxx"
local_provider_model: "xxx"
```

The `_synthesize_models_from_legacy()` method converts flat keys to the new
tier-based structure at load time. This ensures zero-downtime migration.

---

## 8. Testing Strategy

| Layer | Tool | Coverage |
|-------|------|----------|
| Unit | pytest + pytest-asyncio | 139 tests |
| Integration | pytest + httpx.AsyncClient | 6 tests |
| Contract | Pydantic models | Request/response validation |
| Chaos | scripts/chaos_test.py | Provider failure, cache failure, latency, timeout, rate limit, cascade |

### Running Tests

```bash
# All tests
pytest

# Unit only
pytest tests/unit/ -v

# Integration only
pytest tests/integration/ -v

# With coverage
pytest --cov=app --cov-report=term-missing
```

---

## 9. Deployment

### Docker

```bash
cd deploy/docker
docker-compose up -d  # Starts gateway + Redis
```

### Kubernetes

```bash
kubectl apply -f deploy/k8s/
```

### Environment Variables (Production)

```bash
APP_ENV=prod
APP_LOG_LEVEL=INFO
CACHE_ENABLED=true
CACHE_REDIS_URL=redis://redis:6379/0
NVIDIA_API_KEY=${SECRET_NVIDIA_API_KEY}
```

---

## 10. Future Work

- [ ] **Latency-optimized routing**: Track per-provider latency, route to fastest
- [ ] **Streaming support**: Cache streaming responses (chunk-level quality gate)
- [ ] **Multi-tenancy**: Per-tenant cache namespaces, rate limits, model access
- [ ] **Prompt templating**: Server-side prompt templates with variable injection
- [ ] **Cost tracking**: Persistent cost accumulation per tenant/project
- [ ] **OpenTelemetry**: Full distributed tracing integration