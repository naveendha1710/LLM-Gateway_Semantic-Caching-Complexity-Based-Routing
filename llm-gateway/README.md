# Semantic Router & LLM Caching Gateway

A production-grade LLM gateway that sits between your application and one or more
model providers. It provides:

- **Semantic caching** — repeated or near-duplicate requests are served from cache,
  cutting latency and cost.
- **Semantic routing** — simple queries go to a cheap/fast model, complex queries
  to a capable one.
- **Resilience** — retry with exponential backoff, circuit breakers, and automatic
  failover across providers.
- **Observability** — structured JSON logs with correlation IDs, Prometheus metrics,
  and OpenTelemetry-ready tracing.

## Quick start

```bash
# from the llm-gateway/ directory
pip install -e ".[dev]"

# copy and edit env
cp .env.example .env

# run the dev server
uvicorn app.main:app --reload --port 8000
```

## API

### `POST /v1/chat/completions`

OpenAI-compatible request body:

```json
{
  "model": "auto",
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ]
}
```

Response includes two extra fields:

| Field | Type | Description |
|-------|------|-------------|
| `source` | `string` | `"cache"`, `"local"`, or `"cloud"` |
| `degraded` | `bool` | `true` if the response came through a fallback/degraded path |

### `GET /healthz` / `GET /readyz`

Liveness and readiness probes.

## Project layout

See `llm-gateway-project-structure.md` at the repository root for the full
folder-by-folder breakdown and design rationale.

## Build phases

| Phase | Scope |
|-------|-------|
| 1 — Bare gateway | API, config, one cloud provider, cache stubbed |
| 2 — Semantic cache | Embedder, vector store, similarity, quality gate, invalidation |
| 3 — Router + resilience | Classifier, tier config, selector, retry, circuit breaker |
| 4 — Benchmarking | Latency/cost harness, load test, chaos test |

## License

MIT
