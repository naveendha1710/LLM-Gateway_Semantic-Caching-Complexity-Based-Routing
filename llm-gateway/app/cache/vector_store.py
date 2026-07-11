"""Vector store — persistent embedding + payload storage backed by Redis.

The store keeps two Redis structures per cache entry:

* ``cache:vec:<key>``  — the embedding bytes (numpy float32, little-endian).
* ``cache:payload:<key>`` — the serialised :class:`ChatCompletionResponse`.

Both share a TTL so stale entries expire automatically. The
:class:`SimilarityIndex` is rebuilt from Redis on startup and kept in sync
on every write/remove.

Design notes
------------
* **Fail-open**: every Redis error is caught and logged. The cache is a
  performance optimisation, never a correctness dependency. If Redis is
  down the gateway still serves requests — just without caching.
* **fakeredis-friendly**: tests inject a ``fakeredis.aioredis.FakeRedis``
  client; production injects a real ``redis.asyncio.Redis``.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import numpy as np

from app.cache.similarity import SimilarityHit, SimilarityIndex

logger = logging.getLogger(__name__)

# Redis key prefixes — kept short to save memory.
_VEC_PREFIX = "cache:vec:"
_PAYLOAD_PREFIX = "cache:payload:"


@runtime_checkable
class AsyncRedisLike(Protocol):
    """Minimal subset of redis.asyncio.Redis we depend on."""

    async def get(self, name: str | bytes) -> Any: ...
    async def set(self, name: str, value: Any, ex: int | None = ...) -> Any: ...
    async def delete(self, *names: str) -> int: ...
    async def keys(self, pattern: str) -> list[Any]: ...
    async def close(self) -> None: ...
    async def ping(self) -> bool: ...


def _vector_to_bytes(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _bytes_to_vector(data: bytes, dimension: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32).reshape(-1).copy()


class VectorStore:
    """Redis-backed vector + payload store with an in-memory similarity index."""

    def __init__(
        self,
        redis: AsyncRedisLike,
        embedder_dimension: int,
        ttl_seconds: int = 86400,
    ) -> None:
        self._redis = redis
        self._ttl = ttl_seconds
        self._index = SimilarityIndex(dimension=embedder_dimension)

    # ── Lifecycle ────────────────────────────────────────────────
    async def initialize(self) -> None:
        """Rebuild the in-memory index from Redis (call once on startup)."""
        try:
            vec_keys = await self._redis.keys(f"{_VEC_PREFIX}*")
            for key in vec_keys:
                k = key.decode() if isinstance(key, bytes) else key
                cache_key = k[len(_VEC_PREFIX):]
                raw = await self._redis.get(k)
                if raw is None:
                    continue
                vec = _bytes_to_vector(raw, self._index._dimension)
                self._index.add(cache_key, vec)
            logger.info("vector_store_initialized", extra={"entries": self._index.size})
        except Exception as exc:
            logger.warning("vector_store_init_failed", extra={"error": str(exc)})

    async def close(self) -> None:
        try:
            await self._redis.close()
        except Exception:  # pragma: no cover - best effort
            pass

    async def health(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception:
            return False

    # ── Read path ────────────────────────────────────────────────
    async def search(self, query_vec: np.ndarray, threshold: float) -> SimilarityHit | None:
        """Return the best similarity hit above ``threshold`` (or ``None``)."""
        try:
            return self._index.search(query_vec, threshold)
        except Exception as exc:
            logger.warning("vector_store_search_failed", extra={"error": str(exc)})
            return None

    async def search_top_n(
        self, query_vec: np.ndarray, threshold: float, top_n: int
    ) -> list[SimilarityHit]:
        """Return up to ``top_n`` similarity hits above ``threshold``."""
        try:
            return self._index.search_top_n(query_vec, threshold, top_n)
        except Exception as exc:
            logger.warning("vector_store_search_top_n_failed", extra={"error": str(exc)})
            return []

    async def get_payload(self, key: str) -> Any | None:
        """Retrieve the cached payload for ``key`` (or ``None``)."""
        try:
            import json

            raw = await self._redis.get(f"{_PAYLOAD_PREFIX}{key}")
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("vector_store_get_failed", extra={"error": str(exc)})
            return None

    # ── Write path ───────────────────────────────────────────────
    async def put(self, key: str, vector: np.ndarray, payload: dict[str, Any]) -> None:
        """Store an embedding + payload, updating the in-memory index."""
        try:
            import json

            vec_bytes = _vector_to_bytes(vector)
            await self._redis.set(f"{_VEC_PREFIX}{key}", vec_bytes, ex=self._ttl)
            await self._redis.set(
                f"{_PAYLOAD_PREFIX}{key}", json.dumps(payload), ex=self._ttl
            )
            self._index.add(key, vector)
        except Exception as exc:
            logger.warning("vector_store_put_failed", extra={"error": str(exc)})

    # ── Delete path ──────────────────────────────────────────────
    async def remove(self, key: str) -> bool:
        """Remove a single entry; return ``True`` if it existed."""
        try:
            deleted = await self._redis.delete(
                f"{_VEC_PREFIX}{key}", f"{_PAYLOAD_PREFIX}{key}"
            )
            self._index.remove(key)
            return deleted > 0
        except Exception as exc:
            logger.warning("vector_store_remove_failed", extra={"error": str(exc)})
            return False

    async def clear(self) -> None:
        """Remove all cache entries (used by invalidation + tests)."""
        try:
            vec_keys = await self._redis.keys(f"{_VEC_PREFIX}*")
            pay_keys = await self._redis.keys(f"{_PAYLOAD_PREFIX}*")
            all_keys = list(vec_keys) + list(pay_keys)
            if all_keys:
                await self._redis.delete(*all_keys)
            self._index.clear()
        except Exception as exc:
            logger.warning("vector_store_clear_failed", extra={"error": str(exc)})

    @property
    def size(self) -> int:
        return self._index.size
