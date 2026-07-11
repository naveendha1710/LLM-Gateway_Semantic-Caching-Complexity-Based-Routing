"""Cache invalidation strategies.

Two complementary mechanisms:

* **TTL-based** — every cache entry has a TTL set at write time (see
  :class:`VectorStore`). Redis evicts expired keys automatically; no
  application code is needed.

* **Explicit** — :func:`invalidate` removes a single entry by cache key,
  and :func:`invalidate_all` clears the entire cache. These are exposed so
  an admin endpoint (or a model-deploy webhook) can force a flush.

All operations are fail-open: a Redis error is logged but never propagated
to the caller, because invalidation failure should not break a request.
"""

from __future__ import annotations

import logging

from app.cache.vector_store import VectorStore

logger = logging.getLogger(__name__)


async def invalidate(store: VectorStore, key: str) -> bool:
    """Remove a single cache entry by key. Fail-open."""
    try:
        removed = await store.remove(key)
        if removed:
            logger.info("cache_invalidated", extra={"key": key})
        return removed
    except Exception as exc:
        logger.warning("cache_invalidate_failed", extra={"key": key, "error": str(exc)})
        return False


async def invalidate_all(store: VectorStore) -> None:
    """Clear the entire cache. Fail-open."""
    try:
        await store.clear()
        logger.info("cache_invalidated_all")
    except Exception as exc:
        logger.warning("cache_invalidate_all_failed", extra={"error": str(exc)})
