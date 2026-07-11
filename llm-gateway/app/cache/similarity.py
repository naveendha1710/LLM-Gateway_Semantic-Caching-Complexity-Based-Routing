"""Similarity search over cached embeddings.

Pure functions on numpy arrays — no I/O — so they are trivial to unit test.
The :class:`SimilarityIndex` keeps a small in-memory matrix of cached
embeddings and returns the best match above a configurable threshold. The
:class:`VectorStore` (see ``vector_store.py``) owns persistence; this module
owns the math.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimilarityHit:
    """A single similarity-search result."""

    key: str
    score: float


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.

    Vectors are assumed already L2-normalised by the embedder, but we guard
    against zero vectors to avoid division-by-zero.
    """
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def best_match(
    query: np.ndarray, vectors: np.ndarray, keys: list[str]
) -> SimilarityHit | None:
    """Return the highest-scoring (key, score) pair, or ``None`` if empty."""
    if vectors.shape[0] == 0 or not keys:
        return None
    # Batched cosine similarity. Vectors are normalised, so dot product ≈ cosine.
    q = query / max(float(np.linalg.norm(query)), 1e-12)
    norms = np.linalg.norm(vectors, axis=1)
    safe = np.where(norms == 0.0, 1e-12, norms)
    normalised = vectors / safe[:, None]
    scores = normalised @ q
    idx = int(np.argmax(scores))
    return SimilarityHit(key=keys[idx], score=float(scores[idx]))


class SimilarityIndex:
    """In-memory similarity index over a fixed embedding dimension.

    Not persistent — the :class:`VectorStore` reloads embeddings from Redis
    on startup and delegates lookups here. Kept separate so the math is
    testable without any I/O.
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._keys: list[str] = []
        self._vectors: np.ndarray = np.zeros((0, dimension), dtype=np.float32)

    @property
    def size(self) -> int:
        return len(self._keys)

    def add(self, key: str, vector: np.ndarray) -> None:
        """Insert or replace an embedding keyed by ``key``."""
        if vector.shape != (self._dimension,):
            raise ValueError(
                f"vector dimension mismatch: expected {self._dimension}, "
                f"got {vector.shape}"
            )
        if key in self._keys:
            idx = self._keys.index(key)
            self._vectors[idx] = vector
            return
        self._keys.append(key)
        self._vectors = np.vstack([self._vectors, vector[None, :]]).astype(np.float32)

    def remove(self, key: str) -> bool:
        """Remove an entry; return ``True`` if it was present."""
        if key not in self._keys:
            return False
        idx = self._keys.index(key)
        self._keys.pop(idx)
        self._vectors = np.delete(self._vectors, idx, axis=0)
        return True

    def search(self, query: np.ndarray, threshold: float) -> SimilarityHit | None:
        """Return the best match above ``threshold``, else ``None``."""
        hit = best_match(query, self._vectors, self._keys)
        if hit is None or hit.score < threshold:
            return None
        return hit

    def search_top_n(
        self, query: np.ndarray, threshold: float, top_n: int
    ) -> list[SimilarityHit]:
        """Return up to ``top_n`` matches above ``threshold``, sorted by score descending."""
        if self._vectors.shape[0] == 0 or not self._keys:
            return []
        q = query / max(float(np.linalg.norm(query)), 1e-12)
        norms = np.linalg.norm(self._vectors, axis=1)
        safe = np.where(norms == 0.0, 1e-12, norms)
        normalised = self._vectors / safe[:, None]
        scores = normalised @ q
        # Get indices of top_n scores above threshold
        valid_indices = np.where(scores >= threshold)[0]
        if len(valid_indices) == 0:
            return []
        # Sort by score descending
        sorted_indices = valid_indices[np.argsort(scores[valid_indices])[::-1]]
        top_indices = sorted_indices[:top_n]
        return [
            SimilarityHit(key=self._keys[i], score=float(scores[i]))
            for i in top_indices
        ]

    def clear(self) -> None:
        self._keys.clear()
        self._vectors = np.zeros((0, self._dimension), dtype=np.float32)
