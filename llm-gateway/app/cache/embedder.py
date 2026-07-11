"""Embedders — turn text into fixed-size vectors for similarity search.

Two implementations are provided:

* ``HashEmbedder`` — deterministic, dependency-free, fast. Good enough for
  semantic-cache demos and tests where you control the query distribution.
  It hashes token n-grams into a fixed-dimensional vector and L2-normalises
  the result, so cosine similarity reduces to a dot product.

* ``SentenceTransformerEmbedder`` — wraps a real sentence-transformers model.
  Loaded lazily so the heavy dependency is only required when configured.

Both implement the :class:`Embedder` protocol so the cache layer is agnostic
to the embedding strategy.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Turn a piece of text into a normalised float vector."""

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the produced vectors."""
        ...

    def embed(self, text: str) -> np.ndarray:
        """Return an L2-normalised embedding for ``text``."""
        ...


def _tokenize(text: str) -> list[str]:
    """Cheap, deterministic, lowercase whitespace + punctuation tokenizer."""
    import re

    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t]


def _ngrams(tokens: list[str], n: int) -> list[str]:
    """Generate word n-grams from a token list."""
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class HashEmbedder:
    """Deterministic hash-based embedder — no external model required.

    Hashes unigrams + bigrams into a fixed-size vector. Identical inputs
    always produce identical vectors; semantically similar inputs share many
    n-gram buckets and therefore have high cosine similarity. This is *not*
    a substitute for a real embedding model in production, but it is
    deterministic, fast, and dependency-free — ideal for tests and demos.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype=np.float32)
        tokens = _tokenize(text)
        if not tokens:
            return vec

        # Combine unigrams and bigrams for a bit of context.
        grams = _ngrams(tokens, 1) + _ngrams(tokens, 2)
        for gram in grams:
            # Stable hash across processes (no PYTHONHASHSEED dependence).
            h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h, "little") % self._dimension
            # Sign the contribution so antonyms don't pile up positively.
            sign = 1.0 if (h[0] & 1) == 0 else -1.0
            vec[idx] += sign

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model (loaded lazily).

    The ``sentence-transformers`` package is an optional dependency. Import
    it only when this embedder is actually constructed so the gateway can
    run without it installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "sentence-transformers is required for "
                "SentenceTransformerEmbedder; install it with "
                "`pip install sentence-transformers`"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self._dimension = int(self._model.get_sentence_embedding_dimension() or 384)
        logger.info(
            "sentence_transformer_loaded",
            extra={"model": model_name, "dimension": self._dimension},
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        vec = np.asarray(self._model.encode(text, normalize_embeddings=True), dtype=np.float32)
        return vec


def get_embedder(kind: str = "hash", dimension: int = 256, model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Factory used by the gateway to build an embedder from config."""
    if kind == "hash":
        return HashEmbedder(dimension=dimension)
    if kind == "sentence_transformer":
        return SentenceTransformerEmbedder(model_name=model_name)
    raise ValueError(f"unknown embedder kind: {kind!r}")
