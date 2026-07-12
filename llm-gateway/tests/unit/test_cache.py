"""Unit tests for the semantic cache subsystem (Phase 2).

Covers:
  - HashEmbedder: determinism, dimension, L2 normalisation, empty input.
  - SimilarityIndex: add, search, remove, clear, threshold behaviour.
  - cosine_similarity / best_match: pure math helpers.
  - VectorStore: put / get_payload / search / remove / clear / health
    using fakeredis.
  - QualityGate: accept/reject paths.
  - Gateway cache integration: cache miss → write → hit, bypass_cache,
    fail-open on cache errors, degraded responses not cached.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.api.schemas.requests import ChatCompletionRequest, Message
from app.api.schemas.responses import (
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Usage,
)
from app.cache.embedder import HashEmbedder, get_embedder
from app.cache.quality_gate import passes_quality_gate
from app.cache.similarity import SimilarityHit, SimilarityIndex, best_match, cosine_similarity
from app.cache.vector_store import VectorStore
from app.core.gateway import Gateway
from app.router.selector import ProviderSelector
from app.router.classifier import RequestClassifier
from app.router.tier_config import get_default_tier_config


# ── Helpers ────────────────────────────────────────────────────────


def _make_response(
    content: str = "Hello back.",
    *,
    degraded: bool = False,
    finish_reason: str = "stop",
    source: str = "cloud",
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="resp-1",
        created=1700000000,
        model="gpt-4o-mini",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        source=source,
        degraded=degraded,
    )


def _make_request(content: str = "Hello world") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="llama3.2",
        messages=[Message(role="user", content=content)],
    )


class FakeProvider:
    """Minimal provider stub for gateway tests."""

    def __init__(self, response: ChatCompletionResponse | None = None, name: str = "local") -> None:
        self._response = response or _make_response(source=name)
        self.generate_calls = 0
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def cost_per_1k(self) -> float:
        return 0.0

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.generate_calls += 1
        return self._response

    async def health(self) -> bool:
        return True


# ── HashEmbedder tests ────────────────────────────────────────────


class TestHashEmbedder:
    def test_dimension(self) -> None:
        emb = HashEmbedder(dimension=128)
        assert emb.dimension == 128

    def test_embed_returns_correct_shape(self) -> None:
        emb = HashEmbedder(dimension=64)
        vec = emb.embed("hello world")
        assert vec.shape == (64,)
        assert vec.dtype == np.float32

    def test_deterministic_same_input(self) -> None:
        emb = HashEmbedder(dimension=128)
        v1 = emb.embed("the quick brown fox")
        v2 = emb.embed("the quick brown fox")
        np.testing.assert_array_equal(v1, v2)

    def test_different_inputs_different_vectors(self) -> None:
        emb = HashEmbedder(dimension=128)
        v1 = emb.embed("hello world")
        v2 = emb.embed("completely different text")
        assert not np.array_equal(v1, v2)

    def test_l2_normalised(self) -> None:
        emb = HashEmbedder(dimension=256)
        vec = emb.embed("some non-empty text with words")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_empty_string_returns_zero_vector(self) -> None:
        emb = HashEmbedder(dimension=64)
        vec = emb.embed("")
        assert float(np.linalg.norm(vec)) == 0.0

    def test_similar_texts_have_higher_similarity_than_dissimilar(self) -> None:
        emb = HashEmbedder(dimension=256)
        v1 = emb.embed("what is the weather today")
        v2 = emb.embed("what is the weather today")
        v3 = emb.embed("xyzzy qux frobnicate")
        sim_same = cosine_similarity(v1, v2)
        sim_diff = cosine_similarity(v1, v3)
        assert sim_same == pytest.approx(1.0)
        assert sim_same > sim_diff

    def test_invalid_dimension_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension must be positive"):
            HashEmbedder(dimension=0)

    def test_get_embedder_factory_hash(self) -> None:
        emb = get_embedder(kind="hash", dimension=32)
        assert isinstance(emb, HashEmbedder)
        assert emb.dimension == 32

    def test_get_embedder_factory_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="unknown embedder kind"):
            get_embedder(kind="nonexistent")


# ── Similarity pure-function tests ────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self) -> None:
        a = np.zeros(3, dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0


class TestBestMatch:
    def test_empty_returns_none(self) -> None:
        vectors = np.zeros((0, 4), dtype=np.float32)
        assert best_match(np.zeros(4), vectors, []) is None

    def test_returns_highest_scoring(self) -> None:
        query = np.array([1.0, 0.0], dtype=np.float32)
        vectors = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        keys = ["a", "b", "c"]
        hit = best_match(query, vectors, keys)
        assert hit is not None
        assert hit.key == "b"


# ── SimilarityIndex tests ─────────────────────────────────────────


class TestSimilarityIndex:
    def test_add_and_search(self) -> None:
        idx = SimilarityIndex(dimension=4)
        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        idx.add("key1", v)
        assert idx.size == 1

        hit = idx.search(v, threshold=0.9)
        assert hit is not None
        assert hit.key == "key1"
        assert hit.score == pytest.approx(1.0)

    def test_search_below_threshold_returns_none(self) -> None:
        idx = SimilarityIndex(dimension=4)
        idx.add("key1", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        query = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        hit = idx.search(query, threshold=0.5)
        assert hit is None

    def test_search_empty_index(self) -> None:
        idx = SimilarityIndex(dimension=4)
        hit = idx.search(np.zeros(4, dtype=np.float32), threshold=0.0)
        assert hit is None

    def test_add_replaces_existing_key(self) -> None:
        idx = SimilarityIndex(dimension=4)
        idx.add("key1", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        idx.add("key1", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        assert idx.size == 1  # not duplicated

        hit = idx.search(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), 0.9)
        assert hit is not None
        assert hit.key == "key1"

    def test_dimension_mismatch_raises(self) -> None:
        idx = SimilarityIndex(dimension=4)
        with pytest.raises(ValueError, match="dimension mismatch"):
            idx.add("bad", np.zeros(3, dtype=np.float32))

    def test_remove(self) -> None:
        idx = SimilarityIndex(dimension=4)
        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        idx.add("key1", v)
        assert idx.remove("key1") is True
        assert idx.size == 0
        assert idx.remove("key1") is False  # already gone

    def test_remove_nonexistent(self) -> None:
        idx = SimilarityIndex(dimension=4)
        assert idx.remove("nope") is False

    def test_clear(self) -> None:
        idx = SimilarityIndex(dimension=4)
        idx.add("a", np.zeros(4, dtype=np.float32))
        idx.add("b", np.zeros(4, dtype=np.float32))
        idx.clear()
        assert idx.size == 0

    def test_search_returns_best_of_multiple(self) -> None:
        idx = SimilarityIndex(dimension=2)
        idx.add("a", np.array([1.0, 0.0], dtype=np.float32))
        idx.add("b", np.array([0.9, 0.1], dtype=np.float32))
        idx.add("c", np.array([0.0, 1.0], dtype=np.float32))
        query = np.array([1.0, 0.0], dtype=np.float32)
        hit = idx.search(query, threshold=0.5)
        assert hit is not None
        assert hit.key == "a"


# ── VectorStore tests (with fakeredis) ────────────────────────────


@pytest.fixture()
async def fake_store():
    """A VectorStore backed by fakeredis."""
    import fakeredis.aioredis

    redis = fakeredis.aioredis.FakeRedis()
    store = VectorStore(redis=redis, embedder_dimension=64, ttl_seconds=300)
    await store.initialize()
    yield store
    await store.close()


class TestVectorStore:
    @pytest.mark.asyncio
    async def test_put_and_search(self, fake_store: VectorStore) -> None:
        emb = HashEmbedder(dimension=64)
        vec = emb.embed("hello world")
        payload = _make_response().model_dump(mode="json")

        await fake_store.put("key1", vec, payload)

        # Search with the same vector should find it
        hit = await fake_store.search(vec, threshold=0.9)
        assert hit is not None
        assert hit.key == "key1"

    @pytest.mark.asyncio
    async def test_get_payload(self, fake_store: VectorStore) -> None:
        emb = HashEmbedder(dimension=64)
        vec = emb.embed("test query")
        payload = _make_response("cached answer").model_dump(mode="json")
        await fake_store.put("key1", vec, payload)

        result = await fake_store.get_payload("key1")
        assert result is not None
        assert result["choices"][0]["message"]["content"] == "cached answer"

    @pytest.mark.asyncio
    async def test_get_payload_missing_key(self, fake_store: VectorStore) -> None:
        result = await fake_store.get_payload("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_remove(self, fake_store: VectorStore) -> None:
        emb = HashEmbedder(dimension=64)
        vec = emb.embed("to be removed")
        payload = _make_response().model_dump(mode="json")
        await fake_store.put("key1", vec, payload)

        assert await fake_store.remove("key1") is True
        assert await fake_store.get_payload("key1") is None
        assert await fake_store.search(vec, threshold=0.5) is None

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, fake_store: VectorStore) -> None:
        assert await fake_store.remove("nope") is False

    @pytest.mark.asyncio
    async def test_clear(self, fake_store: VectorStore) -> None:
        emb = HashEmbedder(dimension=64)
        vec = emb.embed("clear me")
        payload = _make_response().model_dump(mode="json")
        await fake_store.put("key1", vec, payload)
        assert fake_store.size == 1

        await fake_store.clear()
        assert fake_store.size == 0
        assert await fake_store.get_payload("key1") is None

    @pytest.mark.asyncio
    async def test_health_ok(self, fake_store: VectorStore) -> None:
        assert await fake_store.health() is True

    @pytest.mark.asyncio
    async def test_initialize_rebuilds_index(self) -> None:
        """initialize() should rebuild the in-memory index from Redis."""
        import fakeredis.aioredis

        redis = fakeredis.aioredis.FakeRedis()
        emb = HashEmbedder(dimension=64)
        vec = emb.embed("persisted entry")
        payload = _make_response().model_dump(mode="json")

        # Write directly, then create a *new* store and initialize
        store1 = VectorStore(redis=redis, embedder_dimension=64, ttl_seconds=300)
        await store1.put("pk1", vec, payload)
        await store1.close()

        store2 = VectorStore(redis=redis, embedder_dimension=64, ttl_seconds=300)
        await store2.initialize()
        assert store2.size == 1

        hit = await store2.search(vec, threshold=0.9)
        assert hit is not None
        assert hit.key == "pk1"
        await store2.close()

    @pytest.mark.asyncio
    async def test_search_no_match(self, fake_store: VectorStore) -> None:
        emb = HashEmbedder(dimension=64)
        vec = emb.embed("hello world")
        payload = _make_response().model_dump(mode="json")
        await fake_store.put("key1", vec, payload)

        # Search with a very different vector and high threshold
        other_vec = emb.embed("completely different unrelated text xyzzy")
        hit = await fake_store.search(other_vec, threshold=0.99)
        assert hit is None


# ── QualityGate tests ─────────────────────────────────────────────


class TestQualityGate:
    def test_good_response_passes(self) -> None:
        resp = _make_response("A good answer.")
        assert passes_quality_gate(resp) is True

    def test_degraded_rejected(self) -> None:
        resp = _make_response("fallback", degraded=True)
        assert passes_quality_gate(resp) is False

    def test_empty_content_rejected(self) -> None:
        resp = _make_response("   ")
        assert passes_quality_gate(resp) is False

    def test_no_choices_rejected(self) -> None:
        resp = ChatCompletionResponse(
            id="r",
            created=1,
            model="m",
            choices=[],
            usage=Usage(),
            source="cloud",
        )
        assert passes_quality_gate(resp) is False

    def test_bad_finish_reason_rejected(self) -> None:
        # "length" should be rejected (truncated due to token limit)
        resp = _make_response("text", finish_reason="length")
        assert passes_quality_gate(resp) is False

        # "content_filter" should also be rejected
        resp_bad = ChatCompletionResponse.model_construct(
            id="r",
            created=1,
            model="m",
            choices=[
                Choice.model_construct(
                    index=0,
                    message=ChoiceMessage(content="text"),
                    finish_reason="content_filter",
                )
            ],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            source="cloud",
            degraded=False,
        )
        assert passes_quality_gate(resp_bad) is False

    def test_mid_sentence_truncation_rejected(self) -> None:
        # Response without terminal punctuation should be rejected
        resp = _make_response("This is an incomplete sentence without punctuation")
        assert passes_quality_gate(resp) is False

        # Response with terminal punctuation should pass
        resp_good = _make_response("This is a complete sentence.")
        assert passes_quality_gate(resp_good) is True

        # Response ending with question mark should pass
        resp_q = _make_response("What is the capital of India?")
        assert passes_quality_gate(resp_q) is True

        # Response ending with exclamation should pass
        resp_excl = _make_response("Hello world!")
        assert passes_quality_gate(resp_excl) is True

    def test_no_usage_rejected(self) -> None:
        resp = ChatCompletionResponse(
            id="r",
            created=1,
            model="m",
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(content="text"),
                    finish_reason="stop",
                )
            ],
            source="cloud",
        )
        # usage defaults to Usage() with all zeros — not None
        # The gate checks `usage is None`, so default Usage passes the None check
        # but zeros are non-negative so it passes. Let's test actual None:
        resp_no_usage = resp.model_copy(update={"usage": None})  # type: ignore[arg-type]
        assert passes_quality_gate(resp_no_usage) is False

    def test_negative_usage_rejected(self) -> None:
        resp = ChatCompletionResponse(
            id="r",
            created=1,
            model="m",
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(content="text"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=-1, completion_tokens=0, total_tokens=0),
            source="cloud",
        )
        assert passes_quality_gate(resp) is False


# ── Gateway cache integration tests ───────────────────────────────


class TestGatewayCache:
    @pytest.mark.asyncio
    async def test_cache_miss_then_hit(self) -> None:
        """First request misses cache and writes; second request hits."""
        embedder = HashEmbedder(dimension=64)
        provider = FakeProvider(name="local")
        store = MagicMock(spec=VectorStore)
        store.search = AsyncMock(return_value=None)
        store.get_payload = AsyncMock(return_value=None)
        store.put = AsyncMock()
        store.health = AsyncMock(return_value=True)

        # Create router with provider
        tier_configs = get_default_tier_config()
        classifier = RequestClassifier(tier_configs)
        router = ProviderSelector(
            providers={"local": provider},
            tier_configs=tier_configs,
            classifier=classifier,
        )

        gateway = Gateway(
            router=router,
            cache=store,
            embedder=embedder,
            similarity_threshold=0.85,
        )

        request = _make_request("What is 2+2?")
        resp1 = await gateway.handle(request)
        assert resp1.source == "local"
        assert provider.generate_calls == 1
        assert store.put.call_count == 1

        # Second call: simulate cache hit by returning a key + payload
        cached_payload = resp1.model_dump(mode="json")
        store.search = AsyncMock(
            return_value=SimilarityHit(key="abc", score=0.95)
        )
        store.get_payload = AsyncMock(return_value=cached_payload)

        resp2 = await gateway.handle(request)
        assert resp2.source == "cache"
        assert provider.generate_calls == 1  # provider NOT called again

    @pytest.mark.asyncio
    async def test_bypass_cache_skips_lookup(self) -> None:
        """bypass_cache=True skips cache lookup but still writes."""
        embedder = HashEmbedder(dimension=64)
        provider = FakeProvider(name="local")
        store = MagicMock(spec=VectorStore)
        store.search = AsyncMock(return_value=None)
        store.get_payload = AsyncMock(return_value=None)
        store.put = AsyncMock()
        store.health = AsyncMock(return_value=True)

        tier_configs = get_default_tier_config()
        classifier = RequestClassifier(tier_configs)
        router = ProviderSelector(
            providers={"local": provider},
            tier_configs=tier_configs,
            classifier=classifier,
        )

        gateway = Gateway(router=router, cache=store, embedder=embedder)

        request = _make_request("Hello")
        request.bypass_cache = True
        await gateway.handle(request)

        # search should NOT be called (bypass), but put SHOULD be called
        store.search.assert_not_called()
        store.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_lookup_failure_is_fail_open(self) -> None:
        """If cache.search raises, the gateway still serves from provider."""
        embedder = HashEmbedder(dimension=64)
        provider = FakeProvider(name="local")
        store = MagicMock(spec=VectorStore)
        store.search = AsyncMock(side_effect=RuntimeError("redis down"))
        store.get_payload = AsyncMock(return_value=None)
        store.put = AsyncMock()
        store.health = AsyncMock(return_value=True)

        tier_configs = get_default_tier_config()
        classifier = RequestClassifier(tier_configs)
        router = ProviderSelector(
            providers={"local": provider},
            tier_configs=tier_configs,
            classifier=classifier,
        )

        gateway = Gateway(router=router, cache=store, embedder=embedder)

        resp = await gateway.handle(_make_request("Hello"))
        assert resp.source == "local"
        assert provider.generate_calls == 1

    @pytest.mark.asyncio
    async def test_cache_write_failure_is_fail_open(self) -> None:
        """If cache.put raises, the response is still returned."""
        embedder = HashEmbedder(dimension=64)
        provider = FakeProvider(name="local")
        store = MagicMock(spec=VectorStore)
        store.search = AsyncMock(return_value=None)
        store.get_payload = AsyncMock(return_value=None)
        store.put = AsyncMock(side_effect=RuntimeError("write error"))
        store.health = AsyncMock(return_value=True)

        tier_configs = get_default_tier_config()
        classifier = RequestClassifier(tier_configs)
        router = ProviderSelector(
            providers={"local": provider},
            tier_configs=tier_configs,
            classifier=classifier,
        )

        gateway = Gateway(router=router, cache=store, embedder=embedder)

        resp = await gateway.handle(_make_request("Hello"))
        assert resp.source == "local"
        assert provider.generate_calls == 1

    @pytest.mark.asyncio
    async def test_degraded_response_not_cached(self) -> None:
        """A degraded response passes through but is not written to cache."""
        embedder = HashEmbedder(dimension=64)
        degraded_resp = _make_response("fallback", degraded=True)
        provider = FakeProvider(response=degraded_resp)
        store = MagicMock(spec=VectorStore)
        store.search = AsyncMock(return_value=None)
        store.get_payload = AsyncMock(return_value=None)
        store.put = AsyncMock()
        store.health = AsyncMock(return_value=True)

        tier_configs = get_default_tier_config()
        classifier = RequestClassifier(tier_configs)
        router = ProviderSelector(
            providers={"local": provider},
            tier_configs=tier_configs,
            classifier=classifier,
        )

        gateway = Gateway(router=router, cache=store, embedder=embedder)

        resp = await gateway.handle(_make_request("Hello"))
        assert resp.degraded is True
        store.put.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_cache_no_embedder(self) -> None:
        """Gateway works fine without cache or embedder (Phase 1 mode)."""
        provider = FakeProvider(name="local")
        tier_configs = get_default_tier_config()
        classifier = RequestClassifier(tier_configs)
        router = ProviderSelector(
            providers={"local": provider},
            tier_configs=tier_configs,
            classifier=classifier,
        )
        gateway = Gateway(router=router)

        resp = await gateway.handle(_make_request("Hello"))
        assert resp.source == "local"
        assert provider.generate_calls == 1

    @pytest.mark.asyncio
    async def test_cache_key_is_deterministic(self) -> None:
        """Same request produces same cache key."""
        req1 = _make_request("What is the weather?")
        req2 = _make_request("What is the weather?")
        key1 = Gateway._cache_key(req1)
        key2 = Gateway._cache_key(req2)
        assert key1 == key2

        # Different request → different key
        req3 = _make_request("What is the time?")
        assert Gateway._cache_key(req3) != key1

    @pytest.mark.asyncio
    async def test_request_to_text_flattens_messages(self) -> None:
        req = ChatCompletionRequest(
            model="gpt-4o-mini",
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="What is 2+2?"),
            ],
        )
        text = Gateway._request_to_text(req)
        assert "You are helpful." in text
        assert "What is 2+2?" in text

    @pytest.mark.asyncio
    async def test_cache_health_no_cache(self) -> None:
        provider = FakeProvider(name="local")
        tier_configs = get_default_tier_config()
        classifier = RequestClassifier(tier_configs)
        router = ProviderSelector(
            providers={"local": provider},
            tier_configs=tier_configs,
            classifier=classifier,
        )
        gateway = Gateway(router=router)
        assert await gateway.cache_health() is True

    @pytest.mark.asyncio
    async def test_cache_health_with_store(self) -> None:
        store = MagicMock(spec=VectorStore)
        store.health = AsyncMock(return_value=False)
        provider = FakeProvider(name="local")
        tier_configs = get_default_tier_config()
        classifier = RequestClassifier(tier_configs)
        router = ProviderSelector(
            providers={"local": provider},
            tier_configs=tier_configs,
            classifier=classifier,
        )
        gateway = Gateway(router=router, cache=store)
        assert await gateway.cache_health() is False


# ── Invalidation tests ────────────────────────────────────────────


class TestInvalidation:
    @pytest.mark.asyncio
    async def test_invalidate_existing(self, fake_store: VectorStore) -> None:
        from app.cache.invalidation import invalidate

        emb = HashEmbedder(dimension=64)
        vec = emb.embed("invalidate me")
        payload = _make_response().model_dump(mode="json")
        await fake_store.put("inv1", vec, payload)

        result = await invalidate(fake_store, "inv1")
        assert result is True
        assert await fake_store.get_payload("inv1") is None

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent(self, fake_store: VectorStore) -> None:
        from app.cache.invalidation import invalidate

        result = await invalidate(fake_store, "does-not-exist")
        assert result is False

    @pytest.mark.asyncio
    async def test_invalidate_all(self, fake_store: VectorStore) -> None:
        from app.cache.invalidation import invalidate_all

        emb = HashEmbedder(dimension=64)
        vec = emb.embed("clear all")
        payload = _make_response().model_dump(mode="json")
        await fake_store.put("k1", vec, payload)
        await fake_store.put("k2", vec, payload)
        assert fake_store.size == 2

        await invalidate_all(fake_store)
        assert fake_store.size == 0

    @pytest.mark.asyncio
    async def test_invalidate_fail_open(self) -> None:
        """invalidate should not raise even if the store errors."""
        from app.cache.invalidation import invalidate

        store = MagicMock(spec=VectorStore)
        store.remove = AsyncMock(side_effect=RuntimeError("boom"))
        result = await invalidate(store, "key")
        assert result is False

    @pytest.mark.asyncio
    async def test_invalidate_all_fail_open(self) -> None:
        from app.cache.invalidation import invalidate_all

        store = MagicMock(spec=VectorStore)
        store.clear = AsyncMock(side_effect=RuntimeError("boom"))
        # Should not raise
        await invalidate_all(store)
