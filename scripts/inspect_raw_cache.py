import asyncio
import json
import hashlib
from app.core.config import Settings
from app.cache.embedder import get_embedder
from app.cache.vector_store import VectorStore
import redis.asyncio as redis

async def main():
    settings = Settings()
    embedder = get_embedder(settings.cache_embedder_kind, settings.cache_embedder_dimension)
    redis_client = redis.from_url(settings.cache_redis_url)
    vector_store = VectorStore(redis_client, embedder.dimension, ttl_seconds=settings.cache_ttl_seconds)
    # key for prompt
    prompt = "What is the capital of France?"
    key = hashlib.sha256(prompt.encode()).hexdigest()
    payload = await vector_store.get_payload(key)
    print('Payload for key:', key)
    print(json.dumps(payload, indent=2))
    await redis_client.close()

if __name__ == "__main__":
    asyncio.run(main())
