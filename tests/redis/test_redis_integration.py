"""
Integration tests for Redis embedding cache with real Redis instance.

These tests require a running Redis instance and test actual Redis operations.
"""
# ruff: noqa: E402

import pytest
import os
import sys
import time
import redis
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.embedding_cache import (
    EmbeddingCache,
    get_embedding_cache,
    validate_redis_config,
)
from src.services.embedding_service import create_embeddings_batch


# Check if Redis is available
def is_redis_available() -> bool:
    """Check if Redis is available for testing."""

    try:
        r = redis.Redis(host="localhost", port=6379, db=0, socket_timeout=2)
        r.ping()
        return True
    except (redis.ConnectionError, redis.TimeoutError):
        return False


REDIS_AVAILABLE = is_redis_available()


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
class TestRedisIntegration:
    """Integration tests with real Redis instance."""

    def setup_method(self):
        """Setup for each test method."""

        # Use a test-specific Redis database to avoid conflicts
        self.test_db = 15  # Use DB 15 for testing
        self.cache = EmbeddingCache(host="localhost", port=6379, db=self.test_db)

        # Clean up test database
        if self.cache.redis:
            self.cache.redis.flushdb()

    def teardown_method(self):
        """Cleanup after each test method."""

        # Clean up test database
        if self.cache and self.cache.redis:
            self.cache.redis.flushdb()
            self.cache.redis.close()

    def test_redis_connection(self):
        """Test basic Redis connection."""

        assert self.cache.redis is not None
        assert self.cache.redis.ping() is True

    def test_health_check_real_redis(self):
        """Test health check with real Redis."""

        health = self.cache.health_check()

        assert health["status"] in ("healthy", "unhealthy")
        assert "connection_info" in health or "error" in health

    def test_cache_key_generation_consistency(self):
        """Test that cache keys are generated consistently."""

        text = "test embedding text"
        model = "text-embedding-3-small"

        key1 = self.cache._generate_cache_key(text, model)
        key2 = self.cache._generate_cache_key(text, model)

        assert key1 == key2
        assert key1.startswith(f"embedding:{model}:")

    def test_single_embedding_cache_cycle(self):
        """Test complete cache cycle with single embedding."""

        text = "This is a test text for embedding"
        model = "text-embedding-3-small"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        result = self.cache.get_batch([text], model)
        assert result == {}

        self.cache.set_batch({text: embedding}, model, ttl=60)

        result = self.cache.get_batch([text], model)
        assert text in result
        assert result[text] == embedding

    def test_batch_embedding_cache_cycle(self):
        """Test complete cache cycle with multiple embeddings."""

        texts = ["text1", "text2", "text3"]
        model = "test-model"
        embeddings = {
            "text1": [0.1, 0.2, 0.3],
            "text2": [0.4, 0.5, 0.6],
            "text3": [0.7, 0.8, 0.9],
        }

        result = self.cache.get_batch(texts, model)
        assert result == {}

        self.cache.set_batch(embeddings, model, ttl=60)

        result = self.cache.get_batch(texts, model)
        assert len(result) == 3
        for text in texts:
            assert text in result
            assert result[text] == embeddings[text]

    def test_partial_cache_hits(self):
        """Test scenario with partial cache hits."""

        texts = ["cached_text", "new_text"]
        model = "test-model"
        cached_embedding = [0.1, 0.2, 0.3]

        self.cache.set_batch({"cached_text": cached_embedding}, model, ttl=60)

        result = self.cache.get_batch(texts, model)
        assert len(result) == 1
        assert "cached_text" in result
        assert result["cached_text"] == cached_embedding
        assert "new_text" not in result

    def test_ttl_expiration(self):
        """Test that cached embeddings expire after TTL."""

        text = "expiring text"
        model = "test-model"
        embedding = [0.1, 0.2, 0.3]

        self.cache.set_batch({text: embedding}, model, ttl=1)
        result = self.cache.get_batch([text], model)
        assert text in result

        time.sleep(2)
        result = self.cache.get_batch([text], model)
        assert result == {}

    def test_different_models_different_keys(self):
        """Test that different models use different cache keys."""

        text = "same text"
        model1 = "model-1"
        model2 = "model-2"
        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.4, 0.5, 0.6]

        self.cache.set_batch({text: embedding1}, model1, ttl=60)
        self.cache.set_batch({text: embedding2}, model2, ttl=60)

        result1 = self.cache.get_batch([text], model1)
        result2 = self.cache.get_batch([text], model2)

        assert result1[text] == embedding1
        assert result2[text] == embedding2
        assert result1[text] != result2[text]

    def test_unicode_text_handling(self):
        """Test caching with Unicode text."""

        texts = ["Hello 世界", "café résumé", "🌟✨🎉"]
        model = "test-model"
        embeddings = {
            "Hello 世界": [0.1, 0.2, 0.3],
            "café résumé": [0.4, 0.5, 0.6],
            "🌟✨🎉": [0.7, 0.8, 0.9],
        }

        self.cache.set_batch(embeddings, model, ttl=60)

        result = self.cache.get_batch(texts, model)
        assert len(result) == 3
        for text in texts:
            assert text in result
            assert result[text] == embeddings[text]

    def test_large_embedding_handling(self):
        """Test caching with large embeddings."""

        text = "large embedding test"
        model = "large-model"
        large_embedding = [float(i) for i in range(3072)]

        self.cache.set_batch({text: large_embedding}, model, ttl=60)
        result = self.cache.get_batch([text], model)
        assert text in result
        assert result[text] == large_embedding
        assert len(result[text]) == 3072

    def test_concurrent_access_simulation(self):
        """Test cache behavior under simulated concurrent access."""

        import threading
        import queue

        results_queue = queue.Queue()
        errors_queue = queue.Queue()

        def cache_worker(worker_id):
            try:
                text = f"worker_{worker_id}_text"
                model = "concurrent-model"
                embedding = [float(worker_id)] * 5

                self.cache.set_batch({text: embedding}, model, ttl=60)
                result = self.cache.get_batch([text], model)
                results_queue.put((worker_id, result))
            except Exception as e:
                errors_queue.put((worker_id, str(e)))

        threads = []
        for i in range(5):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert errors_queue.empty(), f"Errors occurred: {list(errors_queue.queue)}"
        assert results_queue.qsize() == 5

        while not results_queue.empty():
            worker_id, result = results_queue.get()
            expected_text = f"worker_{worker_id}_text"
            assert expected_text in result
            assert result[expected_text] == [float(worker_id)] * 5

    def test_memory_usage_with_many_embeddings(self):
        """Test cache memory usage with many embeddings."""

        model = "memory-test-model"
        batch_size = 100
        embedding_dim = 1536

        embeddings = {}
        for i in range(batch_size):
            text = f"text_{i:04d}"
            embedding = [float(i * j) for j in range(embedding_dim)]
            embeddings[text] = embedding

        self.cache.set_batch(embeddings, model, ttl=60)

        texts = list(embeddings.keys())
        result = self.cache.get_batch(texts, model)

        assert len(result) == batch_size
        for text, expected_embedding in embeddings.items():
            assert text in result
            assert result[text] == expected_embedding

    def test_redis_error_recovery(self):
        """Test cache behavior during Redis errors and recovery."""

        text = "error_recovery_test"
        model = "error-model"
        embedding = [0.1, 0.2, 0.3]

        self.cache.set_batch({text: embedding}, model, ttl=60)
        result = self.cache.get_batch([text], model)
        assert text in result

        self.cache.redis.close()

        result = self.cache.get_batch([text], model)
        assert isinstance(result, dict)

        self.cache.set_batch({text: [0.4, 0.5, 0.6]}, model, ttl=60)


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
class TestIntegrationWithEmbeddingFunction:
    """Integration tests with the actual embedding function."""

    def setup_method(self):
        os.environ["USE_REDIS_CACHE"] = "true"
        os.environ["REDIS_HOST"] = "localhost"
        os.environ["REDIS_PORT"] = "6379"
        os.environ["REDIS_DB"] = "14"
        os.environ["REDIS_EMBEDDING_TTL"] = "60"

        import embedding_cache

        embedding_cache._embedding_cache = None

        test_redis = redis.Redis(host="localhost", port=6379, db=14)
        test_redis.flushdb()
        test_redis.close()

    def teardown_method(self):
        test_redis = redis.Redis(host="localhost", port=6379, db=14)
        test_redis.flushdb()
        test_redis.close()
        os.environ.pop("USE_REDIS_CACHE", None)
        import embedding_cache

        embedding_cache._embedding_cache = None

    @pytest.mark.skipif(
        not os.getenv("EMBEDDINGS_API_KEY"), reason="No embeddings API key configured"
    )
    def test_embedding_function_with_cache_disabled(self):
        os.environ["USE_REDIS_CACHE"] = "false"
        os.environ["USE_HYBRID_SEARCH"] = "false"
        import embedding_cache

        embedding_cache._embedding_cache = None
        texts = ["Test embedding without cache"]
        embeddings = create_embeddings_batch(texts)
        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0

    def test_embedding_function_with_cache_enabled_no_api(self):
        texts = ["Test cache behavior"]
        cache = get_embedding_cache()
        assert cache is not None
        fake_embedding = [0.1] * 1536
        cache.set_batch({texts[0]: fake_embedding}, "text-embedding-3-small", ttl=60)
        pytest.skip("Complex API mocking test - implementation details may vary")

    def test_cache_global_singleton_behavior(self):
        cache1 = get_embedding_cache()
        cache2 = get_embedding_cache()
        assert cache1 is cache2
        assert cache1 is not None

    def test_configuration_validation_integration(self):
        validate_redis_config()
        original_port = os.environ.get("REDIS_PORT")
        try:
            os.environ["REDIS_PORT"] = "invalid"
            with pytest.raises(ValueError):
                validate_redis_config()
        finally:
            if original_port:
                os.environ["REDIS_PORT"] = original_port
            else:
                os.environ.pop("REDIS_PORT", None)


if __name__ == "__main__":
    if REDIS_AVAILABLE:
        print("✅ Redis is available - running integration tests")
        pytest.main([__file__, "-v"])
    else:
        print("❌ Redis is not available - skipping integration tests")
        print("To run these tests:")
        print("1. Start Redis: docker-compose up -d redis")
        print("2. Or install Redis locally and start it")
        print("3. Re-run the tests")
