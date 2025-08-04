"""
Unit tests for EmbeddingCache with Redis integration.

Tests the core functionality of the Redis-based embedding cache with mocked Redis operations.
"""

import pytest
import os
import sys
import pickle
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from embedding_cache import (
    EmbeddingCache,
    CircuitBreaker,
    CircuitState,
    get_embedding_cache,
    validate_redis_config,
)


class TestCircuitBreaker:
    """Test cases for CircuitBreaker class."""

    def test_init_default_values(self):
        """Test circuit breaker initialization with default values."""
        breaker = CircuitBreaker()

        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout == 60
        assert breaker.failure_count == 0
        assert breaker.last_failure_time is None
        assert breaker.state == CircuitState.CLOSED

    def test_init_custom_values(self):
        """Test circuit breaker initialization with custom values."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 30
        assert breaker.state == CircuitState.CLOSED

    def test_successful_call_closed_state(self):
        """Test successful function call in closed state."""
        breaker = CircuitBreaker()
        mock_func = Mock(return_value="success")

        result = breaker.call(mock_func, "arg1", key="value")

        assert result == "success"
        assert breaker.failure_count == 0
        assert breaker.state == CircuitState.CLOSED
        mock_func.assert_called_once_with("arg1", key="value")

    def test_failed_call_threshold_not_reached(self):
        """Test failed call that doesn't reach failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3)
        mock_func = Mock(side_effect=Exception("Redis error"))

        result = breaker.call(mock_func)

        assert result is None
        assert breaker.failure_count == 1
        assert breaker.state == CircuitState.CLOSED
        assert breaker.last_failure_time is not None

    def test_failed_call_opens_circuit(self):
        """Test that circuit opens after reaching failure threshold."""
        breaker = CircuitBreaker(failure_threshold=2)
        mock_func = Mock(side_effect=Exception("Redis error"))

        # First failure
        breaker.call(mock_func)
        assert breaker.state == CircuitState.CLOSED

        # Second failure - should open circuit
        breaker.call(mock_func)
        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 2

    def test_call_in_open_state_without_recovery(self):
        """Test call in open state before recovery timeout."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        mock_func = Mock(side_effect=Exception("Redis error"))

        # Trigger circuit opening
        breaker.call(mock_func)
        assert breaker.state == CircuitState.OPEN

        # Try again immediately - should fail fast
        mock_func.reset_mock()
        result = breaker.call(mock_func)

        assert result is None
        assert breaker.state == CircuitState.OPEN
        mock_func.assert_not_called()

    @patch("embedding_cache.time.time")
    def test_call_in_open_state_with_recovery(self, mock_time):
        """Test call in open state after recovery timeout."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        mock_func = Mock(return_value="success")

        # Set initial time and trigger circuit opening
        mock_time.return_value = 1000
        breaker.call(Mock(side_effect=Exception("Redis error")))
        assert breaker.state == CircuitState.OPEN

        # Set time after recovery timeout
        mock_time.return_value = 1070  # 70 seconds later

        result = breaker.call(mock_func)

        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_half_open_to_closed_on_success(self):
        """Test transition from half-open to closed on successful call."""
        breaker = CircuitBreaker()
        breaker.state = CircuitState.HALF_OPEN
        mock_func = Mock(return_value="success")

        result = breaker.call(mock_func)

        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_half_open_to_open_on_failure(self):
        """Test transition from half-open to open on failed call."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.state = CircuitState.HALF_OPEN
        breaker.failure_count = 0  # Reset for half-open state
        mock_func = Mock(side_effect=Exception("Redis error"))

        result = breaker.call(mock_func)

        assert result is None
        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 1


class TestEmbeddingCache:
    """Test cases for EmbeddingCache class."""

    @patch("embedding_cache.redis.Redis")
    def test_init_default_config(self, mock_redis_class):
        """Test cache initialization with default configuration."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        cache = EmbeddingCache()

        assert cache.host == "localhost"
        assert cache.port == 6379
        assert cache.db == 0
        assert cache.default_ttl == 86400
        assert cache.redis == mock_redis_instance
        mock_redis_instance.ping.assert_called_once()

    @patch("embedding_cache.redis.Redis")
    def test_init_custom_config(self, mock_redis_class):
        """Test cache initialization with custom configuration."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_class.return_value = mock_redis_instance

        cache = EmbeddingCache(host="custom-host", port=1234, db=2, password="secret")

        assert cache.host == "custom-host"
        assert cache.port == 1234
        assert cache.db == 2
        assert cache.password == "secret"

    @patch("embedding_cache.redis.Redis")
    def test_init_redis_connection_failure(self, mock_redis_class):
        """Test graceful handling of Redis connection failure."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.side_effect = Exception("Connection failed")
        mock_redis_class.return_value = mock_redis_instance

        cache = EmbeddingCache()

        assert cache.redis is None  # Should set to None on failure

    def test_generate_cache_key(self):
        """Test cache key generation."""
        cache = EmbeddingCache.__new__(EmbeddingCache)  # Skip __init__

        key1 = cache._generate_cache_key("hello world", "text-embedding-3-small")
        key2 = cache._generate_cache_key("hello world", "text-embedding-3-small")
        key3 = cache._generate_cache_key("different text", "text-embedding-3-small")
        key4 = cache._generate_cache_key("hello world", "different-model")

        # Same inputs should generate same key
        assert key1 == key2

        # Different inputs should generate different keys
        assert key1 != key3
        assert key1 != key4

        # Key format should be correct
        assert key1.startswith("embedding:text-embedding-3-small:")
        assert len(key1.split(":")[2]) == 16  # SHA256 hash truncated to 16 chars

    @patch("embedding_cache.redis.Redis")
    def test_get_batch_no_redis(self, mock_redis_class):
        """Test get_batch when Redis is not available."""
        mock_redis_class.return_value.ping.side_effect = Exception("No Redis")
        cache = EmbeddingCache()

        result = cache.get_batch(["text1", "text2"], "model")

        assert result == {}

    @patch("embedding_cache.redis.Redis")
    def test_get_batch_empty_texts(self, mock_redis_class):
        """Test get_batch with empty text list."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_class.return_value = mock_redis_instance
        cache = EmbeddingCache()

        result = cache.get_batch([], "model")

        assert result == {}

    @patch("embedding_cache.redis.Redis")
    def test_get_batch_cache_hits(self, mock_redis_class):
        """Test get_batch with cache hits."""
        # Setup mock Redis
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_pipeline = Mock()
        mock_redis_instance.pipeline.return_value = mock_pipeline
        mock_redis_class.return_value = mock_redis_instance

        # Setup cached embeddings
        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.4, 0.5, 0.6]
        mock_pipeline.execute.return_value = [
            pickle.dumps(embedding1),
            pickle.dumps(embedding2),
        ]

        cache = EmbeddingCache()
        result = cache.get_batch(["text1", "text2"], "model")

        expected = {"text1": embedding1, "text2": embedding2}
        assert result == expected

    @patch("embedding_cache.redis.Redis")
    def test_get_batch_cache_misses(self, mock_redis_class):
        """Test get_batch with cache misses."""
        # Setup mock Redis
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_pipeline = Mock()
        mock_redis_instance.pipeline.return_value = mock_pipeline
        mock_redis_class.return_value = mock_redis_instance

        # Setup cache misses (None values)
        mock_pipeline.execute.return_value = [None, None]

        cache = EmbeddingCache()
        result = cache.get_batch(["text1", "text2"], "model")

        assert result == {}

    @patch("embedding_cache.redis.Redis")
    def test_get_batch_mixed_hits_misses(self, mock_redis_class):
        """Test get_batch with mixed cache hits and misses."""
        # Setup mock Redis
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_pipeline = Mock()
        mock_redis_instance.pipeline.return_value = mock_pipeline
        mock_redis_class.return_value = mock_redis_instance

        # Setup mixed results
        embedding1 = [0.1, 0.2, 0.3]
        mock_pipeline.execute.return_value = [
            pickle.dumps(embedding1),
            None,  # Cache miss
        ]

        cache = EmbeddingCache()
        result = cache.get_batch(["text1", "text2"], "model")

        expected = {"text1": embedding1}
        assert result == expected

    @patch("embedding_cache.redis.Redis")
    def test_get_batch_deserialization_error(self, mock_redis_class):
        """Test get_batch handling deserialization errors gracefully."""
        # Setup mock Redis
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_pipeline = Mock()
        mock_redis_instance.pipeline.return_value = mock_pipeline
        mock_redis_class.return_value = mock_redis_instance

        # Setup corrupted data
        mock_pipeline.execute.return_value = [b"corrupted_data"]

        cache = EmbeddingCache()
        result = cache.get_batch(["text1"], "model")

        assert result == {}  # Should handle error gracefully

    @patch("embedding_cache.redis.Redis")
    def test_set_batch_no_redis(self, mock_redis_class):
        """Test set_batch when Redis is not available."""
        mock_redis_class.return_value.ping.side_effect = Exception("No Redis")
        cache = EmbeddingCache()

        # Should not raise exception
        cache.set_batch({"text": [0.1, 0.2]}, "model")

    @patch("embedding_cache.redis.Redis")
    def test_set_batch_empty_embeddings(self, mock_redis_class):
        """Test set_batch with empty embeddings."""
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_class.return_value = mock_redis_instance
        cache = EmbeddingCache()

        cache.set_batch({}, "model")

        # Should not call pipeline if no embeddings
        mock_redis_instance.pipeline.assert_not_called()

    @patch("embedding_cache.redis.Redis")
    def test_set_batch_success(self, mock_redis_class):
        """Test successful set_batch operation."""
        # Setup mock Redis
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_pipeline = Mock()
        mock_redis_instance.pipeline.return_value = mock_pipeline
        mock_redis_class.return_value = mock_redis_instance

        cache = EmbeddingCache()
        embeddings = {"text1": [0.1, 0.2, 0.3], "text2": [0.4, 0.5, 0.6]}

        cache.set_batch(embeddings, "model", ttl=3600)

        # Verify pipeline operations
        mock_redis_instance.pipeline.assert_called_once()
        assert mock_pipeline.setex.call_count == 2
        mock_pipeline.execute.assert_called_once()

    @patch("embedding_cache.redis.Redis")
    def test_set_batch_serialization_error(self, mock_redis_class):
        """Test set_batch handling serialization errors gracefully."""
        # Setup mock Redis
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_pipeline = Mock()
        mock_redis_instance.pipeline.return_value = mock_pipeline
        mock_redis_class.return_value = mock_redis_instance

        cache = EmbeddingCache()

        # Create an object that can't be pickled
        class UnpicklableObject:
            def __reduce__(self):
                raise TypeError("Cannot pickle this object")

        embeddings = {"text": UnpicklableObject()}

        # Should not raise exception
        cache.set_batch(embeddings, "model")

    @patch("embedding_cache.redis.Redis")
    def test_health_check_no_redis(self, mock_redis_class):
        """Test health check when Redis client is not initialized."""
        mock_redis_class.return_value.ping.side_effect = Exception("No Redis")
        cache = EmbeddingCache()

        health = cache.health_check()

        assert health["status"] == "unhealthy"
        assert "No Redis" in health["error"]  # Should contain the actual error message
        assert "circuit_breaker_state" in health

    @patch("embedding_cache.redis.Redis")
    def test_health_check_success(self, mock_redis_class):
        """Test successful health check."""
        # Setup mock Redis
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.setex.return_value = True
        mock_redis_instance.get.return_value = b"test"
        mock_redis_instance.delete.return_value = 1
        mock_redis_class.return_value = mock_redis_instance

        cache = EmbeddingCache()
        health = cache.health_check()

        assert health["status"] == "healthy"
        assert health["ping"] is True
        assert health["read_write"] is True
        assert "latency_ms" in health
        assert health["latency_ms"] >= 0
        assert "connection_info" in health

    @patch("embedding_cache.redis.Redis")
    def test_health_check_failure(self, mock_redis_class):
        """Test health check with Redis operation failure."""
        # Setup mock Redis
        mock_redis_instance = Mock()
        mock_redis_instance.ping.side_effect = Exception("Redis down")
        mock_redis_class.return_value = mock_redis_instance

        cache = EmbeddingCache()
        health = cache.health_check()

        assert health["status"] == "unhealthy"
        assert "Redis down" in health["error"]

    @patch("embedding_cache.redis.Redis")
    def test_circuit_breaker_integration(self, mock_redis_class):
        """Test circuit breaker integration with Redis operations."""
        # Setup mock Redis that fails
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_pipeline = Mock()
        mock_pipeline.execute.side_effect = Exception("Redis error")
        mock_redis_instance.pipeline.return_value = mock_pipeline
        mock_redis_class.return_value = mock_redis_instance

        cache = EmbeddingCache()
        cache.circuit_breaker.failure_threshold = 2

        # First failure
        result1 = cache.get_batch(["text"], "model")
        assert result1 == {}
        assert cache.circuit_breaker.state == CircuitState.CLOSED

        # Second failure - should open circuit
        result2 = cache.get_batch(["text"], "model")
        assert result2 == {}
        assert cache.circuit_breaker.state == CircuitState.OPEN

        # Third call should fail fast without calling Redis
        mock_pipeline.execute.reset_mock()
        result3 = cache.get_batch(["text"], "model")
        assert result3 == {}
        mock_pipeline.execute.assert_not_called()


class TestGlobalFunctions:
    """Test cases for global functions."""

    @patch.dict(os.environ, {"USE_REDIS_CACHE": "false"})
    def test_get_embedding_cache_disabled(self):
        """Test get_embedding_cache when caching is disabled."""
        result = get_embedding_cache()
        assert result is None

    @patch.dict(os.environ, {"USE_REDIS_CACHE": "true"})
    @patch("embedding_cache.EmbeddingCache")
    def test_get_embedding_cache_enabled_success(self, mock_cache_class):
        """Test get_embedding_cache when caching is enabled and succeeds."""
        mock_cache_instance = Mock()
        mock_cache_class.return_value = mock_cache_instance

        # Clear global cache
        import embedding_cache

        embedding_cache._embedding_cache = None

        result = get_embedding_cache()

        assert result == mock_cache_instance
        mock_cache_class.assert_called_once()

    @patch.dict(os.environ, {"USE_REDIS_CACHE": "true"})
    @patch("embedding_cache.EmbeddingCache")
    def test_get_embedding_cache_initialization_failure(self, mock_cache_class):
        """Test get_embedding_cache when cache initialization fails."""
        mock_cache_class.side_effect = Exception("Redis unavailable")

        # Clear global cache
        import embedding_cache

        embedding_cache._embedding_cache = None

        result = get_embedding_cache()

        assert result is None
        mock_cache_class.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "REDIS_HOST": "localhost",
            "REDIS_PORT": "6379",
            "REDIS_EMBEDDING_TTL": "86400",
        },
    )
    def test_validate_redis_config_success(self):
        """Test successful Redis configuration validation."""
        # Should not raise exception
        validate_redis_config()

    @patch.dict(os.environ, {"REDIS_PORT": "invalid"})
    def test_validate_redis_config_invalid_port(self):
        """Test Redis configuration validation with invalid port."""
        with pytest.raises(ValueError, match="Invalid Redis port"):
            validate_redis_config()

    @patch.dict(os.environ, {"REDIS_PORT": "70000"})
    def test_validate_redis_config_port_out_of_range(self):
        """Test Redis configuration validation with port out of range."""
        with pytest.raises(ValueError, match="Invalid Redis port"):
            validate_redis_config()

    @patch.dict(os.environ, {"REDIS_EMBEDDING_TTL": "-1"})
    def test_validate_redis_config_invalid_ttl(self):
        """Test Redis configuration validation with invalid TTL."""
        with pytest.raises(ValueError, match="Invalid Redis TTL"):
            validate_redis_config()

    @patch.dict(os.environ, {"REDIS_EMBEDDING_TTL": "not_a_number"})
    def test_validate_redis_config_non_numeric_ttl(self):
        """Test Redis configuration validation with non-numeric TTL."""
        with pytest.raises(ValueError, match="Invalid Redis TTL"):
            validate_redis_config()


if __name__ == "__main__":
    pytest.main([__file__])
