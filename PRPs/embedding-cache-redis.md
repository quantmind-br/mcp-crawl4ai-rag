# PRP: Embedding Cache with Redis Integration

## Feature Overview

**Embedding Cache with Redis Integration** introduces a high-performance caching layer for vector embeddings to significantly reduce API costs, decrease latency during data ingestion, and improve system resilience. The feature intercepts requests for embeddings, checks a Redis cache for existing results, and only calls external embedding APIs (OpenAI/DeepInfra) for texts that haven't been processed before. Newly generated embeddings are stored in the cache for future use.

### Core Value Proposition
- **Cost Reduction**: 60-85% reduction in embedding API calls based on real-world implementations
- **Performance Improvement**: 6.86x faster embedding retrieval (Redis official benchmarks)
- **Resilience**: Graceful degradation when Redis unavailable, never breaks main application flow
- **Memory Efficiency**: Optimized serialization and TTL strategies for production use

## Deep Technical Context

### Current Implementation Analysis

The current `create_embeddings_batch()` function in `src/utils.py:368-424` implements:
- **Retry Logic**: 3 attempts with exponential backoff (1s → 2s → 4s)
- **Batch → Individual Fallback**: Falls back to individual embeddings on batch failure
- **Zero Padding Fallback**: Returns zero vectors when all attempts fail
- **Client Management**: Uses `get_embeddings_client()` with sophisticated configuration

### Existing Patterns to Follow

**Configuration Pattern** (from utils.py):
```python
# Environment variables: SERVICE_SETTING_SUBSETTING
CHAT_API_KEY, CHAT_API_BASE, CHAT_MODEL
EMBEDDINGS_API_KEY, EMBEDDINGS_API_BASE, EMBEDDINGS_MODEL
QDRANT_HOST, QDRANT_PORT
```

**Client Wrapper Pattern** (from qdrant_wrapper.py):
```python
class QdrantClientWrapper:
    def __init__(self, host=None, port=None):
        # Configuration with env defaults
    def _create_client(self):
        # Connection with health check
    def health_check(self):
        # Functional validation
```

**Retry Pattern** (from qdrant_wrapper.py):
```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _robust_operation(self, operation_func, *args, **kwargs):
```

**Test Pattern** (from existing tests):
```python
@patch('utils.openai.embeddings.create')
def test_create_embeddings_batch_success(self, mock_openai_create):
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1] * 1536)]
    mock_openai_create.return_value = mock_response
```

### Production Implementation Examples

**Real-world Performance Data**:
- **chainainexus-cloud**: Base64 serialization, 600s TTL for queries, database for long-term
- **Prometheus-Gateway**: Two-tier (ChromaDB + Redis), 6.86x performance improvement
- **ProjectHadesTesting**: Bulk loading, metadata separation, resource management

**Serialization Performance** (from production systems):
| Format | Speed | Size | Best Use |
|--------|-------|------|----------|
| Pickle | Fast | Good | Python-specific (our choice) |
| MessagePack | Very Fast | Excellent | Cross-language |
| JSON | Slow | Poor | Development only |

## Architecture Design

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   create_embeddings_batch()                 │
│  ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │  Cache Check    │───▶│ Cache Miss   │───▶│   API Call  │ │
│  │  (Batch)        │    │ Processing   │    │  (Existing) │ │
│  └─────────────────┘    └──────────────┘    └─────────────┘ │
│           │                       │                  │      │
│           ▼                       ▼                  ▼      │
│  ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │  Cache Hit      │    │ Cache Store  │    │   Combine   │ │
│  │  Return         │    │  (Batch)     │    │   Results   │ │
│  └─────────────────┘    └──────────────┘    └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    EmbeddingCache                           │
│  ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │ Redis Client    │    │ Circuit      │    │ Health      │ │
│  │ (Pooled)        │    │ Breaker      │    │ Check       │ │
│  └─────────────────┘    └──────────────┘    └─────────────┘ │
│           │                       │                  │      │
│           ▼                       ▼                  ▼      │
│  ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │ Serialization   │    │ Error        │    │ Graceful    │ │
│  │ (Pickle)        │    │ Handling     │    │ Degradation │ │
│  └─────────────────┘    └──────────────┘    └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Integration Points

1. **Primary Integration**: `src/utils.py:create_embeddings_batch()` - Cache layer before API calls
2. **Configuration**: Environment variables following existing patterns
3. **Docker**: Redis service in docker-compose.yaml
4. **Dependencies**: Redis-py added to pyproject.toml
5. **Testing**: Mock Redis in unit tests, real Redis for integration

## Implementation Blueprint

### Task 1: Create Core Cache Module (`src/embedding_cache.py`)

**Pseudocode Approach**:
```python
class EmbeddingCache:
    def __init__(self):
        # Initialize Redis client with connection pooling
        # Setup circuit breaker for graceful degradation
        # Configure serialization strategy (pickle)
    
    def get_batch(self, texts: List[str], model: str) -> Dict[str, List[float]]:
        # Generate cache keys for all texts
        # Batch retrieve from Redis using pipeline
        # Deserialize found embeddings
        # Return dict of text -> embedding mappings
    
    def set_batch(self, embeddings: Dict[str, List[float]], model: str, ttl: int):
        # Serialize embeddings using pickle
        # Batch store to Redis using pipeline
        # Handle errors gracefully (don't break main flow)
    
    def health_check(self) -> bool:
        # Test Redis connection with PING
        # Return connection status
```

**Key Design Decisions**:
- **Cache Key Format**: `embedding:{model}:{sha256(text)[:16]}` (collision-resistant, space-efficient)
- **Serialization**: Pickle (Python-native, fast, follows codebase patterns)
- **TTL Strategy**: Configurable via `REDIS_EMBEDDING_TTL` (default: 86400s/24h)
- **Circuit Breaker**: 5 failures trigger open state, 60s recovery timeout
- **Connection Pool**: 20 max connections, 5s timeout, health check interval 30s

### Task 2: Integrate with Existing Embedding Function (`src/utils.py`)

**Modification Strategy**:
```python
# Import at top of file
from .embedding_cache import EmbeddingCache

# Global instance (follows existing patterns)
_embedding_cache = None

def get_embedding_cache():
    """Get global embedding cache instance (singleton pattern)."""
    global _embedding_cache
    if _embedding_cache is None and os.getenv("USE_REDIS_CACHE", "false") == "true":
        try:
            _embedding_cache = EmbeddingCache()
        except Exception as e:
            print(f"Failed to initialize embedding cache: {e}")
            _embedding_cache = None
    return _embedding_cache

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    
    embeddings_model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    final_embeddings = [None] * len(texts)
    
    # NEW: Try cache first
    cache = get_embedding_cache()
    if cache:
        cached_embeddings = cache.get_batch(texts, embeddings_model)
        
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            if text in cached_embeddings:
                final_embeddings[i] = cached_embeddings[text]  # Cache hit
            else:
                texts_to_embed.append(text)  # Cache miss
                indices_to_embed.append(i)
    else:
        # No cache available, embed all texts
        texts_to_embed = texts
        indices_to_embed = list(range(len(texts)))
    
    # EXISTING: API call logic (only for cache misses)
    if texts_to_embed:
        # [Existing retry logic unchanged]
        new_embeddings_list = [API call results]
        
        # NEW: Store in cache
        if cache and new_embeddings_list:
            new_to_cache = {text: emb for text, emb in zip(texts_to_embed, new_embeddings_list)}
            cache.set_batch(new_to_cache, embeddings_model, 
                          int(os.getenv("REDIS_EMBEDDING_TTL", "86400")))
        
        # Place new embeddings in correct positions
        for i, new_embedding in enumerate(new_embeddings_list):
            original_index = indices_to_embed[i]
            final_embeddings[original_index] = new_embedding
    
    return final_embeddings
```

### Task 3: Configuration and Environment Setup

**Environment Variables** (add to `.env.example`):
```bash
# ===============================
# REDIS CACHE CONFIGURATION
# ===============================

# Redis Connection
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_USERNAME=
REDIS_SSL=false

# Redis Performance Settings
REDIS_CONNECTION_TIMEOUT=5
REDIS_SOCKET_TIMEOUT=5
REDIS_MAX_CONNECTIONS=20
REDIS_HEALTH_CHECK_INTERVAL=30

# Cache Behavior
USE_REDIS_CACHE=false
REDIS_EMBEDDING_TTL=86400
REDIS_CIRCUIT_BREAKER_FAILURES=5
REDIS_CIRCUIT_BREAKER_TIMEOUT=60
```

**Docker Service** (add to `docker-compose.yaml`):
```yaml
redis:
  image: redis:7-alpine
  restart: unless-stopped
  container_name: mcp-redis
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data
  environment:
    - REDIS_PASSWORD=${REDIS_PASSWORD:-}
  command: >
    redis-server
    --appendonly yes
    --maxmemory 256mb
    --maxmemory-policy allkeys-lru
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
    interval: 30s
    timeout: 10s
    retries: 5
```

**Dependencies** (add to `pyproject.toml`):
```toml
[tool.uv.sources]
redis = ">=5.0.0,<6.0.0"
```

### Task 4: Comprehensive Testing Strategy

**Unit Tests** (`tests/test_embedding_cache.py`):
```python
@patch('embedding_cache.redis.Redis')
class TestEmbeddingCache(unittest.TestCase):
    def test_cache_hit_scenario(self, mock_redis_class):
        # Mock Redis client returning cached data
        # Test cache retrieval and deserialization
        # Verify no API calls made for cached items
    
    def test_cache_miss_scenario(self, mock_redis_class):
        # Mock Redis client returning None
        # Test fallback to API calls
        # Verify cache storage of new embeddings
    
    def test_redis_connection_failure(self, mock_redis_class):
        # Mock Redis connection failure
        # Test graceful degradation (no cache, direct API)
        # Verify circuit breaker activation
    
    def test_circuit_breaker_behavior(self, mock_redis_class):
        # Simulate repeated Redis failures
        # Test circuit breaker state transitions
        # Verify recovery behavior
```

**Integration Tests** (`tests/test_redis_integration.py`):
```python
@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
class TestRedisIntegration:
    def test_real_redis_operations(self):
        # Test with actual Redis container
        # Verify serialization/deserialization
        # Test TTL behavior
```

### Task 5: Error Handling and Graceful Degradation

**Circuit Breaker Implementation**:
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                return None  # Fail fast
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self._on_failure()
            return None  # Graceful degradation
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

### Task 6: Health Checks and Monitoring

**Health Check Implementation**:
```python
def health_check(self) -> Dict[str, Any]:
    """Comprehensive cache health check."""
    try:
        start_time = time.time()
        
        # Test basic connectivity
        ping_result = self.redis.ping()
        
        # Test read/write operations
        test_key = f"health_check:{int(time.time())}"
        self.redis.setex(test_key, 10, "test")
        read_result = self.redis.get(test_key)
        self.redis.delete(test_key)
        
        latency = (time.time() - start_time) * 1000  # ms
        
        return {
            'status': 'healthy',
            'ping': ping_result,
            'read_write': read_result == b'test',
            'latency_ms': round(latency, 2),
            'circuit_breaker_state': self.circuit_breaker.state.value
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'circuit_breaker_state': self.circuit_breaker.state.value
        }
```

## Risk Mitigation

### High-Risk Scenarios

1. **Redis Connection Failure**
   - **Mitigation**: Circuit breaker pattern, graceful degradation
   - **Fallback**: Continue without caching, log warnings
   - **Recovery**: Automatic retry with exponential backoff

2. **Memory Exhaustion in Redis**
   - **Mitigation**: LRU eviction policy, reasonable TTLs
   - **Monitoring**: Memory usage alerts
   - **Prevention**: TTL-based cleanup, connection limits

3. **Serialization Failures**
   - **Mitigation**: Try-catch around pickle operations
   - **Fallback**: Skip caching for problematic embeddings
   - **Logging**: Detailed error context for debugging

4. **Cache Key Collisions**
   - **Mitigation**: SHA256 hash with model prefix
   - **Detection**: Monitor for unexpected cache hits
   - **Prevention**: Include model name in key generation

### Performance Risks

1. **Latency Introduction**
   - **Target**: <10ms average for Redis operations
   - **Monitoring**: Track cache operation latencies
   - **Optimization**: Connection pooling, pipelining

2. **Serialization Overhead**
   - **Benchmark**: Pickle vs alternatives
   - **Optimization**: Consider compression for large embeddings
   - **Fallback**: Direct API if serialization too slow

## Validation Gates

### Executable Validation Commands

```bash
# Syntax and Style Validation
ruff check --fix src/embedding_cache.py src/utils.py
mypy src/embedding_cache.py src/utils.py

# Unit Tests (with Redis mocked)
uv run pytest tests/test_embedding_cache.py -v

# Integration Tests (requires Docker Redis)
docker-compose up -d redis
uv run pytest tests/test_redis_integration.py -v

# Performance Benchmarks
uv run pytest tests/test_embedding_performance.py -v --benchmark-only

# Health Check Validation
uv run python -c "
from src.embedding_cache import EmbeddingCache
cache = EmbeddingCache()
health = cache.health_check()
print(f'Cache health: {health}')
assert health['status'] == 'healthy', f'Cache unhealthy: {health}'
print('✅ Cache health check passed')
"

# MCP Tool Integration Test
uv run python -c "
import os
os.environ['USE_REDIS_CACHE'] = 'true'
from src.utils import create_embeddings_batch
result = create_embeddings_batch(['test embedding'])
assert len(result) == 1 and len(result[0]) > 0
print('✅ MCP integration test passed')
"

# Docker Services Health Check
docker-compose ps redis
docker-compose exec redis redis-cli ping

# Configuration Validation
uv run python -c "
from src.embedding_cache import validate_redis_config
validate_redis_config()
print('✅ Configuration validation passed')
"
```

### Success Criteria

- **Functionality**: All unit tests pass (≥95% coverage)
- **Integration**: Redis integration tests pass with real Redis
- **Performance**: Cache operations <10ms, API call reduction ≥60%
- **Reliability**: Circuit breaker activates/recovers properly
- **Compatibility**: No breaking changes to existing functionality
- **Documentation**: Configuration and usage clearly documented

## Performance Benchmarks

### Expected Performance Improvements

**Cache Hit Scenarios**:
- **Embedding Retrieval**: 6.86x faster (from Redis benchmarks)
- **API Cost Reduction**: 60-85% fewer embedding API calls
- **Latency Reduction**: ~0.039s per query (from production data)

**Cache Miss Scenarios**:
- **Overhead**: <5ms additional latency for cache check
- **Batch Benefits**: Store multiple embeddings with single pipeline operation

### Monitoring Metrics

```python
# Key performance indicators to track
metrics = {
    'cache_hit_rate': 'Target: >70%',
    'avg_cache_latency': 'Target: <10ms',
    'api_call_reduction': 'Target: >60%',
    'circuit_breaker_activations': 'Target: <1/day',
    'memory_usage': 'Target: <256MB'
}
```

## Documentation Requirements

### User Documentation

1. **Configuration Guide**: Redis setup, environment variables
2. **Performance Tuning**: TTL strategies, connection pool sizing
3. **Troubleshooting**: Common issues, health check procedures
4. **Monitoring**: Key metrics, alerting recommendations

### Developer Documentation

1. **Architecture Overview**: Component interactions, data flow
2. **API Reference**: EmbeddingCache class methods, parameters
3. **Testing Guide**: Running tests, adding new test cases
4. **Contributing**: Code style, review process

## External Documentation References

### Critical Documentation URLs

1. **Redis-py Documentation**: https://redis.readthedocs.io/en/stable/
2. **Connection Pools**: https://redis.readthedocs.io/en/stable/connections.html
3. **Redis Protocol**: https://redis.io/docs/latest/develop/reference/protocol-spec/
4. **Production Best Practices**: https://redis.io/docs/latest/operate/oss_and_stack/management/optimization/

### Implementation Examples

1. **Production RAG System**: https://github.com/Ubheee/chainainexus-cloud/blob/main/api/core/rag/embedding/cached_embedding.py
2. **Semantic Cache**: https://github.com/ozanunal0/Prometheus-Gateway/blob/main/app/vector_cache.py
3. **High-Performance Pipeline**: https://github.com/r3d91ll/ProjectHadesTesting/blob/main/rag-dataset-builder/run_redis_embedding.py

## Confidence Assessment

**Implementation Success Confidence: 9/10**

**Justification**:
- ✅ **Comprehensive Research**: Analyzed existing patterns, real-world implementations, performance data
- ✅ **Architectural Alignment**: Follows established codebase patterns (configuration, error handling, testing)
- ✅ **Risk Mitigation**: Circuit breaker, graceful degradation, comprehensive error handling
- ✅ **Validation Strategy**: Executable tests, health checks, performance benchmarks
- ✅ **Production Readiness**: Based on real-world implementations with proven performance gains

**Potential Challenges** (confidence reducers):
- Redis connection configuration complexity in different environments
- Serialization edge cases with specific embedding formats
- Performance tuning for optimal cache hit rates

The high confidence level is justified by the comprehensive research, alignment with existing patterns, and proven real-world implementations demonstrating similar performance benefits.