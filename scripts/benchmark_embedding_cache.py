#!/usr/bin/env python3
"""
Benchmark script for Redis embedding cache performance.

This script demonstrates the performance benefits of the Redis embedding cache
by measuring cache operations and simulating API call reductions.
"""

import sys
import time
import statistics
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from embedding_cache import EmbeddingCache


def benchmark_cache_operations():
    """Benchmark basic cache operations."""
    print("Redis Embedding Cache Performance Benchmark")
    print("=" * 50)

    # Initialize cache
    cache = EmbeddingCache()
    health = cache.health_check()

    if health["status"] != "healthy":
        print(f"Redis not available: {health}")
        return

    print(f"Redis connected: {health['latency_ms']:.2f}ms initial latency")
    print()

    # Test data
    test_texts = [f"Test embedding text {i}" for i in range(100)]
    model = "benchmark-model"
    embeddings = {text: [float(i)] * 1536 for i, text in enumerate(test_texts)}

    # Benchmark cache writes
    print("Benchmarking Cache Writes...")
    write_times = []
    batch_sizes = [1, 5, 10, 25, 50, 100]

    for batch_size in batch_sizes:
        batch_texts = test_texts[:batch_size]
        batch_embeddings = {text: embeddings[text] for text in batch_texts}

        start_time = time.time()
        cache.set_batch(batch_embeddings, model, ttl=60)
        end_time = time.time()

        duration_ms = (end_time - start_time) * 1000
        write_times.append(duration_ms)

        print(
            f"  Batch size {batch_size:3d}: {duration_ms:6.2f}ms ({duration_ms / batch_size:.2f}ms per item)"
        )

    print()

    # Benchmark cache reads
    print("Benchmarking Cache Reads...")
    read_times = []

    for batch_size in batch_sizes:
        batch_texts = test_texts[:batch_size]

        # Warm up cache
        cache.get_batch(batch_texts, model)

        # Measure read performance
        times = []
        for _ in range(10):  # Multiple runs for accuracy
            start_time = time.time()
            result = cache.get_batch(batch_texts, model)
            end_time = time.time()

            duration_ms = (end_time - start_time) * 1000
            times.append(duration_ms)

            # Verify all items were found
            assert len(result) == batch_size, (
                f"Expected {batch_size} items, got {len(result)}"
            )

        avg_time = statistics.mean(times)
        read_times.append(avg_time)

        print(
            f"  Batch size {batch_size:3d}: {avg_time:6.2f}ms ({avg_time / batch_size:.2f}ms per item)"
        )

    print()

    # Performance summary
    print("Performance Summary")
    print("-" * 30)
    print(f"• Average write latency: {statistics.mean(write_times):.2f}ms")
    print(f"• Average read latency:  {statistics.mean(read_times):.2f}ms")
    print(f"• Best read performance: {min(read_times):.2f}ms")
    print(f"• Single item read:      {read_times[0]:.2f}ms")
    print()

    # Simulate API cost savings
    print("Simulated Cost Savings")
    print("-" * 30)

    # Assume different cache hit rates
    hit_rates = [0.5, 0.7, 0.8, 0.9, 0.95]
    monthly_embeddings = 1000000  # 1M embeddings per month
    cost_per_1k = 0.02  # $0.02 per 1K embeddings (text-embedding-3-small)

    for hit_rate in hit_rates:
        api_calls_saved = monthly_embeddings * hit_rate
        cost_saved = (api_calls_saved / 1000) * cost_per_1k

        print(
            f"  {hit_rate * 100:2.0f}% hit rate: {api_calls_saved:8,.0f} API calls saved = ${cost_saved:5.2f}/month"
        )

    print()

    # Memory usage estimate
    embedding_size = 1536 * 4  # 1536 floats * 4 bytes each = 6KB per embedding
    cache_entries = [10000, 50000, 100000, 250000]

    print("Memory Usage Estimates")
    print("-" * 30)

    for entries in cache_entries:
        memory_mb = (entries * embedding_size) / (1024 * 1024)
        ttl_hours = 24

        print(f"  {entries:6,} embeddings: {memory_mb:5.1f}MB (TTL: {ttl_hours}h)")

    print()
    print("Recommendations")
    print("-" * 30)
    print("* Target 70-85% cache hit rate for optimal cost savings")
    print("* Monitor Redis memory usage and adjust TTL as needed")
    print("* Use batch operations when possible for better performance")
    print("* Circuit breaker provides automatic failover protection")


if __name__ == "__main__":
    try:
        benchmark_cache_operations()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        sys.exit(1)
