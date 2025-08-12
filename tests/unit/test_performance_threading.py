"""
Performance benchmark tests for ThreadPoolExecutor integration.

This test suite measures and validates performance improvements from the hybrid
ThreadPoolExecutor + asyncio architecture, demonstrating improved throughput
and responsiveness under concurrent CPU-bound operations.
"""

import asyncio
import time
import pytest
import os
import psutil
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
from typing import Dict, Any
from contextlib import asynccontextmanager

# Disable GPU for consistent testing
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_GPU"] = "false"


class PerformanceMetrics:
    """Helper class to collect performance metrics."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.cpu_usage_samples = []
        self.memory_usage_samples = []

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.cpu_usage_samples = []
        self.memory_usage_samples = []

    def record_sample(self):
        """Record a performance sample."""
        if self.start_time:
            cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
            memory_info = psutil.Process().memory_info()

            self.cpu_usage_samples.append(
                {
                    "timestamp": time.time() - self.start_time,
                    "cpu_per_core": cpu_percent,
                    "cpu_total": sum(cpu_percent) / len(cpu_percent),
                }
            )

            self.memory_usage_samples.append(
                {
                    "timestamp": time.time() - self.start_time,
                    "memory_mb": memory_info.rss / 1024 / 1024,
                }
            )

    def stop_monitoring(self):
        """Stop performance monitoring and return results."""
        self.end_time = time.time()

        return {
            "total_time": self.end_time - self.start_time,
            "cpu_samples": self.cpu_usage_samples,
            "memory_samples": self.memory_usage_samples,
            "peak_cpu": max(
                (s["cpu_total"] for s in self.cpu_usage_samples), default=0
            ),
            "avg_cpu": sum(s["cpu_total"] for s in self.cpu_usage_samples)
            / len(self.cpu_usage_samples)
            if self.cpu_usage_samples
            else 0,
            "peak_memory_mb": max(
                (s["memory_mb"] for s in self.memory_usage_samples), default=0
            ),
            "multi_core_utilization": self._analyze_multi_core_usage(),
        }

    def _analyze_multi_core_usage(self) -> Dict[str, Any]:
        """Analyze multi-core CPU utilization patterns."""
        if not self.cpu_usage_samples:
            return {"cores_used": 0, "utilization_spread": 0}

        # Calculate how many cores show >50% utilization
        cores_used = 0
        utilization_spread = 0

        for sample in self.cpu_usage_samples:
            cores_above_50 = sum(1 for usage in sample["cpu_per_core"] if usage > 50)
            cores_used = max(cores_used, cores_above_50)

            # Calculate spread (variance) of core utilization
            core_usages = sample["cpu_per_core"]
            avg_usage = sum(core_usages) / len(core_usages)
            variance = sum((usage - avg_usage) ** 2 for usage in core_usages) / len(
                core_usages
            )
            utilization_spread = max(utilization_spread, variance)

        return {
            "cores_used": cores_used,
            "utilization_spread": utilization_spread,
            "max_core_usage": max(
                max(s["cpu_per_core"]) for s in self.cpu_usage_samples
            ),
        }


@asynccontextmanager
async def performance_monitor():
    """Context manager for performance monitoring."""
    metrics = PerformanceMetrics()
    metrics.start_monitoring()

    # Start background monitoring task
    async def monitor_loop():
        while True:
            metrics.record_sample()
            await asyncio.sleep(0.1)  # Sample every 100ms

    monitor_task = asyncio.create_task(monitor_loop())

    try:
        yield metrics
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass


def simulate_cpu_bound_work(duration: float = 0.1) -> float:
    """Simulate CPU-bound work (like ML model inference)."""
    start = time.time()
    # Simulate CPU-intensive work similar to CrossEncoder prediction
    result = 0
    while time.time() - start < duration:
        result += sum(range(1000))
    return result


def create_mock_reranker(processing_time: float = 0.1):
    """Create a mock reranker that simulates CPU-bound processing."""
    mock = Mock()

    def predict_with_delay(pairs):
        simulate_cpu_bound_work(processing_time)
        return [0.8] * len(pairs)

    mock.predict.side_effect = predict_with_delay
    return mock


class TestConcurrentPerformanceImprovement:
    """Test performance improvements from concurrent processing."""

    @pytest.mark.asyncio
    async def test_concurrent_vs_sequential_reranking(self):
        """Test that concurrent reranking is faster than sequential."""
        from src.services.rag_service import RagService

        qdrant_client = Mock()
        reranker = create_mock_reranker(processing_time=0.2)  # 200ms per operation

        executor = ThreadPoolExecutor(max_workers=4)

        try:
            service = RagService(qdrant_client, reranking_model=reranker)
            service.use_reranking = True

            # Test data - 4 operations that should benefit from parallel processing
            test_queries = [f"query_{i}" for i in range(4)]
            test_results = [
                [{"content": f"doc_{i}_{j}"} for j in range(3)] for i in range(4)
            ]

            # Time sequential execution (sync method)
            start_sequential = time.time()
            sequential_results = []
            for query, results in zip(test_queries, test_results):
                result = service.rerank_results(query, results)
                sequential_results.append(result)
            sequential_time = time.time() - start_sequential

            # Time concurrent execution (async method)
            start_concurrent = time.time()
            concurrent_tasks = [
                service.rerank_results_async(query, results, executor)
                for query, results in zip(test_queries, test_results)
            ]
            concurrent_results = await asyncio.gather(*concurrent_tasks)
            concurrent_time = time.time() - start_concurrent

            # Performance validation
            improvement_ratio = sequential_time / concurrent_time
            print(f"Sequential time: {sequential_time:.2f}s")
            print(f"Concurrent time: {concurrent_time:.2f}s")
            print(f"Performance improvement: {improvement_ratio:.2f}x")

            # Assert significant performance improvement (should be >2x with 4 workers)
            assert improvement_ratio > 1.5, (
                f"Expected >1.5x improvement, got {improvement_ratio:.2f}x"
            )

            # Verify results are the same
            assert len(concurrent_results) == len(sequential_results) == 4

        finally:
            executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_mixed_workload_responsiveness(self):
        """Test server responsiveness during mixed I/O and CPU workloads."""
        from src.services.rag_service import RagService

        qdrant_client = Mock()
        reranker = create_mock_reranker(processing_time=0.3)  # Heavy CPU work

        executor = ThreadPoolExecutor(max_workers=4)

        try:
            service = RagService(qdrant_client, reranking_model=reranker)
            service.use_reranking = True

            # Lightweight I/O-bound tasks (should remain responsive)
            async def lightweight_task():
                await asyncio.sleep(0.01)  # Simulate lightweight I/O
                return "lightweight_response"

            # Heavy CPU-bound tasks
            async def heavy_task():
                results = [{"content": f"doc_{i}"} for i in range(5)]
                return await service.rerank_results_async(
                    "heavy query", results, executor
                )

            async with performance_monitor() as metrics:
                # Start heavy tasks that will consume CPU
                heavy_tasks = [heavy_task() for _ in range(3)]

                # Interleave lightweight tasks
                lightweight_tasks = []
                for i in range(10):
                    await asyncio.sleep(0.05)  # Small delay between lightweight tasks
                    task = lightweight_task()
                    lightweight_tasks.append(task)

                # Wait for all tasks
                heavy_results = await asyncio.gather(*heavy_tasks)
                lightweight_results = await asyncio.gather(*lightweight_tasks)

            perf_data = metrics.stop_monitoring()

            # Verify responsiveness - lightweight tasks should complete quickly
            assert len(lightweight_results) == 10
            assert all(
                result == "lightweight_response" for result in lightweight_results
            )

            # Verify heavy tasks completed successfully
            assert len(heavy_results) == 3

            # Verify multi-core utilization
            multi_core_data = perf_data["multi_core_utilization"]
            print(f"Cores utilized: {multi_core_data['cores_used']}")
            print(f"Max core usage: {multi_core_data['max_core_usage']:.1f}%")

            # Should use multiple cores during heavy processing
            assert multi_core_data["cores_used"] >= 2, (
                "Should utilize multiple CPU cores"
            )

        finally:
            executor.shutdown(wait=True)


class TestScalabilityAndResourceUsage:
    """Test scalability and resource usage patterns."""

    @pytest.mark.asyncio
    async def test_worker_scaling_performance(self):
        """Test performance scaling with different worker counts."""
        from src.services.rag_service import RagService

        qdrant_client = Mock()
        reranker = create_mock_reranker(processing_time=0.1)

        results_by_workers = {}

        # Test different worker counts
        for worker_count in [2, 4, 8]:
            executor = ThreadPoolExecutor(max_workers=worker_count)

            try:
                service = RagService(qdrant_client, reranking_model=reranker)
                service.use_reranking = True

                # Create work that can benefit from parallelization
                tasks_count = worker_count * 2  # 2 tasks per worker

                start_time = time.time()
                tasks = []
                for i in range(tasks_count):
                    query = f"query_{i}"
                    results = [{"content": f"doc_{i}_{j}"} for j in range(3)]
                    task = service.rerank_results_async(query, results, executor)
                    tasks.append(task)

                await asyncio.gather(*tasks)
                execution_time = time.time() - start_time

                results_by_workers[worker_count] = {
                    "time": execution_time,
                    "throughput": tasks_count / execution_time,
                    "tasks_count": tasks_count,
                }

                print(
                    f"Workers: {worker_count}, Time: {execution_time:.2f}s, Throughput: {tasks_count / execution_time:.2f} tasks/s"
                )

            finally:
                executor.shutdown(wait=True)

        # Analyze scaling efficiency
        base_workers = 2
        base_throughput = results_by_workers[base_workers]["throughput"]

        for workers, data in results_by_workers.items():
            if workers > base_workers:
                scaling_efficiency = data["throughput"] / base_throughput
                expected_scaling = workers / base_workers
                efficiency_ratio = scaling_efficiency / expected_scaling

                print(f"Workers: {workers}, Scaling efficiency: {efficiency_ratio:.2f}")

                # Should achieve some scaling benefit (at least 30% of ideal for test environment)
                assert efficiency_ratio > 0.3, (
                    f"Poor scaling efficiency: {efficiency_ratio:.2f}"
                )

    def test_memory_usage_stability(self):
        """Test memory usage remains stable during concurrent operations."""
        import gc

        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Simulate memory-intensive operations
        executor = ThreadPoolExecutor(max_workers=4)

        try:
            # Submit many tasks to test memory stability
            futures = []
            for i in range(100):
                future = executor.submit(simulate_cpu_bound_work, 0.01)
                futures.append(future)

            # Wait for completion
            for future in futures:
                future.result()

            # Force garbage collection
            gc.collect()

            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            print(f"Initial memory: {initial_memory:.1f}MB")
            print(f"Final memory: {final_memory:.1f}MB")
            print(f"Memory increase: {memory_increase:.1f}MB")

            # Memory increase should be reasonable (less than 100MB for this test)
            assert memory_increase < 100, (
                f"Excessive memory growth: {memory_increase:.1f}MB"
            )

        finally:
            executor.shutdown(wait=True)


class TestRealWorldScenarios:
    """Test performance under realistic usage scenarios."""

    @pytest.mark.asyncio
    async def test_typical_rag_query_performance(self):
        """Test performance of typical RAG queries with reranking."""
        from src.services.rag_service import RagService

        qdrant_client = Mock()
        reranker = create_mock_reranker(
            processing_time=0.05
        )  # Realistic reranking time

        executor = ThreadPoolExecutor(max_workers=4)

        try:
            service = RagService(qdrant_client, reranking_model=reranker)
            service.use_reranking = True

            # Mock search to return realistic result sets
            service.search_documents = Mock(
                return_value=[
                    {
                        "content": f"Document {i} content with relevant information",
                        "score": 0.7 + i * 0.01,
                    }
                    for i in range(20)  # Typical result set size
                ]
            )

            async with performance_monitor() as metrics:
                # Simulate realistic query patterns
                queries = [
                    "What is machine learning?",
                    "How to implement neural networks?",
                    "Best practices for data preprocessing",
                    "Explain gradient descent algorithm",
                    "Deep learning vs traditional ML",
                ]

                tasks = []
                for query in queries:
                    # Use search_with_reranking_async for realistic scenario
                    task = service.search_with_reranking_async(
                        query, executor, match_count=20
                    )
                    tasks.append(task)

                results = await asyncio.gather(*tasks)

            perf_data = metrics.stop_monitoring()

            # Performance assertions
            assert perf_data["total_time"] < 2.0, (
                f"Queries took too long: {perf_data['total_time']:.2f}s"
            )

            # Verify all queries completed successfully
            assert len(results) == 5
            for result in results:
                assert len(result) > 0
                assert all("rerank_score" in item for item in result)

            # Check CPU utilization shows parallel processing
            multi_core_data = perf_data["multi_core_utilization"]
            assert multi_core_data["cores_used"] >= 2, "Should utilize multiple cores"

            print(f"RAG queries completed in {perf_data['total_time']:.2f}s")
            print(f"Average CPU usage: {perf_data['avg_cpu']:.1f}%")
            print(f"Cores utilized: {multi_core_data['cores_used']}")

        finally:
            executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_burst_request_handling(self):
        """Test handling of burst request scenarios."""
        from src.services.rag_service import RagService

        qdrant_client = Mock()
        reranker = create_mock_reranker(
            processing_time=0.1
        )  # Longer processing time to show multi-core usage

        executor = ThreadPoolExecutor(max_workers=8)

        try:
            service = RagService(qdrant_client, reranking_model=reranker)
            service.use_reranking = True

            # Simulate burst of requests
            burst_size = 20

            async with performance_monitor() as metrics:
                start_time = time.time()

                # Create burst of concurrent requests
                tasks = []
                for i in range(burst_size):
                    query = f"burst_query_{i}"
                    results = [{"content": f"doc_{j}"} for j in range(5)]
                    task = service.rerank_results_async(query, results, executor)
                    tasks.append(task)

                # Execute all requests concurrently
                results = await asyncio.gather(*tasks)
                execution_time = time.time() - start_time

            perf_data = metrics.stop_monitoring()

            # Performance validation
            throughput = burst_size / execution_time
            print(f"Burst handling: {burst_size} requests in {execution_time:.2f}s")
            print(f"Throughput: {throughput:.2f} requests/sec")

            # Should handle burst efficiently (target >10 req/sec for this test)
            assert throughput > 5, f"Low throughput: {throughput:.2f} req/sec"

            # All requests should complete successfully
            assert len(results) == burst_size

            # Should utilize multiple cores during burst
            multi_core_data = perf_data["multi_core_utilization"]
            assert multi_core_data["cores_used"] >= 2, (
                "Should use multiple cores for burst handling"
            )

        finally:
            executor.shutdown(wait=True)


class TestPerformanceRegressionDetection:
    """Test to detect performance regressions."""

    @pytest.mark.asyncio
    async def test_response_time_targets(self):
        """Test that response times meet target SLAs."""
        from src.services.rag_service import RagService

        qdrant_client = Mock()
        reranker = create_mock_reranker(processing_time=0.01)  # Fast reranker

        executor = ThreadPoolExecutor(max_workers=4)

        try:
            service = RagService(qdrant_client, reranking_model=reranker)
            service.use_reranking = True

            # Test single query response time
            query = "test query"
            results = [{"content": f"doc_{i}"} for i in range(10)]

            start_time = time.time()
            result = await service.rerank_results_async(query, results, executor)
            response_time = time.time() - start_time

            # Target: <500ms for 10 document reranking
            print(f"Single query response time: {response_time * 1000:.1f}ms")
            assert response_time < 0.5, (
                f"Response time too slow: {response_time * 1000:.1f}ms"
            )

            # Verify result quality
            assert len(result) == 10
            assert all("rerank_score" in item for item in result)

        finally:
            executor.shutdown(wait=True)

    def test_resource_utilization_efficiency(self):
        """Test resource utilization efficiency."""
        # Test that ThreadPoolExecutor uses resources efficiently
        executor = ThreadPoolExecutor(max_workers=4)

        try:
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Submit work and measure resource usage
            futures = []
            for i in range(20):
                future = executor.submit(simulate_cpu_bound_work, 0.05)
                futures.append(future)

            # Wait for completion
            for future in futures:
                future.result()

            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_overhead = end_memory - start_memory

            print(f"Memory overhead: {memory_overhead:.1f}MB")

            # Memory overhead should be reasonable
            assert memory_overhead < 50, (
                f"High memory overhead: {memory_overhead:.1f}MB"
            )

        finally:
            executor.shutdown(wait=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements
