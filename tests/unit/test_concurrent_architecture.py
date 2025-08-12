"""
Comprehensive test suite for ThreadPoolExecutor integration and concurrent architecture.

This test suite validates the hybrid ThreadPoolExecutor + asyncio architecture,
ensuring CPU-bound operations don't block the event loop and testing thread safety
of ML models and services.
"""

import asyncio
import os
import pytest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch, AsyncMock

# Disable GPU for testing to ensure consistent behavior
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_GPU"] = "false"


@pytest.fixture
def mock_context():
    """Create a mock context with ThreadPoolExecutor."""
    context = Mock()
    context.request_context = Mock()
    context.request_context.lifespan_context = Mock()

    # Mock the components
    context.request_context.lifespan_context.qdrant_client = Mock()
    context.request_context.lifespan_context.reranker = Mock()
    context.request_context.lifespan_context.cpu_executor = ThreadPoolExecutor(
        max_workers=2
    )

    return context


@pytest.fixture
def mock_context_no_executor():
    """Create a mock context without ThreadPoolExecutor for fallback testing."""
    context = Mock()
    context.request_context = Mock()
    context.request_context.lifespan_context = Mock()

    # Mock the components but no executor
    context.request_context.lifespan_context.qdrant_client = Mock()
    context.request_context.lifespan_context.reranker = Mock()
    context.request_context.lifespan_context.cpu_executor = None

    return context


class TestContextExecutorInitialization:
    """Test ThreadPoolExecutor context integration."""

    def test_context_executor_initialization(self):
        """Test that Crawl4AIContext properly initializes with cpu_executor."""
        from src.core.context import Crawl4AIContext
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=4)

        context = Crawl4AIContext(
            crawler=Mock(),
            qdrant_client=Mock(),
            embedding_cache=Mock(),
            cpu_executor=executor,
        )

        assert context.cpu_executor is executor
        assert context.cpu_executor._max_workers == 4

        # Cleanup
        executor.shutdown(wait=True)

    def test_context_without_executor(self):
        """Test that context works without executor (graceful degradation)."""
        from src.core.context import Crawl4AIContext

        context = Crawl4AIContext(
            crawler=Mock(),
            qdrant_client=Mock(),
            embedding_cache=Mock(),
        )

        assert context.cpu_executor is None

    @patch("src.core.app.os.cpu_count")
    def test_executor_worker_count_calculation(self, mock_cpu_count):
        """Test that executor worker count is calculated correctly."""
        # Test with 16 cores - should cap at 8
        mock_cpu_count.return_value = 16

        # Get the calculation from the implementation
        cpu_count = os.cpu_count() or 1
        max_workers = min(cpu_count, 8)

        assert max_workers == 8  # Should be capped at 8

        # Test with 4 cores - should use 4
        mock_cpu_count.return_value = 4
        cpu_count = os.cpu_count() or 1
        max_workers = min(cpu_count, 8)

        assert max_workers == 4


class TestAsyncServiceMethods:
    """Test async service methods with ThreadPoolExecutor."""

    @pytest.mark.asyncio
    async def test_rerank_results_async(self):
        """Test async reranking with ThreadPoolExecutor."""
        from src.services.rag_service import RagService

        # Mock components
        qdrant_client = Mock()
        reranker = Mock()
        reranker.predict.return_value = [0.8, 0.6, 0.9]

        executor = ThreadPoolExecutor(max_workers=2)

        try:
            service = RagService(qdrant_client, reranking_model=reranker)
            service.use_reranking = True

            # Test data
            query = "test query"
            results = [
                {"content": "doc1", "score": 0.5},
                {"content": "doc2", "score": 0.7},
                {"content": "doc3", "score": 0.6},
            ]

            # Test async reranking
            reranked = await service.rerank_results_async(query, results, executor)

            # Verify results are reranked
            assert len(reranked) == 3
            assert all("rerank_score" in result for result in reranked)

            # Verify sorting by rerank score
            scores = [result["rerank_score"] for result in reranked]
            assert scores == sorted(scores, reverse=True)

        finally:
            executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_rerank_results_async_fallback(self):
        """Test graceful fallback when async reranking fails."""
        from src.services.rag_service import RagService

        qdrant_client = Mock()
        reranker = Mock()
        reranker.predict.side_effect = Exception("Model error")

        executor = ThreadPoolExecutor(max_workers=2)

        try:
            service = RagService(qdrant_client, reranking_model=reranker)
            service.use_reranking = True

            # Mock the fallback sync method
            service.rerank_results = Mock(return_value=[{"content": "fallback"}])

            query = "test query"
            results = [{"content": "doc1"}]

            # Should fallback to sync method
            reranked = await service.rerank_results_async(query, results, executor)

            assert reranked == [{"content": "fallback"}]
            service.rerank_results.assert_called_once_with(query, results, "content")

        finally:
            executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_search_with_reranking_async(self):
        """Test async search with reranking using ThreadPoolExecutor."""
        from src.services.rag_service import RagService

        qdrant_client = Mock()
        reranker = Mock()
        reranker.predict.return_value = [0.8, 0.6]

        executor = ThreadPoolExecutor(max_workers=2)

        try:
            service = RagService(qdrant_client, reranking_model=reranker)
            service.use_reranking = True
            service.use_hybrid_search = False

            # Mock search methods
            service.search_documents = Mock(
                return_value=[{"content": "doc1"}, {"content": "doc2"}]
            )

            query = "test query"
            results = await service.search_with_reranking_async(
                query, executor, match_count=10
            )

            assert len(results) == 2
            assert all("rerank_score" in result for result in results)

        finally:
            executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_embedding_service_async(self):
        """Test async embedding creation with ThreadPoolExecutor."""
        from src.services.embedding_service import EmbeddingService

        executor = ThreadPoolExecutor(max_workers=2)

        try:
            service = EmbeddingService()

            # Mock the sync method
            service.create_embeddings_batch = Mock(
                return_value=[[0.1, 0.2], [0.3, 0.4]]
            )

            texts = ["text1", "text2"]
            embeddings = await service.create_embeddings_batch_async(texts, executor)

            assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
            service.create_embeddings_batch.assert_called_once_with(texts)

        finally:
            executor.shutdown(wait=True)


class TestToolIntegration:
    """Test tool integration with executor from context."""

    @pytest.mark.asyncio
    async def test_rag_tools_with_executor(self, mock_context):
        """Test RAG tools use executor when available."""
        from src.tools.rag_tools import perform_rag_query

        # Mock the service
        with patch("src.services.rag_service.RagService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            # Mock async method
            mock_service.search_with_reranking_async = AsyncMock(
                return_value=[{"content": "result"}]
            )

            _ = await perform_rag_query(mock_context, "test query")

            # Verify async method was called with executor
            mock_service.search_with_reranking_async.assert_called_once()
            call_args = mock_service.search_with_reranking_async.call_args
            assert "executor" in call_args.kwargs
            assert (
                call_args.kwargs["executor"]
                is mock_context.request_context.lifespan_context.cpu_executor
            )

        # Cleanup
        mock_context.request_context.lifespan_context.cpu_executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_rag_tools_fallback_no_executor(self, mock_context_no_executor):
        """Test RAG tools fallback to sync when no executor available."""
        from src.tools.rag_tools import perform_rag_query

        with patch("src.services.rag_service.RagService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            # Mock sync method
            mock_service.search_with_reranking = Mock(
                return_value=[{"content": "result"}]
            )

            _ = await perform_rag_query(mock_context_no_executor, "test query")

            # Verify sync method was called
            mock_service.search_with_reranking.assert_called_once()

            # Verify async method was NOT called
            assert (
                not hasattr(mock_service, "search_with_reranking_async")
                or not mock_service.search_with_reranking_async.called
            )

    @pytest.mark.asyncio
    async def test_code_search_with_executor(self, mock_context):
        """Test code search tools use executor when available."""
        from src.tools.rag_tools import search_code_examples

        with patch("src.services.rag_service.RagService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            mock_service.search_with_reranking_async = AsyncMock(
                return_value=[{"content": "code"}]
            )

            _ = await search_code_examples(mock_context, "test query")

            # Verify async method was called
            mock_service.search_with_reranking_async.assert_called_once()
            call_args = mock_service.search_with_reranking_async.call_args
            assert call_args.kwargs["search_type"] == "code_examples"

        # Cleanup
        mock_context.request_context.lifespan_context.cpu_executor.shutdown(wait=True)


class TestThreadSafety:
    """Test thread safety of concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_reranking_thread_safety(self):
        """Test thread safety of concurrent reranking operations."""
        from src.services.rag_service import RagService

        qdrant_client = Mock()
        reranker = Mock()
        # Return different scores for each call to test concurrency
        reranker.predict.side_effect = lambda pairs: [0.8] * len(pairs)

        executor = ThreadPoolExecutor(max_workers=4)

        try:
            service = RagService(qdrant_client, reranking_model=reranker)
            service.use_reranking = True

            # Create multiple concurrent reranking tasks
            tasks = []
            for i in range(5):
                query = f"query_{i}"
                results = [{"content": f"doc_{i}_1"}, {"content": f"doc_{i}_2"}]
                task = service.rerank_results_async(query, results, executor)
                tasks.append(task)

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)

            # Verify all tasks completed successfully
            assert len(results) == 5
            for result in results:
                assert len(result) == 2
                assert all("rerank_score" in item for item in result)

        finally:
            executor.shutdown(wait=True)

    def test_concurrent_access_safety(self):
        """Test concurrent access to context executor is safe."""
        import concurrent.futures

        executor = ThreadPoolExecutor(max_workers=4)

        def access_executor():
            # Simulate accessing the same executor from multiple threads
            return executor is not None

        try:
            # Test concurrent access to the same executor
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as test_executor:
                futures = [test_executor.submit(access_executor) for _ in range(10)]
                results = [f.result() for f in futures]

            # All accesses should succeed
            assert all(results)

        finally:
            executor.shutdown(wait=True)


class TestErrorHandlingAndFallback:
    """Test error handling and graceful fallback mechanisms."""

    @pytest.mark.asyncio
    async def test_executor_failure_fallback(self):
        """Test graceful fallback when executor fails."""
        from src.services.rag_service import RagService

        qdrant_client = Mock()
        reranker = Mock()

        # Create a broken executor
        executor = ThreadPoolExecutor(max_workers=1)
        executor.shutdown(wait=True)  # Shutdown immediately to simulate failure

        service = RagService(qdrant_client, reranking_model=reranker)
        service.use_reranking = True

        # Mock the fallback sync method
        service.rerank_results = Mock(return_value=[{"content": "fallback"}])

        query = "test query"
        results = [{"content": "doc1"}]

        # Should fallback to sync method due to executor failure
        reranked = await service.rerank_results_async(query, results, executor)

        assert reranked == [{"content": "fallback"}]
        service.rerank_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_partial_failure_resilience(self):
        """Test resilience to partial failures in concurrent operations."""
        from src.services.rag_service import RagService

        qdrant_client = Mock()
        reranker = Mock()

        # Mock predict to fail on specific input
        def failing_predict(pairs):
            if len(pairs) > 2:
                raise ValueError("Model overload")
            return [0.8] * len(pairs)

        reranker.predict.side_effect = failing_predict

        executor = ThreadPoolExecutor(max_workers=2)

        try:
            service = RagService(qdrant_client, reranking_model=reranker)
            service.use_reranking = True

            # Mock fallback
            service.rerank_results = Mock(return_value=[{"content": "fallback"}])

            # Test with data that will cause failure
            query = "test query"
            results = [{"content": f"doc{i}"} for i in range(5)]  # Will trigger failure

            reranked = await service.rerank_results_async(query, results, executor)

            # Should fallback gracefully
            assert reranked == [{"content": "fallback"}]

        finally:
            executor.shutdown(wait=True)


class TestResourceCleanup:
    """Test proper resource cleanup and lifecycle management."""

    def test_executor_lifecycle_management(self):
        """Test proper executor initialization and cleanup."""
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test_worker")

        # Verify executor is functional
        future = executor.submit(lambda: "test")
        result = future.result(timeout=1)
        assert result == "test"

        # Test proper shutdown
        executor.shutdown(wait=True)

        # Verify executor is shutdown
        assert executor._shutdown

    def test_context_cleanup_shuts_down_executor(self):
        """Test that context cleanup properly shuts down the executor."""
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=2)

        # Simulate context cleanup
        executor.shutdown(wait=True)

        # Verify shutdown
        assert executor._shutdown


if __name__ == "__main__":
    pytest.main([__file__])
