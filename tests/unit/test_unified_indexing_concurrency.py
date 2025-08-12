"""
Test concurrent operation of UnifiedIndexingService with ThreadPoolExecutor integration.

This test validates that the UnifiedIndexingService properly uses the context's
ThreadPoolExecutor and doesn't block the event loop during repository processing.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor

# Test imports
try:
    from src.services.unified_indexing_service import UnifiedIndexingService
    from src.models.unified_indexing_models import (
        UnifiedIndexingRequest,
        IndexingDestination,
    )
except ImportError:
    # Fallback for different import paths
    from services.unified_indexing_service import UnifiedIndexingService
    from models.unified_indexing_models import (
        UnifiedIndexingRequest,
        IndexingDestination,
    )


@pytest.fixture
def mock_context_executor():
    """Create a mock context with ThreadPoolExecutor."""
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="test_executor")


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    client = Mock()
    client.update_source_info = Mock()
    client.add_documents_to_qdrant = Mock(
        return_value=iter([])
    )  # Return empty iterator
    return client


@pytest.fixture
def mock_neo4j_parser():
    """Create a mock Neo4j parser."""
    parser = Mock()
    parser.initialize = AsyncMock()
    return parser


class TestUnifiedIndexingConcurrency:
    """Test concurrent operation of UnifiedIndexingService."""

    def test_service_uses_context_executor(
        self, mock_context_executor, mock_qdrant_client
    ):
        """Test that service uses provided context executor."""
        service = UnifiedIndexingService(
            qdrant_client=mock_qdrant_client, cpu_executor=mock_context_executor
        )

        # Verify service uses provided executor
        assert service.cpu_executor is mock_context_executor
        assert service._own_executor is None

        # Cleanup shouldn't shutdown the context executor
        asyncio.run(service.cleanup())
        assert not mock_context_executor._shutdown

        # Cleanup the test executor
        mock_context_executor.shutdown(wait=True)

    def test_service_creates_fallback_executor(self, mock_qdrant_client):
        """Test that service creates fallback executor when none provided."""
        service = UnifiedIndexingService(qdrant_client=mock_qdrant_client)

        # Verify service created its own executor
        assert service.cpu_executor is not None
        assert service._own_executor is not None
        assert service.cpu_executor is service._own_executor

        # Cleanup should shutdown the owned executor
        asyncio.run(service.cleanup())
        assert service._own_executor is None

    @pytest.mark.asyncio
    async def test_concurrent_event_loop_not_blocked(
        self, mock_context_executor, mock_qdrant_client, mock_neo4j_parser
    ):
        """Test that event loop remains responsive during processing."""

        # Mock GitHub processor to simulate work
        with patch(
            "src.services.unified_indexing_service.GitHubProcessor"
        ) as mock_github_proc_class:
            mock_github_proc = Mock()
            mock_github_proc.clone_repository_temp = Mock(
                return_value={"success": True, "temp_directory": "/tmp/test_repo"}
            )
            mock_github_proc_class.return_value = mock_github_proc

            # Create service with context executor
            service = UnifiedIndexingService(
                qdrant_client=mock_qdrant_client,
                neo4j_parser=mock_neo4j_parser,
                cpu_executor=mock_context_executor,
            )

            # Mock file discovery to return empty list (avoid complex file processing)
            with patch.object(service, "_discover_repository_files", return_value=[]):
                # Create request
                request = UnifiedIndexingRequest(
                    repo_url="https://github.com/test/test",
                    destination=IndexingDestination.QDRANT,
                    file_types=[".py"],
                    max_files=10,
                )

                # Task to track event loop responsiveness
                responsiveness_checks = []

                async def check_responsiveness():
                    """Task that tracks if event loop is responsive."""
                    for i in range(10):  # Check 10 times over 1 second
                        start_time = asyncio.get_event_loop().time()
                        await asyncio.sleep(0.1)
                        end_time = asyncio.get_event_loop().time()
                        elapsed = end_time - start_time
                        responsiveness_checks.append(elapsed)

                # Run processing and responsiveness check concurrently
                processing_task = service.process_repository_unified(request)
                responsiveness_task = check_responsiveness()

                # Execute both tasks concurrently
                response, _ = await asyncio.gather(processing_task, responsiveness_task)

                # Verify processing completed successfully
                assert response.success

                # Verify event loop remained responsive (sleep times should be close to 0.1s)
                assert len(responsiveness_checks) == 10
                for elapsed in responsiveness_checks:
                    # Allow some tolerance, but should be close to 0.1s (not blocked)
                    assert 0.08 <= elapsed <= 0.15, (
                        f"Event loop blocked, sleep took {elapsed:.3f}s"
                    )

                print(
                    f"Event loop responsiveness: avg={sum(responsiveness_checks) / len(responsiveness_checks):.3f}s"
                )

        # Cleanup
        await service.cleanup()
        mock_context_executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(
        self, mock_context_executor, mock_qdrant_client, mock_neo4j_parser
    ):
        """Test handling multiple concurrent indexing requests."""

        with patch(
            "src.services.unified_indexing_service.GitHubProcessor"
        ) as mock_github_proc_class:
            # Simulate some processing time in clone operation
            def slow_clone(*args, **kwargs):
                import time

                time.sleep(0.1)  # Simulate I/O work
                return {"success": True, "temp_directory": "/tmp/test_repo"}

            mock_github_proc = Mock()
            mock_github_proc.clone_repository_temp = Mock(side_effect=slow_clone)
            mock_github_proc_class.return_value = mock_github_proc

            # Create service
            service = UnifiedIndexingService(
                qdrant_client=mock_qdrant_client,
                neo4j_parser=mock_neo4j_parser,
                cpu_executor=mock_context_executor,
            )

            # Mock file discovery to return empty list
            with patch.object(service, "_discover_repository_files", return_value=[]):
                # Create multiple concurrent requests
                requests = [
                    UnifiedIndexingRequest(
                        repo_url=f"https://github.com/test/repo{i}",
                        destination=IndexingDestination.QDRANT,
                        file_types=[".py"],
                        max_files=5,
                    )
                    for i in range(3)
                ]

                # Start all requests concurrently
                start_time = asyncio.get_event_loop().time()

                tasks = [service.process_repository_unified(req) for req in requests]
                responses = await asyncio.gather(*tasks)

                end_time = asyncio.get_event_loop().time()
                total_time = end_time - start_time

                # Verify all requests completed successfully
                assert all(response.success for response in responses)
                assert len(responses) == 3

                # Concurrent execution should be faster than sequential
                # Sequential would take at least 3 * 0.1 = 0.3s
                # Concurrent should take around 0.1s (with some overhead)
                print(f"Concurrent processing time: {total_time:.3f}s")
                assert total_time < 0.25, (
                    f"Concurrent processing took too long: {total_time:.3f}s"
                )

        # Cleanup
        await service.cleanup()
        mock_context_executor.shutdown(wait=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
