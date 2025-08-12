"""
Tests for the unified indexing service.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from src.services.unified_indexing_service import (
    UnifiedIndexingService,
    ResourceManager,
    ProgressTracker,
    process_repository_unified,
)
from src.models.unified_indexing_models import (
    UnifiedIndexingRequest,
    UnifiedIndexingResponse,
    IndexingDestination,
    FileProcessingResult,
)


class TestResourceManager:
    """Test cases for the ResourceManager class."""

    @pytest.mark.asyncio
    async def test_resource_manager_context_manager(self):
        """Test ResourceManager as context manager."""
        async with ResourceManager() as manager:
            assert manager is not None
            # Should not raise any exceptions

    @pytest.mark.asyncio
    async def test_register_and_cleanup_temp_directory(self):
        """Test registering and cleaning up temporary directories."""
        manager = ResourceManager()

        # Create a temporary directory
        temp_dir = Path(tempfile.mkdtemp())

        # Register it with the manager
        manager.register_temp_directory(temp_dir)

        # Verify it exists before cleanup
        assert temp_dir.exists()

        # Clean up resources
        await manager.cleanup_all_resources()

        # Verify it was removed
        assert not temp_dir.exists()

    @pytest.mark.asyncio
    async def test_register_and_execute_cleanup_task(self):
        """Test registering and executing cleanup tasks."""
        manager = ResourceManager()

        # Create a mock cleanup function
        cleanup_called = False

        def mock_cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        # Register the cleanup function
        manager.register_cleanup_task(mock_cleanup)

        # Execute cleanup
        await manager.cleanup_all_resources()

        # Verify the cleanup function was called
        assert cleanup_called

    @pytest.mark.asyncio
    async def test_cleanup_with_exceptions(self):
        """Test cleanup handling of exceptions."""
        manager = ResourceManager()

        # Create a cleanup function that raises an exception
        def failing_cleanup():
            raise Exception("Cleanup failed")

        # Register the failing cleanup function
        manager.register_cleanup_task(failing_cleanup)

        # Should not raise an exception even if cleanup fails
        await manager.cleanup_all_resources()


class TestProgressTracker:
    """Test cases for the ProgressTracker class."""

    def test_progress_tracker_initialization(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker(total_files=100)
        assert tracker.total_files == 100
        assert tracker.processed_files == 0
        assert tracker.failed_files == 0

    def test_update_progress(self):
        """Test updating progress metrics."""
        tracker = ProgressTracker(total_files=100)

        # Update with default values
        tracker.update_progress()
        assert tracker.processed_files == 1
        assert tracker.failed_files == 0

        # Update with custom values
        tracker.update_progress(
            processed_increment=5, failed_increment=2, operation="Testing"
        )
        assert tracker.processed_files == 6
        assert tracker.failed_files == 2
        assert tracker.current_operation == "Testing"


class TestUnifiedIndexingService:
    """Test cases for the UnifiedIndexingService class."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        return Mock()

    @pytest.fixture
    def mock_neo4j_parser(self):
        """Create a mock Neo4j parser."""
        return Mock()

    @pytest.fixture
    def mock_cpu_executor(self):
        """Create a mock CPU executor."""
        return Mock()

    @pytest.fixture
    def unified_indexing_service(
        self, mock_qdrant_client, mock_neo4j_parser, mock_cpu_executor
    ):
        """Create a UnifiedIndexingService instance."""
        return UnifiedIndexingService(
            qdrant_client=mock_qdrant_client,
            neo4j_parser=mock_neo4j_parser,
            cpu_executor=mock_cpu_executor,
        )

    def test_init_with_provided_dependencies(
        self, mock_qdrant_client, mock_neo4j_parser, mock_cpu_executor
    ):
        """Test initialization with provided dependencies."""
        service = UnifiedIndexingService(
            qdrant_client=mock_qdrant_client,
            neo4j_parser=mock_neo4j_parser,
            cpu_executor=mock_cpu_executor,
        )

        assert service.qdrant_client == mock_qdrant_client
        assert service.neo4j_parser == mock_neo4j_parser
        assert service.cpu_executor == mock_cpu_executor

    @patch("src.services.unified_indexing_service.get_qdrant_client")
    def test_init_without_qdrant_client(
        self, mock_get_qdrant_client, mock_neo4j_parser, mock_cpu_executor
    ):
        """Test initialization without Qdrant client."""
        mock_qdrant_client = Mock()
        mock_get_qdrant_client.return_value = mock_qdrant_client

        service = UnifiedIndexingService(
            qdrant_client=None,
            neo4j_parser=mock_neo4j_parser,
            cpu_executor=mock_cpu_executor,
        )

        assert service.qdrant_client == mock_qdrant_client
        mock_get_qdrant_client.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "NEO4J_URI": "bolt://test:7687",
            "NEO4J_USER": "test_user",
            "NEO4J_PASSWORD": "test_password",
        },
    )
    @patch("src.services.unified_indexing_service.DirectNeo4jExtractor")
    def test_init_without_neo4j_parser(
        self, mock_neo4j_extractor, mock_qdrant_client, mock_cpu_executor
    ):
        """Test initialization without Neo4j parser."""
        mock_neo4j_instance = Mock()
        mock_neo4j_extractor.return_value = mock_neo4j_instance

        service = UnifiedIndexingService(
            qdrant_client=mock_qdrant_client,
            neo4j_parser=None,
            cpu_executor=mock_cpu_executor,
        )

        assert service.neo4j_parser == mock_neo4j_instance
        mock_neo4j_extractor.assert_called_once_with(
            "bolt://test:7687", "test_user", "test_password"
        )

    @pytest.mark.asyncio
    async def test_process_repository_unified_success(self, unified_indexing_service):
        """Test successful repository processing."""
        # Create a mock request
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.BOTH,
            file_types=[".py"],
            max_files=10,
            chunk_size=1000,
            max_size_mb=100,
        )

        # Mock the internal methods
        with (
            patch.object(unified_indexing_service, "_clone_repository") as mock_clone,
            patch.object(
                unified_indexing_service, "_discover_repository_files"
            ) as mock_discover,
            patch.object(
                unified_indexing_service, "_process_files_unified"
            ) as mock_process,
        ):
            # Mock return values
            mock_temp_dir = Mock()
            mock_clone.return_value = mock_temp_dir

            mock_discovered_files = [Mock()]
            mock_discover.return_value = mock_discovered_files

            # Create a mock file processing result
            mock_file_result = FileProcessingResult(
                file_id="test_file_id",
                file_path="/test/file.py",
                relative_path="file.py",
                language="python",
                file_type=".py",
            )

            # Mock the async iterator
            async def mock_process_files(*args, **kwargs):
                yield mock_file_result

            mock_process.return_value = mock_process_files()

            # Call the method
            response = await unified_indexing_service.process_repository_unified(
                request
            )

            # Verify the response
            assert isinstance(response, UnifiedIndexingResponse)
            assert response.success is True
            assert response.repo_url == "https://github.com/test/repo"
            assert len(response.file_results) == 1

    @pytest.mark.asyncio
    async def test_process_repository_unified_no_files(self, unified_indexing_service):
        """Test repository processing with no files found."""
        # Create a mock request
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.BOTH,
            file_types=[".py"],
            max_files=10,
            chunk_size=1000,
            max_size_mb=100,
        )

        # Mock the internal methods to return no files
        with (
            patch.object(unified_indexing_service, "_clone_repository") as mock_clone,
            patch.object(
                unified_indexing_service, "_discover_repository_files"
            ) as mock_discover,
        ):
            mock_temp_dir = Mock()
            mock_clone.return_value = mock_temp_dir
            mock_discover.return_value = []  # No files found

            # Call the method
            response = await unified_indexing_service.process_repository_unified(
                request
            )

            # Verify the response
            assert isinstance(response, UnifiedIndexingResponse)
            assert response.success is True  # Still successful, just no files
            assert len(response.file_results) == 0
            assert len(response.errors) == 1
            assert "No files found" in response.errors[0]

    @pytest.mark.asyncio
    async def test_process_repository_unified_with_exception(
        self, unified_indexing_service
    ):
        """Test repository processing with exception."""
        # Create a mock request
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.BOTH,
            file_types=[".py"],
            max_files=10,
            chunk_size=1000,
            max_size_mb=100,
        )

        # Mock _clone_repository to raise an exception
        with patch.object(
            unified_indexing_service,
            "_clone_repository",
            side_effect=Exception("Clone failed"),
        ):
            # Call the method
            response = await unified_indexing_service.process_repository_unified(
                request
            )

            # Verify the response
            assert isinstance(response, UnifiedIndexingResponse)
            assert response.success is False
            assert len(response.errors) == 1
            assert "Clone failed" in response.errors[0]

    def test_chunk_content_small_content(self, unified_indexing_service):
        """Test chunking small content."""
        content = "This is a small piece of content."
        chunks = unified_indexing_service._chunk_content(content, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == content

    def test_chunk_content_large_content(self, unified_indexing_service):
        """Test chunking large content."""
        content = "A" * 250  # 250 characters
        chunks = unified_indexing_service._chunk_content(content, chunk_size=100)
        assert len(chunks) == 3
        assert len(chunks[0]) <= 100
        assert len(chunks[1]) <= 100
        assert len(chunks[2]) <= 100

    def test_detect_file_language(self, unified_indexing_service):
        """Test file language detection."""
        test_cases = [
            (Path("test.py"), "python"),
            (Path("test.js"), "javascript"),
            (Path("test.ts"), "typescript"),
            (Path("test.java"), "java"),
            (Path("test.unknown"), "text"),
        ]

        for file_path, expected_language in test_cases:
            language = unified_indexing_service._detect_file_language(file_path)
            assert language == expected_language

    def test_is_code_file(self, unified_indexing_service):
        """Test code file detection."""
        code_files = [
            Path("test.py"),
            Path("test.js"),
            Path("test.java"),
            Path("test.cpp"),
        ]

        non_code_files = [
            Path("test.txt"),
            Path("test.md"),
            Path("test.json"),
            Path("test.unknown"),
        ]

        for file_path in code_files:
            assert unified_indexing_service._is_code_file(file_path) is True

        for file_path in non_code_files:
            assert unified_indexing_service._is_code_file(file_path) is False


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    @pytest.mark.asyncio
    async def test_process_repository_unified_convenience_function(self):
        """Test the process_repository_unified convenience function."""
        with patch(
            "src.services.unified_indexing_service.UnifiedIndexingService"
        ) as mock_service_class:
            # Create mock service instance
            mock_service_instance = AsyncMock()
            mock_response = UnifiedIndexingResponse(
                success=True,
                repo_url="https://github.com/test/repo",
                repo_name="test/repo",
                destination="both",
                files_processed=0,
                start_time="2023-01-01T00:00:00",
            )
            mock_service_instance.process_repository_unified.return_value = (
                mock_response
            )
            mock_service_instance.cleanup = AsyncMock()
            mock_service_class.return_value = mock_service_instance

            # Call the function
            response = await process_repository_unified("https://github.com/test/repo")

            # Verify the response
            assert isinstance(response, UnifiedIndexingResponse)
            assert response.success is True

            # Verify service was created and used correctly
            mock_service_class.assert_called_once()
            mock_service_instance.process_repository_unified.assert_called_once()
            mock_service_instance.cleanup.assert_called_once()
