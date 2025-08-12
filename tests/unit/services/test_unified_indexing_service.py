"""
Tests for the unified indexing service.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from src.services.unified_indexing_service import (
    UnifiedIndexingService,
    ResourceManager,
    ProgressTracker,
    process_repository_unified,
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


class TestUnifiedIndexingModels:
    """Test cases for the unified indexing models."""

    def test_indexing_destination_enum(self):
        """Test IndexingDestination enum."""
        assert IndexingDestination.QDRANT.value == "qdrant"
        assert IndexingDestination.NEO4J.value == "neo4j"
        assert IndexingDestination.BOTH.value == "both"

    def test_unified_indexing_request(self):
        """Test UnifiedIndexingRequest class."""
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.BOTH,
            file_types=[".py", ".js"],
            max_files=10,
            chunk_size=1000,
            max_size_mb=100,
        )

        assert request.repo_url == "https://github.com/test/repo"
        assert request.destination == IndexingDestination.BOTH
        assert request.file_types == [".py", ".js"]
        assert request.max_files == 10
        assert request.chunk_size == 1000
        assert request.max_size_mb == 100

        # Test property methods
        assert request.should_process_rag is True
        assert request.should_process_kg is True

    def test_unified_indexing_request_rag_only(self):
        """Test UnifiedIndexingRequest for RAG only."""
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.QDRANT,
        )

        assert request.should_process_rag is True
        assert request.should_process_kg is False

    def test_unified_indexing_request_kg_only(self):
        """Test UnifiedIndexingRequest for Knowledge Graph only."""
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.NEO4J,
        )

        assert request.should_process_rag is False
        assert request.should_process_kg is True

    def test_file_processing_result(self):
        """Test FileProcessingResult class."""
        result = FileProcessingResult(
            file_id="test_file_id",
            relative_path="test/file.py",
            file_path="/full/path/to/test/file.py",
            language="python",
            file_type=".py",
            processed_for_rag=True,
            processed_for_kg=True,
            rag_chunks=5,
            kg_entities=10,
            processing_time_seconds=2.5,
        )

        assert result.file_id == "test_file_id"
        assert result.relative_path == "test/file.py"
        assert result.file_path == "/full/path/to/test/file.py"
        assert result.language == "python"
        assert result.file_type == ".py"
        assert result.processed_for_rag is True
        assert result.processed_for_kg is True
        assert result.rag_chunks == 5
        assert result.kg_entities == 10
        assert result.processing_time_seconds == 2.5
        assert result.is_successful is True

    def test_file_processing_result_with_errors(self):
        """Test FileProcessingResult with errors."""
        result = FileProcessingResult(
            file_id="test_file_id",
            relative_path="test/file.py",
            errors=["Processing failed"],
        )

        assert result.is_successful is False
        assert len(result.errors) == 1

    def test_unified_indexing_response(self):
        """Test UnifiedIndexingResponse class."""
        start_time = datetime.now()

        response = UnifiedIndexingResponse(
            success=True,
            repo_url="https://github.com/test/repo",
            repo_name="test-repo",
            destination="both",
            files_processed=0,
            start_time=start_time,
        )

        assert response.success is True
        assert response.repo_url == "https://github.com/test/repo"
        assert response.repo_name == "test-repo"
        assert response.destination == "both"
        assert response.files_processed == 0
        assert response.start_time == start_time

    def test_unified_indexing_response_add_file_result(self):
        """Test adding file results to response."""
        response = UnifiedIndexingResponse(
            success=False,
            repo_url="https://github.com/test/repo",
            repo_name="test-repo",
            destination="both",
            files_processed=0,
            start_time=datetime.now(),
        )

        # Add file results
        result1 = FileProcessingResult(
            file_id="file1",
            relative_path="file1.py",
            processed_for_rag=True,
            rag_chunks=5,
        )
        result2 = FileProcessingResult(
            file_id="file2",
            relative_path="file2.py",
            processed_for_kg=True,
            kg_entities=10,
        )

        response.add_file_result(result1)
        response.add_file_result(result2)

        assert len(response.file_results) == 2
        assert response.files_processed == 2
        assert response.qdrant_documents == 5
        assert response.neo4j_nodes == 10

    def test_unified_indexing_response_finalize(self):
        """Test response finalization."""
        import time

        response = UnifiedIndexingResponse(
            success=False,
            repo_url="https://github.com/test/repo",
            repo_name="test-repo",
            destination="both",
            files_processed=0,
            start_time=datetime.now(),
        )

        # Add a successful file result
        result = FileProcessingResult(
            file_id="file1", relative_path="file1.py", processed_for_rag=True
        )
        response.add_file_result(result)

        # Add a small delay to ensure measurable processing time
        time.sleep(0.01)

        # Finalize
        response.finalize()

        assert response.success is True
        assert response.end_time is not None
        assert response.processing_time_seconds >= 0  # Allow for very fast processing

    def test_unified_indexing_response_success_rate(self):
        """Test success rate calculation."""
        response = UnifiedIndexingResponse(
            success=False,
            repo_url="https://github.com/test/repo",
            repo_name="test-repo",
            destination="both",
            files_processed=0,
            start_time=datetime.now(),
        )

        # Add mixed results
        successful_result = FileProcessingResult(
            file_id="file1", relative_path="file1.py", processed_for_rag=True
        )
        failed_result = FileProcessingResult(
            file_id="file2", relative_path="file2.py", errors=["Failed"]
        )

        response.add_file_result(successful_result)
        response.add_file_result(failed_result)

        assert response.success_rate == 50.0

    def test_unified_indexing_response_error_summary(self):
        """Test error summary generation."""
        response = UnifiedIndexingResponse(
            success=False,
            repo_url="https://github.com/test/repo",
            repo_name="test-repo",
            destination="both",
            files_processed=0,
            start_time=datetime.now(),
            errors=["System error"],
        )

        # Add file with error
        failed_result = FileProcessingResult(
            file_id="file1", relative_path="file1.py", errors=["File error"]
        )
        response.add_file_result(failed_result)

        error_summary = response.error_summary
        assert error_summary["has_errors"] is True
        assert error_summary["error_count"] == 2  # 1 system + 1 file
        assert error_summary["failed_files"] == 1


class TestUnifiedIndexingService:
    """Test cases for the UnifiedIndexingService class."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        return Mock()

    @pytest.fixture
    def mock_neo4j_parser(self):
        """Create a mock Neo4j parser."""
        parser = Mock()
        parser.initialize = AsyncMock()
        return parser

    @pytest.fixture
    def unified_indexing_service(self, mock_qdrant_client, mock_neo4j_parser):
        """Create a UnifiedIndexingService instance."""
        return UnifiedIndexingService(
            qdrant_client=mock_qdrant_client, neo4j_parser=mock_neo4j_parser
        )

    def test_init_with_provided_dependencies(
        self, mock_qdrant_client, mock_neo4j_parser
    ):
        """Test initialization with provided dependencies."""
        service = UnifiedIndexingService(
            qdrant_client=mock_qdrant_client, neo4j_parser=mock_neo4j_parser
        )

        assert service.qdrant_client == mock_qdrant_client
        assert service.neo4j_parser == mock_neo4j_parser

    @patch("src.services.unified_indexing_service.get_qdrant_client")
    def test_init_without_qdrant_client(self, mock_get_qdrant_client):
        """Test initialization without Qdrant client."""
        mock_qdrant_client = Mock()
        mock_get_qdrant_client.return_value = mock_qdrant_client

        service = UnifiedIndexingService(qdrant_client=None)

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
    def test_init_creates_neo4j_config(self, mock_neo4j_extractor):
        """Test initialization creates Neo4j config when parser not provided."""
        service = UnifiedIndexingService()

        # Should store config for lazy initialization
        assert service._neo4j_config is not None
        assert service._neo4j_config["uri"] == "bolt://test:7687"
        assert service._neo4j_config["user"] == "test_user"
        assert service._neo4j_config["password"] == "test_password"

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
                processed_for_rag=True,
                processed_for_kg=True,
                rag_chunks=5,
                kg_entities=10,
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
            assert response.qdrant_documents == 5
            assert response.neo4j_nodes == 10

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
            assert response.success is False  # Should be False when no files processed
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

    @pytest.mark.asyncio
    async def test_process_repository_unified_with_neo4j_result(
        self, unified_indexing_service
    ):
        """Test repository processing with Neo4j batch processing results."""
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
            patch.object(
                unified_indexing_service, "_batch_process_neo4j_analyses"
            ) as mock_batch,
        ):
            # Mock return values
            mock_clone.return_value = Mock()
            mock_discover.return_value = [Mock()]

            # Mock file processing result
            mock_file_result = FileProcessingResult(
                file_id="test_file_id",
                file_path="/test/file.py",
                relative_path="file.py",
                language="python",
                file_type=".py",
                processed_for_kg=True,
                kg_entities=1,  # Initial placeholder value
            )

            async def mock_process_files(*args, **kwargs):
                yield mock_file_result

            mock_process.return_value = mock_process_files()

            # Mock Neo4j analyses exist
            unified_indexing_service._neo4j_analyses = [{"test": "data"}]

            # Mock batch processing result
            neo4j_result = {"entities_created": 150, "files_processed": 1}
            mock_batch.return_value = neo4j_result

            # Call the method
            response = await unified_indexing_service.process_repository_unified(
                request
            )

            # Verify the response incorporates Neo4j results
            assert response.neo4j_nodes == 150
            assert response.file_results[0].kg_entities == 150  # Should be updated

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

    def test_chunk_content_empty_content(self, unified_indexing_service):
        """Test chunking empty content."""
        content = ""
        chunks = unified_indexing_service._chunk_content(content, chunk_size=100)
        assert len(chunks) == 0

    def test_chunk_content_with_sentence_breaks(self, unified_indexing_service):
        """Test chunking respects sentence breaks."""
        content = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = unified_indexing_service._chunk_content(content, chunk_size=30)

        # Should break into multiple chunks for long content
        assert len(chunks) > 1

        # Test with a longer content that will definitely break at sentence boundaries
        long_content = "Short. " * 20  # Each "Short. " is 7 chars, total 140 chars
        long_chunks = unified_indexing_service._chunk_content(
            long_content, chunk_size=50
        )

        # Should break into multiple chunks
        assert len(long_chunks) > 1

        # At least some chunks should end with periods (sentence breaks)
        chunks_ending_with_period = [
            chunk for chunk in long_chunks if chunk.strip().endswith(".")
        ]
        assert len(chunks_ending_with_period) > 0

    def test_detect_file_language(self, unified_indexing_service):
        """Test file language detection."""
        test_cases = [
            (Path("test.py"), "python"),
            (Path("test.js"), "javascript"),
            (Path("test.ts"), "typescript"),
            (Path("test.tsx"), "typescript"),
            (Path("test.jsx"), "javascript"),
            (Path("test.java"), "java"),
            (Path("test.go"), "go"),
            (Path("test.rs"), "rust"),
            (Path("test.cpp"), "cpp"),
            (Path("test.c"), "c"),
            (Path("test.md"), "markdown"),
            (Path("test.json"), "json"),
            (Path("test.yaml"), "yaml"),
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
            Path("test.ts"),
            Path("test.tsx"),
            Path("test.jsx"),
            Path("test.java"),
            Path("test.cpp"),
            Path("test.c"),
            Path("test.go"),
            Path("test.rs"),
        ]

        non_code_files = [
            Path("test.txt"),
            Path("test.md"),
            Path("test.json"),
            Path("test.yaml"),
            Path("test.unknown"),
        ]

        for file_path in code_files:
            assert unified_indexing_service._is_code_file(file_path) is True

        for file_path in non_code_files:
            assert unified_indexing_service._is_code_file(file_path) is False

    @pytest.mark.asyncio
    async def test_estimate_kg_entities(self, unified_indexing_service):
        """Test knowledge graph entity estimation."""
        python_content = """
class TestClass:
    def method1(self):
        pass
    
    def method2(self):
        pass

def function1():
    pass

def function2():
    pass
"""

        entities = await unified_indexing_service._estimate_kg_entities(
            python_content, "python"
        )
        assert entities == 5  # 1 class + 4 functions/methods

        javascript_content = """
function myFunction() {
    return true;
}

const myConst = 5;
let myVar = 10;
"""

        entities = await unified_indexing_service._estimate_kg_entities(
            javascript_content, "javascript"
        )
        assert entities == 3  # function + const + let

    @pytest.mark.asyncio
    async def test_cleanup(self, unified_indexing_service):
        """Test service cleanup."""
        # Should not raise any exceptions
        await unified_indexing_service.cleanup()


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
                repo_name="test-repo",
                destination="both",
                files_processed=1,
                start_time=datetime.now(),
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

    @pytest.mark.asyncio
    async def test_process_repository_unified_with_custom_params(self):
        """Test convenience function with custom parameters."""
        with patch(
            "src.services.unified_indexing_service.UnifiedIndexingService"
        ) as mock_service_class:
            mock_service_instance = AsyncMock()
            mock_response = UnifiedIndexingResponse(
                success=True,
                repo_url="https://github.com/test/repo",
                repo_name="test-repo",
                destination="qdrant",
                files_processed=1,
                start_time=datetime.now(),
            )
            mock_service_instance.process_repository_unified.return_value = (
                mock_response
            )
            mock_service_instance.cleanup = AsyncMock()
            mock_service_class.return_value = mock_service_instance

            # Call with custom parameters
            response = await process_repository_unified(
                repo_url="https://github.com/test/repo",
                destination=IndexingDestination.QDRANT,
                file_types=[".py", ".js"],
                max_files=20,
                chunk_size=2000,
                max_size_mb=200,
            )

            # Verify service was called with correct request
            call_args = mock_service_instance.process_repository_unified.call_args[0][0]
            assert call_args.repo_url == "https://github.com/test/repo"
            assert call_args.destination == IndexingDestination.QDRANT
            assert call_args.file_types == [".py", ".js"]
            assert call_args.max_files == 20
            assert call_args.chunk_size == 2000
            assert call_args.max_size_mb == 200
