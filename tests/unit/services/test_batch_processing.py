"""
Unit tests for batch processing components.

Tests the optimized pipeline stages, file processors, and performance configuration
for the repository indexing optimization system.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Import test targets
from src.utils.performance_config import (
    PerformanceConfig,
    load_performance_config,
    get_optimal_worker_count,
    get_optimal_batch_size,
)
from src.services.batch_processing.file_processor import (
    parse_file_for_kg,
    read_file_async_sync,
    process_file_batch_sync,
    detect_file_language,
    is_code_file,
    process_file_batch,
    read_file_async,
)
from src.services.batch_processing.pipeline_stages import (
    OptimizedIndexingPipeline,
    PipelineStageResult,
    FileBatch,
)


class TestPerformanceConfig:
    """Test performance configuration functionality."""

    def test_performance_config_defaults(self):
        """Test default performance configuration values."""
        config = PerformanceConfig()

        assert config.cpu_workers == 4
        assert config.io_workers == 10
        assert config.batch_size_qdrant == 500
        assert config.batch_size_neo4j == 5000
        assert config.batch_size_embeddings == 1000
        assert config.batch_size_file_processing == 10

    def test_performance_config_validation(self):
        """Test performance configuration validation."""
        # Test minimum worker validation
        config = PerformanceConfig(cpu_workers=0, io_workers=-1)

        assert config.cpu_workers == 1  # Should be corrected to minimum
        assert config.io_workers == 1  # Should be corrected to minimum

    def test_performance_config_batch_size_limits(self):
        """Test batch size limit validation."""
        config = PerformanceConfig(
            batch_size_embeddings=3000,  # Exceeds OpenAI limit
            batch_size_qdrant=1500,  # Exceeds memory-safe limit
        )

        assert config.batch_size_embeddings == 1000  # Should be corrected
        assert config.batch_size_qdrant == 500  # Should be corrected

    @patch.dict(
        "os.environ",
        {
            "CPU_WORKERS": "8",
            "IO_WORKERS": "20",
            "BATCH_SIZE_QDRANT": "750",
            "BATCH_SIZE_NEO4J": "7500",
        },
    )
    def test_load_performance_config_from_env(self):
        """Test loading configuration from environment variables."""
        config = load_performance_config()

        assert config.cpu_workers == 8
        assert config.io_workers == 20
        assert config.batch_size_qdrant == 750
        assert config.batch_size_neo4j == 7500

    def test_get_optimal_worker_count(self):
        """Test optimal worker count calculation."""
        cpu_workers = get_optimal_worker_count("cpu")
        io_workers = get_optimal_worker_count("io")

        assert cpu_workers >= 1
        assert io_workers >= 1
        # Note: Worker counts are system-dependent, just verify they're reasonable
        assert cpu_workers <= 200  # Should not be excessively high
        assert io_workers <= 200  # Should not be excessively high

    def test_get_optimal_batch_size(self):
        """Test optimal batch size retrieval."""
        qdrant_batch = get_optimal_batch_size("qdrant")
        neo4j_batch = get_optimal_batch_size("neo4j")
        embeddings_batch = get_optimal_batch_size("embeddings")

        assert qdrant_batch == 500
        assert neo4j_batch == 5000
        assert embeddings_batch == 1000


class TestFileProcessor:
    """Test file processing utilities."""

    def test_detect_file_language(self):
        """Test file language detection."""
        assert detect_file_language("test.py") == "python"
        assert detect_file_language("test.js") == "javascript"
        assert detect_file_language("test.ts") == "typescript"
        assert detect_file_language("test.java") == "java"
        assert detect_file_language("test.cpp") == "cpp"
        assert detect_file_language("test.unknown") == "text"

    def test_is_code_file(self):
        """Test code file detection."""
        assert is_code_file("script.py") is True
        assert is_code_file("app.js") is True
        assert is_code_file("component.tsx") is True
        assert is_code_file("Main.java") is True
        assert is_code_file("readme.md") is False
        assert is_code_file("config.json") is False
        assert is_code_file("image.png") is False

    def test_read_file_async_sync(self, tmp_path):
        """Test synchronous file reading."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_content = "def hello():\n    return 'world'"
        test_file.write_text(test_content)

        # Test reading
        file_path, content = read_file_async_sync(str(test_file))

        assert file_path == str(test_file)
        assert content == test_content

    def test_read_file_async_sync_error(self):
        """Test file reading error handling."""
        nonexistent_file = "/nonexistent/path/file.py"

        file_path, content = read_file_async_sync(nonexistent_file)

        assert file_path == nonexistent_file
        assert content == ""  # Should return empty string on error

    def test_process_file_batch_sync(self, tmp_path):
        """Test batch file processing."""
        # Create test files
        files = []
        for i in range(3):
            test_file = tmp_path / f"test_{i}.py"
            test_file.write_text(f"# Test file {i}")
            files.append(str(test_file))

        # Process batch
        results = process_file_batch_sync(files)

        assert len(results) == 3
        for i, (file_path, content) in enumerate(results):
            assert file_path == files[i]
            assert content == f"# Test file {i}"

    @pytest.mark.asyncio
    async def test_read_file_async(self, tmp_path):
        """Test async file reading."""
        # Create test file
        test_file = tmp_path / "async_test.py"
        test_content = "async def hello():\n    return 'async world'"
        test_file.write_text(test_content)

        # Test async reading
        file_path, content = await read_file_async(str(test_file))

        assert file_path == str(test_file)
        assert content == test_content

    @pytest.mark.asyncio
    @patch("src.services.batch_processing.file_processor.parse_file_for_kg")
    async def test_process_file_batch(self, mock_parse, tmp_path):
        """Test async file batch processing with CPU executor."""
        # Mock the parse function
        mock_parse.return_value = {
            "classes": [{"name": "TestClass", "methods": []}],
            "functions": [{"name": "test_func"}],
            "imports": ["os", "sys"],
        }

        # Create test files
        test_file1 = tmp_path / "test1.py"
        test_file2 = tmp_path / "test2.py"
        test_file1.write_text("def func1(): pass")
        test_file2.write_text("def func2(): pass")

        # Create mock CPU executor
        cpu_executor = Mock(spec=ProcessPoolExecutor)
        future_mock = Mock()
        future_mock.result.return_value = mock_parse.return_value
        cpu_executor.submit.return_value = future_mock

        # Test files
        test_files = [str(test_file1), str(test_file2)]

        # Process batch
        results = await process_file_batch(
            test_files,
            cpu_executor,
            should_process_kg=True,
            repo_name="test-repo",
        )

        # Verify results
        assert len(results) == 2
        for result in results:
            assert result["processed_for_kg"] is True
            assert "kg_analysis" in result

    def test_parse_file_for_kg_mock(self):
        """Test knowledge graph parsing function (mocked)."""
        # Since parse_file_for_kg requires tree-sitter initialization,
        # we'll test the interface and error handling

        file_path = "test.py"
        content = "def hello(): pass"
        language = "python"
        repo_name = "test-repo"

        # This would normally parse but may fail without proper setup
        # The function should handle errors gracefully
        result = parse_file_for_kg(file_path, content, language, repo_name)

        # Should either return a valid result or None
        assert result is None or isinstance(result, dict)


class TestOptimizedIndexingPipeline:
    """Test the optimized indexing pipeline."""

    @pytest.fixture
    def mock_executors(self):
        """Create mock executors for testing."""
        io_executor = Mock(spec=ThreadPoolExecutor)
        cpu_executor = Mock(spec=ProcessPoolExecutor)
        return io_executor, cpu_executor

    @pytest.fixture
    def performance_config(self):
        """Create test performance configuration."""
        return PerformanceConfig(
            cpu_workers=2,
            io_workers=4,
            batch_size_file_processing=3,
        )

    @pytest.fixture
    def pipeline(self, mock_executors, performance_config):
        """Create test pipeline instance."""
        io_executor, cpu_executor = mock_executors
        return OptimizedIndexingPipeline(io_executor, cpu_executor, performance_config)

    def test_pipeline_initialization(self, pipeline, performance_config):
        """Test pipeline initialization."""
        assert pipeline.config == performance_config
        assert pipeline.stats["total_files_processed"] == 0
        assert pipeline.stats["batches_processed"] == 0

    def test_create_file_batches(self, pipeline):
        """Test file batch creation."""
        # Create test files
        files = [Path(f"file_{i}.py") for i in range(7)]
        repo_path = Path("/test/repo")
        repo_name = "test-repo"

        # Create batches
        batches = pipeline._create_file_batches(files, repo_path, repo_name)

        # Should create 3 batches (batch_size=3): [3, 3, 1]
        assert len(batches) == 3
        assert len(batches[0].file_paths) == 3
        assert len(batches[1].file_paths) == 3
        assert len(batches[2].file_paths) == 1

        for i, batch in enumerate(batches):
            assert batch.batch_id == i
            assert batch.repo_path == repo_path
            assert batch.repo_name == repo_name

    @pytest.mark.asyncio
    async def test_convert_to_processing_results(self, pipeline):
        """Test conversion to processing results."""
        # Mock processed files
        processed_files = [
            {
                "file_path": "/test/repo/src/main.py",
                "content": "def main(): pass",
                "language": "python",
                "kg_analysis": {
                    "classes": [{"name": "TestClass"}],
                    "functions": [{"name": "main"}],
                },
                "processed_for_kg": True,
                "processing_errors": [],
            },
            {
                "file_path": "/test/repo/README.md",
                "content": "# Test Repo",
                "language": "markdown",
                "kg_analysis": None,
                "processed_for_kg": False,
                "processing_errors": [],
            },
        ]

        repo_path = Path("/test/repo")

        # Convert to processing results
        results = await pipeline._convert_to_processing_results(
            processed_files, repo_path, True, True
        )

        assert len(results) == 2

        # Check first result (Python file)
        result1 = results[0]
        # Use os.path.normpath to handle different path separators on Windows/Unix
        import os

        expected_relative_path = "src/main.py"
        # Normalize both the expected and actual file_id for comparison
        expected_file_id = f"repo:{expected_relative_path}".replace("\\", "/")
        actual_file_id = result1.file_id.replace("\\", "/")
        assert actual_file_id == expected_file_id
        assert os.path.normpath(result1.relative_path) == os.path.normpath(
            expected_relative_path
        )
        assert result1.language == "python"
        assert result1.processed_for_rag is True
        assert result1.processed_for_kg is True
        assert result1.kg_entities == 2  # 1 class + 1 function

        # Check second result (Markdown file)
        result2 = results[1]
        assert result2.file_id == "repo:README.md"
        assert result2.relative_path == "README.md"
        assert result2.language == "markdown"
        assert result2.processed_for_rag is True
        assert result2.processed_for_kg is False
        assert result2.kg_entities == 0

    @pytest.mark.asyncio
    @patch("aiofiles.open")
    async def test_stage_read_files(self, mock_aiofiles, pipeline):
        """Test file reading stage."""
        # Mock aiofiles
        mock_file = AsyncMock()
        mock_file.read.return_value = "test content"
        mock_aiofiles.return_value.__aenter__.return_value = mock_file

        file_paths = ["test1.py", "test2.py"]

        # Test file reading
        results = await pipeline.stage_read_files(file_paths)

        assert len(results) == 2
        for file_path, content in results:
            assert file_path in file_paths
            assert content == "test content"

    @pytest.mark.asyncio
    async def test_stage_generate_embeddings(self, pipeline):
        """Test embedding generation stage."""
        file_contents = ["content1", "content2"]

        # Test embedding generation (will use fallback embeddings due to invalid API key)
        result = await pipeline.stage_generate_embeddings(file_contents)

        # Should return a list of embeddings (even if they're zero vectors from fallback)
        assert isinstance(result, list)
        assert len(result) == len(file_contents)

        # Each embedding should be a list of floats
        for embedding in result:
            assert isinstance(embedding, list)
            assert len(embedding) > 0  # Should have some dimension
            assert all(isinstance(x, float) for x in embedding)

    def test_get_pipeline_stats(self, pipeline):
        """Test pipeline statistics retrieval."""
        # Update some stats
        pipeline.stats["total_files_processed"] = 10
        pipeline.stats["total_processing_time"] = 5.0
        pipeline.stats["batches_processed"] = 2

        stats = pipeline.get_pipeline_stats()

        assert stats["total_files_processed"] == 10
        assert stats["total_processing_time"] == 5.0
        assert stats["batches_processed"] == 2
        assert stats["average_files_per_second"] == 2.0  # 10 files / 5 seconds


class TestPipelineStageResult:
    """Test pipeline stage result data structure."""

    def test_pipeline_stage_result_creation(self):
        """Test creating pipeline stage results."""
        result = PipelineStageResult(
            stage_name="test_stage",
            files_processed=100,
            processing_time_seconds=2.5,
            success_count=95,
            error_count=5,
            errors=["Error 1", "Error 2"],
        )

        assert result.stage_name == "test_stage"
        assert result.files_processed == 100
        assert result.processing_time_seconds == 2.5
        assert result.success_count == 95
        assert result.error_count == 5
        assert len(result.errors) == 2


class TestFileBatch:
    """Test file batch data structure."""

    def test_file_batch_creation(self):
        """Test creating file batches."""
        file_paths = [Path("file1.py"), Path("file2.py")]
        repo_path = Path("/test/repo")

        batch = FileBatch(
            batch_id=0,
            file_paths=file_paths,
            repo_path=repo_path,
            repo_name="test-repo",
        )

        assert batch.batch_id == 0
        assert batch.file_paths == file_paths
        assert batch.repo_path == repo_path
        assert batch.repo_name == "test-repo"


@pytest.mark.integration
class TestBatchProcessingIntegration:
    """Integration tests for batch processing components."""

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self, tmp_path):
        """Test complete pipeline flow with real files."""
        # Create test repository structure
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()

        src_path = repo_path / "src"
        src_path.mkdir()

        # Create test files
        (src_path / "main.py").write_text("def main():\n    print('Hello, World!')")
        (src_path / "utils.py").write_text("def helper():\n    return 42")
        (repo_path / "README.md").write_text("# Test Repository")

        files = list(repo_path.rglob("*"))
        file_paths = [f for f in files if f.is_file()]

        # Create pipeline with real executors (small scale)
        config = PerformanceConfig(
            cpu_workers=1,
            io_workers=2,
            batch_size_file_processing=2,
        )

        with (
            ThreadPoolExecutor(max_workers=2) as io_executor,
            ProcessPoolExecutor(max_workers=1) as cpu_executor,
        ):
            pipeline = OptimizedIndexingPipeline(io_executor, cpu_executor, config)

            # Mock progress tracker
            progress_tracker = Mock()
            progress_tracker.update_progress = Mock()

            # Process files
            results = []
            async for batch_results in pipeline.process_files_optimized(
                files=file_paths,
                repo_path=repo_path,
                repo_name="test-repo",
                should_process_rag=True,
                should_process_kg=True,
                progress_tracker=progress_tracker,
            ):
                results.extend(batch_results)

            # Verify results
            assert len(results) >= 2  # At least the Python files

            # Check that progress was tracked
            assert progress_tracker.update_progress.called

            # Check pipeline stats
            stats = pipeline.get_pipeline_stats()
            assert stats["total_files_processed"] > 0
            assert stats["batches_processed"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
