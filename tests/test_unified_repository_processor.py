"""
Comprehensive tests for the unified repository processor.

Tests the UnifiedIndexingService and related components that provide
unified GitHub repository processing across RAG and Neo4j systems.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
from datetime import datetime

from src.services.unified_indexing_service import (
    UnifiedIndexingService,
    ResourceManager,
    ProgressTracker,
    process_repository_unified,
)
from src.models.unified_indexing_models import (
    UnifiedIndexingRequest,
    UnifiedIndexingResponse,
    FileProcessingResult,
    IndexingDestination,
)
from src.utils.file_id_generator import generate_file_id, extract_repo_name


class TestResourceManager:
    """Test the ResourceManager class."""
    
    @pytest.mark.asyncio
    async def test_resource_manager_context(self):
        """Test ResourceManager as async context manager."""
        async with ResourceManager() as manager:
            assert manager is not None
            assert isinstance(manager._temp_directories, list)
            assert isinstance(manager._cleanup_tasks, list)
    
    @pytest.mark.asyncio
    async def test_register_temp_directory(self):
        """Test temporary directory registration."""
        async with ResourceManager() as manager:
            temp_dir = Path("/tmp/test")
            manager.register_temp_directory(temp_dir)
            assert temp_dir in manager._temp_directories
    
    @pytest.mark.asyncio
    async def test_register_cleanup_task(self):
        """Test cleanup task registration."""
        async with ResourceManager() as manager:
            cleanup_func = Mock()
            manager.register_cleanup_task(cleanup_func)
            assert cleanup_func in manager._cleanup_tasks
    
    @pytest.mark.asyncio 
    async def test_cleanup_all_resources(self):
        """Test comprehensive resource cleanup."""
        with patch('shutil.rmtree') as mock_rmtree, \
             patch('pathlib.Path.exists', return_value=True):
            
            async with ResourceManager() as manager:
                # Register resources
                temp_dir = Path("/tmp/test")
                manager.register_temp_directory(temp_dir)
                
                cleanup_func = Mock()
                async_cleanup_func = AsyncMock()
                manager.register_cleanup_task(cleanup_func)
                manager.register_cleanup_task(async_cleanup_func)
            
            # Verify cleanup was called
            cleanup_func.assert_called_once()
            async_cleanup_func.assert_called_once()
            mock_rmtree.assert_called_once_with(temp_dir)


class TestProgressTracker:
    """Test the ProgressTracker class."""
    
    def test_init(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker(total_files=10)
        assert tracker.total_files == 10
        assert tracker.processed_files == 0
        assert tracker.failed_files == 0
        assert tracker.current_operation == "Starting..."
        assert isinstance(tracker.start_time, datetime)
    
    def test_update_progress(self):
        """Test progress updates."""
        tracker = ProgressTracker(total_files=10)
        
        tracker.update_progress(processed_increment=2, failed_increment=1, operation="Processing files")
        
        assert tracker.processed_files == 2
        assert tracker.failed_files == 1
        assert tracker.current_operation == "Processing files"
    
    def test_update_progress_logging(self):
        """Test progress logging at milestones."""
        with patch('src.services.unified_indexing_service.logger') as mock_logger:
            tracker = ProgressTracker(total_files=20)
            
            # Update to trigger logging (every 10 files)
            tracker.update_progress(processed_increment=10)
            
            # Should log at 50% completion
            mock_logger.info.assert_called()
            log_call_args = mock_logger.info.call_args[0][0]
            assert "50.0%" in log_call_args


class TestUnifiedIndexingService:
    """Test the UnifiedIndexingService class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_qdrant_client = Mock()
        self.mock_neo4j_parser = Mock()
        self.service = UnifiedIndexingService(
            qdrant_client=self.mock_qdrant_client,
            neo4j_parser=self.mock_neo4j_parser
        )
    
    def test_init_with_clients(self):
        """Test service initialization with provided clients."""
        assert self.service.qdrant_client == self.mock_qdrant_client
        assert self.service.neo4j_parser == self.mock_neo4j_parser
        assert hasattr(self.service, 'github_processor')
        assert hasattr(self.service, 'executor')
    
    @patch('src.services.unified_indexing_service.get_qdrant_client')
    @patch('src.services.unified_indexing_service.DirectNeo4jExtractor')
    def test_init_without_clients(self, mock_parser_class, mock_get_client):
        """Test service initialization without provided clients."""
        mock_client = Mock()
        mock_parser = Mock()
        mock_get_client.return_value = mock_client
        mock_parser_class.return_value = mock_parser
        
        service = UnifiedIndexingService()
        
        assert service.qdrant_client == mock_client
        assert service.neo4j_parser == mock_parser
        mock_get_client.assert_called_once()
        mock_parser_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clone_repository_success(self):
        """Test successful repository cloning."""
        mock_resource_manager = Mock()
        mock_temp_dir = Path("/tmp/test_repo")
        
        with patch('tempfile.mkdtemp', return_value="/tmp/test_repo"), \
             patch('pathlib.Path', return_value=mock_temp_dir):
            
            self.service.github_processor.clone_repository_temp = Mock(return_value={
                "success": True,
                "temp_directory": "/tmp/test_repo"
            })
            
            result = await self.service._clone_repository(
                "https://github.com/test/repo",
                500,
                mock_resource_manager
            )
            
            assert result == mock_temp_dir
            mock_resource_manager.register_temp_directory.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clone_repository_failure(self):
        """Test repository cloning failure."""
        mock_resource_manager = Mock()
        
        with patch('tempfile.mkdtemp', return_value="/tmp/test_repo"):
            self.service.github_processor.clone_repository_temp = Mock(return_value={
                "success": False,
                "error": "Repository not found"
            })
            
            with pytest.raises(ValueError, match="Repository cloning failed"):
                await self.service._clone_repository(
                    "https://github.com/test/repo",
                    500,
                    mock_resource_manager
                )
    
    @pytest.mark.asyncio
    async def test_discover_repository_files(self):
        """Test repository file discovery."""
        mock_repo_path = Mock()
        mock_files = [
            Mock(is_file=Mock(return_value=True), suffix=".py"),
            Mock(is_file=Mock(return_value=True), suffix=".md"),
            Mock(is_file=Mock(return_value=False), suffix=".py"),  # Directory, should be ignored
        ]
        
        mock_repo_path.rglob.return_value = mock_files
        
        discovered = await self.service._discover_repository_files(
            mock_repo_path,
            [".py", ".md"],
            10
        )
        
        # Should return only files with matching extensions
        assert len(discovered) == 2
        mock_repo_path.rglob.assert_called_once_with('*')
    
    def test_detect_file_language(self):
        """Test file language detection."""
        assert self.service._detect_file_language(Path("test.py")) == "python"
        assert self.service._detect_file_language(Path("test.js")) == "javascript"
        assert self.service._detect_file_language(Path("test.ts")) == "typescript"
        assert self.service._detect_file_language(Path("test.md")) == "markdown"
        assert self.service._detect_file_language(Path("test.unknown")) == "text"
    
    def test_is_code_file(self):
        """Test code file detection."""
        assert self.service._is_code_file(Path("test.py")) is True
        assert self.service._is_code_file(Path("test.js")) is True
        assert self.service._is_code_file(Path("test.md")) is False
        assert self.service._is_code_file(Path("test.txt")) is False
    
    def test_chunk_content_small(self):
        """Test content chunking for small content."""
        content = "This is a small content."
        chunks = self.service._chunk_content(content, 1000)
        
        assert len(chunks) == 1
        assert chunks[0] == content
    
    def test_chunk_content_large(self):
        """Test content chunking for large content."""
        # Create content larger than chunk size
        content = "This is a sentence. " * 100  # ~2000 characters
        chunks = self.service._chunk_content(content, 500)
        
        assert len(chunks) > 1
        # Verify each chunk is within reasonable size bounds
        for chunk in chunks:
            assert len(chunk) <= 600  # Allow some flexibility for sentence boundaries
    
    def test_chunk_content_empty(self):
        """Test content chunking for empty content."""
        chunks = self.service._chunk_content("", 1000)
        assert len(chunks) == 0
    
    @pytest.mark.asyncio
    async def test_estimate_kg_entities_python(self):
        """Test KG entity estimation for Python code."""
        python_content = """
def function1():
    pass
    
class TestClass:
    def method1(self):
        pass
    
    def method2(self):
        pass
        """
        
        count = await self.service._estimate_kg_entities(python_content, "python")
        assert count == 4  # 1 function + 1 class + 2 methods (def statements)
    
    @pytest.mark.asyncio
    async def test_estimate_kg_entities_javascript(self):
        """Test KG entity estimation for JavaScript code."""
        js_content = """
function testFunction() {
    return true;
}

const myVar = 5;
let anotherVar = 10;
        """
        
        count = await self.service._estimate_kg_entities(js_content, "javascript")
        assert count == 3  # 1 function + 2 variables
    
    @pytest.mark.asyncio 
    async def test_process_file_for_rag_success(self):
        """Test successful RAG processing of a file."""
        # Setup test data
        file_path = Path("/repo/test.md")
        content = "This is test content for RAG processing."
        file_id = "test-repo:test.md"
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.QDRANT,
            chunk_size=1000
        )
        
        # Mock the RAG service functions
        with patch('src.services.unified_indexing_service.update_source_info'), \
             patch('src.services.unified_indexing_service.add_documents_to_vector_db'):
            
            result = await self.service._process_file_for_rag(
                file_path, content, file_id, request
            )
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_process_file_for_rag_error(self):
        """Test RAG processing with error."""
        file_path = Path("/repo/test.md")
        content = "Test content"
        file_id = "test-repo:test.md"
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.QDRANT
        )
        
        # Mock to raise an exception
        with patch('src.services.unified_indexing_service.add_documents_to_vector_db', 
                   side_effect=Exception("RAG error")):
            
            result = await self.service._process_file_for_rag(
                file_path, content, file_id, request
            )
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_process_file_for_neo4j_success(self):
        """Test successful Neo4j processing of a code file."""
        file_path = Path("/repo/test.py")
        content = "def test_function(): pass"
        file_id = "test-repo:test.py"
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.NEO4J
        )
        
        # Mock successful parsing
        with patch.object(self.service, '_parse_file_for_neo4j', return_value=True):
            result = await self.service._process_file_for_neo4j(
                file_path, content, file_id, request
            )
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_process_file_for_neo4j_non_code_file(self):
        """Test Neo4j processing of non-code file (should be skipped)."""
        file_path = Path("/repo/readme.md")
        content = "# README"
        file_id = "test-repo:readme.md"
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.NEO4J
        )
        
        result = await self.service._process_file_for_neo4j(
            file_path, content, file_id, request
        )
        
        # Should return True (success) but skip processing
        assert result is True
    
    @pytest.mark.asyncio
    async def test_process_single_file_success(self):
        """Test successful processing of a single file."""
        file_path = Path("/repo/test.py")
        repo_path = Path("/repo")
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.BOTH,
            chunk_size=1000
        )
        
        # Mock file reading and processing
        mock_open_func = Mock()
        mock_open_func.return_value.read.return_value = "def test(): pass"
        
        with patch('builtins.open', mock_open_func), \
             patch.object(self.service, '_process_file_for_rag', return_value=True), \
             patch.object(self.service, '_process_file_for_neo4j', return_value=True):
            
            result = await self.service._process_single_file(file_path, request, repo_path)
            
            assert result is not None
            assert result.file_id == "test-repo:test.py"
            assert result.processed_for_rag is True
            assert result.processed_for_kg is True
            assert result.language == "python"
    
    @pytest.mark.asyncio
    async def test_process_single_file_error(self):
        """Test single file processing with error."""
        file_path = Path("/repo/test.py")
        repo_path = Path("/repo")
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.BOTH
        )
        
        # Mock file reading error
        with patch('builtins.open', side_effect=Exception("File read error")):
            result = await self.service._process_single_file(file_path, request, repo_path)
            
            assert result is not None
            assert len(result.errors) > 0
            assert "File read error" in result.errors[0]
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test service cleanup."""
        # Mock executor
        mock_executor = Mock()
        self.service.executor = mock_executor
        
        await self.service.cleanup()
        
        mock_executor.shutdown.assert_called_once_with(wait=True)


class TestUnifiedIndexingIntegration:
    """Integration tests for unified indexing."""
    
    @pytest.mark.asyncio
    async def test_process_repository_unified_success(self):
        """Test successful unified repository processing."""
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.BOTH,
            file_types=[".py"],
            max_files=5
        )
        
        mock_qdrant_client = Mock()
        mock_neo4j_parser = Mock()
        
        with patch('src.services.unified_indexing_service.tempfile.mkdtemp', 
                   return_value="/tmp/test_repo"), \
             patch('pathlib.Path') as mock_path_class:
            
            # Setup mock repository path and files
            mock_repo_path = Mock()
            mock_path_class.return_value = mock_repo_path
            
            # Mock discovered files  
            mock_files = []
            for i in range(3):
                mock_file = Mock()
                mock_file.is_file.return_value = True
                mock_file.suffix = ".py"  # String suffix, not Mock
                mock_files.append(mock_file)
            mock_repo_path.rglob.return_value = mock_files
            
            service = UnifiedIndexingService(mock_qdrant_client, mock_neo4j_parser)
            
            # Mock the GitHub processor
            service.github_processor.clone_repository_temp = Mock(return_value={
                "success": True,
                "temp_directory": "/tmp/test_repo"
            })
            
            # Mock file discovery to bypass asyncio.to_thread
            with patch.object(service, '_discover_repository_files') as mock_discover:
                mock_discover.return_value = mock_files
                
                # Mock file processing
                with patch.object(service, '_process_single_file') as mock_process:
                    mock_process.return_value = FileProcessingResult(
                        file_id="test-repo:test.py",
                        file_path="/tmp/test_repo/test.py",
                        relative_path="test.py",
                        language="python",
                        file_type=".py"
                    )
                    
                    response = await service.process_repository_unified(request)
                    
                    assert response.success is True
                    assert response.repo_name == "test-repo"
                    assert response.files_processed > 0
    
    @pytest.mark.asyncio
    async def test_process_repository_unified_no_files(self):
        """Test unified processing when no files are found."""
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.BOTH,
            file_types=[".nonexistent"]
        )
        
        with patch('src.services.unified_indexing_service.tempfile.mkdtemp', 
                   return_value="/tmp/test_repo"), \
             patch('pathlib.Path') as mock_path_class:
            
            mock_repo_path = Mock()
            mock_path_class.return_value = mock_repo_path
            mock_repo_path.rglob.return_value = []  # No files found
            
            service = UnifiedIndexingService()
            service.github_processor.clone_repository_temp = Mock(return_value={
                "success": True,
                "temp_directory": "/tmp/test_repo"
            })
            
            response = await service.process_repository_unified(request)
            
            assert response.success is False
            assert len(response.errors) > 0
            assert "No files found" in response.errors[0]
    
    @pytest.mark.asyncio
    async def test_module_level_convenience_function(self):
        """Test the module-level convenience function."""
        with patch('src.services.unified_indexing_service.UnifiedIndexingService') as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            
            mock_response = Mock()
            mock_service.process_repository_unified = AsyncMock(return_value=mock_response)
            mock_service.cleanup = AsyncMock()
            
            result = await process_repository_unified(
                repo_url="https://github.com/test/repo",
                destination=IndexingDestination.BOTH
            )
            
            assert result == mock_response
            mock_service.process_repository_unified.assert_called_once()
            mock_service.cleanup.assert_called_once()


class TestFileIdGeneration:
    """Test file ID generation functionality."""
    
    def test_generate_file_id_standard_url(self):
        """Test file ID generation for standard GitHub URL."""
        file_id = generate_file_id(
            "https://github.com/user/repo",
            "src/main.py"
        )
        assert file_id == "user-repo:src/main.py"
    
    def test_generate_file_id_with_git_suffix(self):
        """Test file ID generation for URL with .git suffix."""
        file_id = generate_file_id(
            "https://github.com/user/repo.git", 
            "docs/readme.md"
        )
        assert file_id == "user-repo:docs/readme.md"
    
    def test_extract_repo_name_standard(self):
        """Test repository name extraction from standard URL."""
        repo_name = extract_repo_name("https://github.com/user/repo")
        assert repo_name == "user-repo"
    
    def test_extract_repo_name_with_git(self):
        """Test repository name extraction from URL with .git suffix."""
        repo_name = extract_repo_name("https://github.com/user/repo.git")
        assert repo_name == "user-repo"
    
    def test_extract_repo_name_complex(self):
        """Test repository name extraction from complex URL."""
        repo_name = extract_repo_name("https://github.com/org-name/repo_name")
        assert repo_name == "org-name-repo_name"


class TestUnifiedIndexingModels:
    """Test the unified indexing data models."""
    
    def test_unified_indexing_request_creation(self):
        """Test creation of UnifiedIndexingRequest."""
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.BOTH,
            file_types=[".py", ".md"],
            max_files=100,
            chunk_size=2000,
            max_size_mb=1000
        )
        
        assert request.repo_url == "https://github.com/test/repo"
        assert request.destination == IndexingDestination.BOTH
        assert request.should_process_rag is True
        assert request.should_process_kg is True
        assert len(request.file_types) == 2
    
    def test_unified_indexing_request_qdrant_only(self):
        """Test UnifiedIndexingRequest for QDRANT-only processing."""
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.QDRANT
        )
        
        assert request.should_process_rag is True
        assert request.should_process_kg is False
    
    def test_unified_indexing_request_neo4j_only(self):
        """Test UnifiedIndexingRequest for NEO4J-only processing."""
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo", 
            destination=IndexingDestination.NEO4J
        )
        
        assert request.should_process_rag is False
        assert request.should_process_kg is True
    
    def test_file_processing_result_creation(self):
        """Test creation of FileProcessingResult."""
        result = FileProcessingResult(
            file_id="test-repo:main.py",
            file_path="/repo/main.py",
            relative_path="main.py",
            language="python",
            file_type=".py"
        )
        
        assert result.file_id == "test-repo:main.py"
        assert result.is_successful is True  # No errors by default
        assert result.processing_time_seconds == 0.0
        assert len(result.errors) == 0
    
    def test_file_processing_result_with_errors(self):
        """Test FileProcessingResult with errors."""
        result = FileProcessingResult(
            file_id="test-repo:main.py",
            file_path="/repo/main.py",
            relative_path="main.py",
            language="python", 
            file_type=".py"
        )
        result.errors.append("Test error")
        
        assert result.is_successful is False
    
    def test_unified_indexing_response_finalization(self):
        """Test UnifiedIndexingResponse finalization."""
        start_time = datetime.now()
        response = UnifiedIndexingResponse(
            success=False,  # Initially False
            repo_url="https://github.com/test/repo",
            repo_name="test-repo",
            destination="both",
            files_processed=0,
            start_time=start_time
        )
        
        # Add some successful file results
        for i in range(3):
            result = FileProcessingResult(
                file_id=f"test-repo:file{i}.py",
                file_path=f"/repo/file{i}.py",
                relative_path=f"file{i}.py",
                language="python",
                file_type=".py"
            )
            response.add_file_result(result)
        
        # Add small delay to ensure processing_time_seconds > 0
        import time
        time.sleep(0.01)
        
        response.finalize()
        
        assert response.success is True  # Should be True with successful results
        assert response.files_processed == 3
        assert response.success_rate == 100.0  # 100% success rate
        assert response.processing_time_seconds > 0


# Mock helper function
def mock_open_func(file_path, mode='r', encoding=None, errors=None):
    """Mock function for opening files."""
    mock_file = Mock()
    if 'test.py' in str(file_path):
        mock_file.read.return_value = "def test(): pass"
    else:
        mock_file.read.return_value = "# Test content"
    return mock_file