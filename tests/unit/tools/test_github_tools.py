"""
Tests for GitHub tools.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from src.tools.github_tools import (
    smart_crawl_github,
    index_github_repository,
    smart_chunk_markdown,
)


class TestGitHubTools:
    """Test cases for GitHub tools."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock MCP context."""
        context = Mock()
        context.request_context = Mock()
        context.request_context.lifespan_context = Mock()
        context.request_context.lifespan_context.qdrant_client = Mock()
        return context

    def test_smart_chunk_markdown_small_text(self):
        """Test smart chunking with small text."""
        text = "This is a small text that should fit in one chunk."
        chunks = smart_chunk_markdown(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_smart_chunk_markdown_large_text(self):
        """Test smart chunking with large text."""
        text = "A" * 250  # 250 characters
        chunks = smart_chunk_markdown(text, chunk_size=100)
        assert len(chunks) == 3
        assert all(len(chunk) <= 100 for chunk in chunks)

    def test_smart_chunk_markdown_with_code_blocks(self):
        """Test smart chunking with code blocks."""
        text = """This is some text.
        
```python
def example():
    return "code block"
```

More text after code block."""

        chunks = smart_chunk_markdown(text, chunk_size=50)
        assert len(chunks) >= 1
        # Should try to break at code block boundaries

    def test_smart_chunk_markdown_empty_text(self):
        """Test smart chunking with empty text."""
        chunks = smart_chunk_markdown("", chunk_size=100)
        assert len(chunks) == 0

    def test_smart_chunk_markdown_whitespace_only(self):
        """Test smart chunking with whitespace only."""
        chunks = smart_chunk_markdown("   \n\n   ", chunk_size=100)
        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_smart_crawl_github_invalid_url(self, mock_context):
        """Test smart crawl with invalid GitHub URL."""
        result = await smart_crawl_github(mock_context, "https://invalid-url.com")
        result_data = json.loads(result)

        assert result_data["success"] is False
        assert "Invalid GitHub URL" in result_data["error"]

    @pytest.mark.asyncio
    async def test_smart_crawl_github_no_files_found(self, mock_context):
        """Test smart crawl when no files are found."""
        with patch(
            "src.tools.github_tools.validate_github_url", return_value=(True, None)
        ):
            with patch(
                "src.tools.github_tools.GitHubRepoManager"
            ) as mock_repo_manager_class:
                with patch(
                    "src.tools.github_tools.MultiFileDiscovery"
                ) as mock_file_discovery_class:
                    mock_repo_manager = Mock()
                    mock_repo_manager.clone_repository.return_value = "/tmp/test-repo"
                    mock_repo_manager_class.return_value = mock_repo_manager

                    mock_file_discovery = Mock()
                    mock_file_discovery.discover_files.return_value = []
                    mock_file_discovery_class.return_value = mock_file_discovery

                    with patch("src.tools.github_tools.GitHubMetadataExtractor"):
                        result = await smart_crawl_github(
                            mock_context,
                            "https://github.com/user/repo",
                            file_types_to_index=[".md"],
                        )
                        result_data = json.loads(result)

                        assert result_data["success"] is False
                        assert "No files found" in result_data["error"]

    @pytest.mark.asyncio
    async def test_smart_crawl_github_success(self, mock_context):
        """Test successful smart crawl."""
        with patch(
            "src.tools.github_tools.validate_github_url", return_value=(True, None)
        ):
            with patch(
                "src.tools.github_tools.GitHubRepoManager"
            ) as mock_repo_manager_class:
                with patch(
                    "src.tools.github_tools.MultiFileDiscovery"
                ) as mock_file_discovery_class:
                    with patch(
                        "src.tools.github_tools.GitHubMetadataExtractor"
                    ) as mock_metadata_extractor_class:
                        # Mock repo manager
                        mock_repo_manager = Mock()
                        mock_repo_manager.clone_repository.return_value = (
                            "/tmp/test-repo"
                        )
                        mock_repo_manager._get_directory_size_mb.return_value = 1.5
                        mock_repo_manager_class.return_value = mock_repo_manager

                        # Mock file discovery
                        mock_file_discovery = Mock()
                        mock_file_discovery.discover_files.return_value = [
                            {
                                "path": "/tmp/test-repo/README.md",
                                "relative_path": "README.md",
                                "file_type": ".md",
                                "filename": "README.md",
                                "size_bytes": 1024,
                                "is_readme": True,
                            }
                        ]
                        mock_file_discovery_class.return_value = mock_file_discovery

                        # Mock metadata extractor
                        mock_metadata_extractor = Mock()
                        mock_repo_info = Mock()
                        mock_repo_info.owner = "user"
                        mock_repo_info.repo_name = "repo"
                        mock_repo_info.description = "Test repository"

                        mock_metadata_extractor.extract_repo_metadata.return_value = (
                            mock_repo_info
                        )
                        mock_metadata_extractor.create_metadata_dict.return_value = {
                            "owner": "user",
                            "repo_name": "repo",
                            "description": "Test repository",
                        }
                        mock_metadata_extractor_class.return_value = (
                            mock_metadata_extractor
                        )

                        # Mock processors
                        with patch(
                            "src.tools.github_tools.MarkdownProcessor"
                        ) as mock_processor_class:
                            mock_processor = Mock()
                            # Mock ProcessedContent object
                            mock_processed_content = Mock()
                            mock_processed_content.content = "Test markdown content"
                            mock_processed_content.content_type = "document"
                            mock_processed_content.name = "README"
                            mock_processed_content.language = "markdown"
                            mock_processed_content.signature = None
                            mock_processed_content.line_number = 1

                            mock_processor.process_file.return_value = [
                                mock_processed_content
                            ]
                            mock_processor_class.return_value = mock_processor

                            # Mock storage functions
                            with patch(
                                "src.tools.github_tools.add_documents_to_vector_db"
                            ):
                                with patch("src.tools.github_tools.update_source_info"):
                                    result = await smart_crawl_github(
                                        mock_context,
                                        "https://github.com/user/repo",
                                        file_types_to_index=[".md"],
                                        max_files=10,
                                    )
                                    result_data = json.loads(result)

                                    assert result_data["success"] is True
                                    assert (
                                        result_data["repo_url"]
                                        == "https://github.com/user/repo"
                                    )
                                    assert result_data["files_discovered"] == 1

    @pytest.mark.asyncio
    async def test_index_github_repository_invalid_url(self, mock_context):
        """Test index repository with invalid GitHub URL."""
        result = await index_github_repository(mock_context, "https://invalid-url.com")
        result_data = json.loads(result)

        assert result_data["success"] is False
        assert "Invalid GitHub URL" in result_data["error"]

    @pytest.mark.asyncio
    async def test_index_github_repository_invalid_destination(self, mock_context):
        """Test index repository with invalid destination."""
        with patch(
            "src.tools.github_tools.validate_github_url", return_value=(True, None)
        ):
            result = await index_github_repository(
                mock_context, "https://github.com/user/repo", destination="invalid"
            )
            result_data = json.loads(result)

            assert result_data["success"] is False
            assert "Invalid destination" in result_data["error"]

    @pytest.mark.asyncio
    async def test_index_github_repository_success(self, mock_context):
        """Test successful repository indexing."""
        with patch(
            "src.tools.github_tools.validate_github_url", return_value=(True, None)
        ):
            # Mock the unified indexing service
            with patch(
                "src.services.unified_indexing_service.UnifiedIndexingService"
            ) as mock_service_class:
                # Mock service instance
                mock_service = AsyncMock()
                mock_service_class.return_value = mock_service

                # Mock response
                mock_response = Mock()
                mock_response.success = True
                mock_response.repo_url = "https://github.com/user/repo"
                mock_response.repo_name = "repo"
                mock_response.destination = "both"
                mock_response.files_processed = 5
                mock_response.success_rate = 100.0
                mock_response.processing_time_seconds = 10.5
                mock_response.qdrant_documents = 25
                mock_response.neo4j_nodes = 15
                mock_response.cross_system_links_created = 20
                mock_response.performance_summary = {
                    "processing_speed": "2.5 files/sec"
                }
                mock_response.error_summary = {}
                mock_response.file_results = [
                    Mock(
                        file_id="test-id-1",
                        relative_path="README.md",
                        language="markdown",
                        file_type=".md",
                        processed_for_rag=True,
                        processed_for_kg=False,
                        rag_chunks=3,
                        kg_entities=0,
                        processing_time_seconds=1.2,
                        processing_summary="Success",
                        errors=[],
                        is_successful=True,
                    )
                ]

                mock_service.process_repository_unified.return_value = mock_response

                result = await index_github_repository(
                    mock_context,
                    "https://github.com/user/repo",
                    destination="both",
                    file_types=[".md"],
                    max_files=10,
                )
                result_data = json.loads(result)

                assert result_data["success"] is True
                assert result_data["repo_url"] == "https://github.com/user/repo"
                assert result_data["processing_summary"]["files_processed"] == 5

    @pytest.mark.asyncio
    async def test_index_github_repository_import_error(self, mock_context):
        """Test repository indexing with import error."""
        with patch(
            "src.tools.github_tools.validate_github_url", return_value=(True, None)
        ):
            with patch(
                "src.services.unified_indexing_service.UnifiedIndexingService"
            ) as mock_service_class:
                mock_service_class.side_effect = ImportError("Module not found")

                result = await index_github_repository(
                    mock_context, "https://github.com/user/repo", destination="both"
                )
                result_data = json.loads(result)

                assert result_data["success"] is False
                assert "Unified indexing service not available" in result_data["error"]

    @pytest.mark.asyncio
    async def test_index_github_repository_exception(self, mock_context):
        """Test repository indexing with general exception."""
        with patch(
            "src.tools.github_tools.validate_github_url", return_value=(True, None)
        ):
            with patch(
                "src.services.unified_indexing_service.UnifiedIndexingService"
            ) as mock_service_class:
                mock_service_class.side_effect = Exception("Test error")

                result = await index_github_repository(
                    mock_context, "https://github.com/user/repo", destination="both"
                )
                result_data = json.loads(result)

                assert result_data["success"] is False
                assert result_data["error"] == "Test error"
