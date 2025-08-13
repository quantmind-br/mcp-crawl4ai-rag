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

    @pytest.mark.asyncio
    async def test_smart_crawl_github_mdx_files(self, mock_context):
        """Test smart crawl with MDX files."""
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
                        with patch(
                            "src.tools.github_tools.MDXProcessor"
                        ) as mock_mdx_processor_class:
                            # Mock repo manager
                            mock_repo_manager = Mock()
                            mock_repo_manager.clone_repository.return_value = (
                                "/tmp/test-repo"
                            )
                            mock_repo_manager._get_directory_size_mb.return_value = 1.5
                            mock_repo_manager_class.return_value = mock_repo_manager

                            # Mock file discovery with MDX files
                            mock_file_discovery = Mock()
                            mock_file_discovery.discover_files.return_value = [
                                {
                                    "path": "/tmp/test-repo/docs/guide.mdx",
                                    "relative_path": "docs/guide.mdx",
                                    "file_type": ".mdx",
                                    "filename": "guide.mdx",
                                    "size_bytes": 2048,
                                    "is_readme": False,
                                }
                            ]
                            mock_file_discovery_class.return_value = mock_file_discovery

                            # Mock metadata extractor
                            mock_metadata_extractor = Mock()
                            mock_repo_info = Mock()
                            mock_repo_info.owner = "user"
                            mock_repo_info.repo_name = "mdx-repo"
                            mock_repo_info.description = "MDX documentation repository"
                            mock_metadata_extractor.extract_repo_metadata.return_value = (
                                mock_repo_info
                            )
                            mock_metadata_extractor.create_metadata_dict.return_value = {
                                "owner": "user",
                                "repo_name": "mdx-repo",
                                "description": "MDX documentation repository",
                            }
                            mock_metadata_extractor_class.return_value = (
                                mock_metadata_extractor
                            )

                            # Mock MDX processor
                            mock_mdx_processor = Mock()
                            mock_processed_content = Mock()
                            mock_processed_content.content = "# Guide\n\nThis is processed MDX content"
                            mock_processed_content.content_type = "mdx"
                            mock_processed_content.name = "guide.mdx"
                            mock_processed_content.signature = None
                            mock_processed_content.line_number = 1
                            mock_processed_content.language = "mdx"
                            
                            # Mock JSX component
                            mock_jsx_component = Mock()
                            mock_jsx_component.content = "JSX Component: Alert"
                            mock_jsx_component.content_type = "jsx_component"
                            mock_jsx_component.name = "Alert"
                            mock_jsx_component.signature = "<Alert type='info'>"
                            mock_jsx_component.line_number = 5
                            mock_jsx_component.language = "jsx"
                            
                            mock_mdx_processor.process_file.return_value = [
                                mock_processed_content,
                                mock_jsx_component
                            ]
                            mock_mdx_processor_class.return_value = mock_mdx_processor

                            # Mock storage functions
                            with patch(
                                "src.tools.github_tools.extract_source_summary",
                                return_value="MDX documentation summary",
                            ):
                                with patch(
                                    "src.tools.github_tools.update_source_info"
                                ):
                                    with patch(
                                        "src.tools.github_tools.add_documents_to_vector_db"
                                    ):
                                        result = await smart_crawl_github(
                                            mock_context,
                                            "https://github.com/user/mdx-repo",
                                            max_files=10,
                                            file_types_to_index=[".mdx"],
                                        )

                                        result_data = json.loads(result)

                                        assert result_data["success"] is True
                                        assert result_data["repo_url"] == "https://github.com/user/mdx-repo"
                                        assert result_data["files_discovered"] == 1
                                        assert ".mdx" in result_data["file_types_requested"]

    @pytest.mark.asyncio
    async def test_index_github_repository_mdx_files(self, mock_context):
        """Test unified repository indexing with MDX files."""
        with patch(
            "src.tools.github_tools.validate_github_url", return_value=(True, None)
        ):
            with patch(
                "src.services.unified_indexing_service.UnifiedIndexingService"
            ) as mock_service_class:
                mock_service = AsyncMock()
                mock_service_class.return_value = mock_service

                # Mock response with MDX processing results
                mock_response = Mock()
                mock_response.success = True
                mock_response.repo_url = "https://github.com/user/mdx-docs"
                mock_response.repo_name = "mdx-docs"
                mock_response.destination = "both"
                mock_response.files_processed = 3
                mock_response.success_rate = 100.0
                mock_response.processing_time_seconds = 5.2
                mock_response.qdrant_documents = 8
                mock_response.neo4j_nodes = 12
                mock_response.cross_system_links_created = 8
                mock_response.performance_summary = {
                    "processing_speed": "1.8 files/sec",
                    "jsx_components_extracted": 15
                }
                mock_response.error_summary = {}
                mock_response.file_results = [
                    Mock(
                        file_id="mdx-id-1",
                        relative_path="docs/intro.mdx",
                        language="mdx",
                        file_type=".mdx",
                        processed_for_rag=True,
                        processed_for_kg=True,
                        rag_chunks=4,
                        kg_entities=5,
                        processing_time_seconds=1.8,
                        processing_summary="MDX processed with 3 JSX components",
                        errors=[],
                        is_successful=True,
                    ),
                    Mock(
                        file_id="mdx-id-2",
                        relative_path="docs/guide.mdx",
                        language="mdx",
                        file_type=".mdx",
                        processed_for_rag=True,
                        processed_for_kg=True,
                        rag_chunks=6,
                        kg_entities=8,
                        processing_time_seconds=2.1,
                        processing_summary="MDX processed with 5 JSX components",
                        errors=[],
                        is_successful=True,
                    )
                ]

                mock_service.process_repository_unified.return_value = mock_response

                result = await index_github_repository(
                    mock_context,
                    "https://github.com/user/mdx-docs",
                    destination="both",
                    file_types=[".mdx"],
                    max_files=10,
                )
                result_data = json.loads(result)

                assert result_data["success"] is True
                assert result_data["repo_url"] == "https://github.com/user/mdx-docs"
                assert result_data["processing_summary"]["files_processed"] == 3
                assert result_data["storage_summary"]["qdrant_documents"] == 8
                assert result_data["storage_summary"]["neo4j_nodes"] == 12
                assert result_data["languages_detected"] == ["mdx"]

    @pytest.mark.asyncio 
    async def test_smart_crawl_github_mixed_file_types_with_mdx(self, mock_context):
        """Test smart crawl with mixed file types including MDX."""
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
                        mock_repo_manager._get_directory_size_mb.return_value = 2.1
                        mock_repo_manager_class.return_value = mock_repo_manager

                        # Mock file discovery with mixed file types
                        mock_file_discovery = Mock()
                        mock_file_discovery.discover_files.return_value = [
                            {
                                "path": "/tmp/test-repo/README.md",
                                "relative_path": "README.md",
                                "file_type": ".md",
                                "filename": "README.md",
                                "size_bytes": 1024,
                                "is_readme": True,
                            },
                            {
                                "path": "/tmp/test-repo/docs/guide.mdx",
                                "relative_path": "docs/guide.mdx",
                                "file_type": ".mdx",
                                "filename": "guide.mdx",
                                "size_bytes": 2048,
                                "is_readme": False,
                            }
                        ]
                        mock_file_discovery_class.return_value = mock_file_discovery

                        # Mock metadata extractor
                        mock_metadata_extractor = Mock()
                        mock_repo_info = Mock()
                        mock_repo_info.owner = "user"
                        mock_repo_info.repo_name = "mixed-repo"
                        mock_repo_info.description = "Repository with mixed documentation"
                        mock_metadata_extractor.extract_repo_metadata.return_value = (
                            mock_repo_info
                        )
                        mock_metadata_extractor.create_metadata_dict.return_value = {
                            "owner": "user",
                            "repo_name": "mixed-repo",
                            "description": "Repository with mixed documentation",
                        }
                        mock_metadata_extractor_class.return_value = (
                            mock_metadata_extractor
                        )

                        # Mock both Markdown and MDX processors for mixed file types
                        with patch(
                            "src.tools.github_tools.MarkdownProcessor"
                        ) as mock_md_processor_class:
                            with patch(
                                "src.tools.github_tools.MDXProcessor"
                            ) as mock_mdx_processor_class:
                                # Mock markdown processor
                                mock_md_processor = Mock()
                                mock_md_content = Mock()
                                mock_md_content.content = "# README\n\nMarkdown content"
                                mock_md_content.content_type = "document"
                                mock_md_content.name = "README"
                                mock_md_content.language = "markdown"
                                mock_md_content.signature = None
                                mock_md_content.line_number = 1
                                mock_md_processor.process_file.return_value = [mock_md_content]
                                mock_md_processor_class.return_value = mock_md_processor

                                # Mock MDX processor  
                                mock_mdx_processor = Mock()
                                mock_mdx_content = Mock()
                                mock_mdx_content.content = "# Guide\n\nMDX content"
                                mock_mdx_content.content_type = "mdx"
                                mock_mdx_content.name = "guide.mdx"
                                mock_mdx_content.language = "mdx"
                                mock_mdx_content.signature = None
                                mock_mdx_content.line_number = 1
                                mock_mdx_processor.process_file.return_value = [mock_mdx_content]
                                mock_mdx_processor_class.return_value = mock_mdx_processor

                                # Mock storage functions
                                with patch(
                                    "src.tools.github_tools.extract_source_summary",
                                    return_value="Mixed documentation summary",
                                ):
                                    with patch(
                                        "src.tools.github_tools.update_source_info"
                                    ):
                                        with patch(
                                            "src.tools.github_tools.add_documents_to_vector_db"
                                        ):
                                            result = await smart_crawl_github(
                                                mock_context,
                                                "https://github.com/user/mixed-repo", 
                                                max_files=10,
                                                file_types_to_index=[".md", ".mdx"],
                                            )

                                            result_data = json.loads(result)

                                            assert result_data["success"] is True
                                            assert result_data["files_discovered"] == 2
                                            assert ".md" in result_data["file_types_requested"]
                                            assert ".mdx" in result_data["file_types_requested"]
