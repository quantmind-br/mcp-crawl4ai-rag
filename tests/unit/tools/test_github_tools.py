"""
Tests for GitHub tools.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from src.tools.github_tools import (
    index_github_repository,
)
from src.utils.chunking import smart_chunk_markdown


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
                    "jsx_components_extracted": 15,
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
                    ),
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
    async def test_index_github_repository_with_intelligent_routing_default(
        self, mock_context
    ):
        """Test index repository with default intelligent routing parameters."""
        with patch(
            "src.tools.github_tools.validate_github_url", return_value=(True, None)
        ):
            with patch(
                "src.services.unified_indexing_service.UnifiedIndexingService"
            ) as mock_service_class:
                mock_service = AsyncMock()
                mock_service_class.return_value = mock_service

                # Mock response
                mock_response = Mock()
                mock_response.success = True
                mock_response.repo_url = "https://github.com/user/repo"
                mock_response.repo_name = "repo"
                mock_response.destination = "both"
                mock_response.files_processed = 3
                mock_response.success_rate = 100.0
                mock_response.processing_time_seconds = 5.0
                mock_response.qdrant_documents = 10
                mock_response.neo4j_nodes = 8
                mock_response.cross_system_links_created = 5
                mock_response.performance_summary = {
                    "processing_speed": "0.6 files/sec"
                }
                mock_response.error_summary = {}
                mock_response.file_results = []

                mock_service.process_repository_unified.return_value = mock_response

                result = await index_github_repository(
                    mock_context,
                    "https://github.com/user/repo",
                    destination="both",
                    # Using default intelligent routing parameters
                )
                result_data = json.loads(result)

                # Verify the request was called with a proper routing config
                call_args = mock_service.process_repository_unified.call_args[0][0]
                assert hasattr(call_args, "routing_config")
                assert call_args.routing_config.enable_intelligent_routing is True
                assert call_args.routing_config.force_rag_patterns == []
                assert call_args.routing_config.force_kg_patterns == []

                # Verify response includes intelligent routing parameters
                assert result_data["success"] is True
                assert "intelligent_routing" in result_data["request_parameters"]
                assert (
                    result_data["request_parameters"]["intelligent_routing"]["enabled"]
                    is True
                )
                assert (
                    result_data["request_parameters"]["intelligent_routing"][
                        "force_rag_patterns"
                    ]
                    == []
                )
                assert (
                    result_data["request_parameters"]["intelligent_routing"][
                        "force_kg_patterns"
                    ]
                    == []
                )

    @pytest.mark.asyncio
    async def test_index_github_repository_with_custom_intelligent_routing(
        self, mock_context
    ):
        """Test index repository with custom intelligent routing parameters."""
        with patch(
            "src.tools.github_tools.validate_github_url", return_value=(True, None)
        ):
            with patch(
                "src.services.unified_indexing_service.UnifiedIndexingService"
            ) as mock_service_class:
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
                mock_response.processing_time_seconds = 8.0
                mock_response.qdrant_documents = 15
                mock_response.neo4j_nodes = 12
                mock_response.cross_system_links_created = 8
                mock_response.performance_summary = {
                    "processing_speed": "0.625 files/sec"
                }
                mock_response.error_summary = {}
                mock_response.file_results = []

                mock_service.process_repository_unified.return_value = mock_response

                # Test with custom intelligent routing parameters
                result = await index_github_repository(
                    mock_context,
                    "https://github.com/user/repo",
                    destination="both",
                    enable_intelligent_routing=True,
                    force_rag_patterns=[".*README.*", ".*docs/.*"],
                    force_kg_patterns=[".*test.*", ".*spec.*"],
                )
                result_data = json.loads(result)

                # Verify the request was called with custom routing config
                call_args = mock_service.process_repository_unified.call_args[0][0]
                assert call_args.routing_config.enable_intelligent_routing is True
                assert call_args.routing_config.force_rag_patterns == [
                    ".*README.*",
                    ".*docs/.*",
                ]
                assert call_args.routing_config.force_kg_patterns == [
                    ".*test.*",
                    ".*spec.*",
                ]

                # Verify response includes custom intelligent routing parameters
                assert result_data["success"] is True
                routing_params = result_data["request_parameters"][
                    "intelligent_routing"
                ]
                assert routing_params["enabled"] is True
                assert routing_params["force_rag_patterns"] == [
                    ".*README.*",
                    ".*docs/.*",
                ]
                assert routing_params["force_kg_patterns"] == [".*test.*", ".*spec.*"]

    @pytest.mark.asyncio
    async def test_index_github_repository_with_disabled_intelligent_routing(
        self, mock_context
    ):
        """Test index repository with intelligent routing disabled."""
        with patch(
            "src.tools.github_tools.validate_github_url", return_value=(True, None)
        ):
            with patch(
                "src.services.unified_indexing_service.UnifiedIndexingService"
            ) as mock_service_class:
                mock_service = AsyncMock()
                mock_service_class.return_value = mock_service

                # Mock response
                mock_response = Mock()
                mock_response.success = True
                mock_response.repo_url = "https://github.com/user/repo"
                mock_response.repo_name = "repo"
                mock_response.destination = "both"
                mock_response.files_processed = 4
                mock_response.success_rate = 100.0
                mock_response.processing_time_seconds = (
                    12.0  # Slower without intelligent routing
                )
                mock_response.qdrant_documents = 20
                mock_response.neo4j_nodes = (
                    20  # Both destinations processed for all files
                )
                mock_response.cross_system_links_created = 20
                mock_response.performance_summary = {
                    "processing_speed": "0.33 files/sec"
                }
                mock_response.error_summary = {}
                mock_response.file_results = []

                mock_service.process_repository_unified.return_value = mock_response

                result = await index_github_repository(
                    mock_context,
                    "https://github.com/user/repo",
                    destination="both",
                    enable_intelligent_routing=False,
                    force_rag_patterns=[
                        ".*README.*"
                    ],  # Should be ignored when disabled
                    force_kg_patterns=[".*test.*"],  # Should be ignored when disabled
                )
                result_data = json.loads(result)

                # Verify the request was called with disabled routing
                call_args = mock_service.process_repository_unified.call_args[0][0]
                assert call_args.routing_config.enable_intelligent_routing is False
                # Patterns should still be passed even if disabled (for potential future use)
                assert call_args.routing_config.force_rag_patterns == [".*README.*"]
                assert call_args.routing_config.force_kg_patterns == [".*test.*"]

                # Verify response reflects disabled state
                assert result_data["success"] is True
                routing_params = result_data["request_parameters"][
                    "intelligent_routing"
                ]
                assert routing_params["enabled"] is False
                assert routing_params["force_rag_patterns"] == [".*README.*"]
                assert routing_params["force_kg_patterns"] == [".*test.*"]

    @pytest.mark.asyncio
    async def test_index_github_repository_intelligent_routing_with_kg_only_destination(
        self, mock_context
    ):
        """Test intelligent routing parameters with KG-only destination."""
        with patch(
            "src.tools.github_tools.validate_github_url", return_value=(True, None)
        ):
            with patch(
                "src.services.unified_indexing_service.UnifiedIndexingService"
            ) as mock_service_class:
                mock_service = AsyncMock()
                mock_service_class.return_value = mock_service

                # Mock response for KG-only processing
                mock_response = Mock()
                mock_response.success = True
                mock_response.repo_url = "https://github.com/user/code-repo"
                mock_response.repo_name = "code-repo"
                mock_response.destination = "neo4j"
                mock_response.files_processed = 8
                mock_response.success_rate = 100.0
                mock_response.processing_time_seconds = 6.0
                mock_response.qdrant_documents = 0  # No RAG processing
                mock_response.neo4j_nodes = 35  # Only KG processing
                mock_response.cross_system_links_created = 0
                mock_response.performance_summary = {
                    "processing_speed": "1.33 files/sec"
                }
                mock_response.error_summary = {}
                mock_response.file_results = []

                mock_service.process_repository_unified.return_value = mock_response

                result = await index_github_repository(
                    mock_context,
                    "https://github.com/user/code-repo",
                    destination="neo4j",  # KG only
                    enable_intelligent_routing=True,
                    force_kg_patterns=[
                        ".*\\.py$",
                        ".*\\.js$",
                    ],  # Force specific files to KG
                )
                result_data = json.loads(result)

                # Verify the request was properly configured
                call_args = mock_service.process_repository_unified.call_args[0][0]
                assert call_args.destination.value == "neo4j"
                assert call_args.routing_config.enable_intelligent_routing is True
                assert call_args.routing_config.force_kg_patterns == [
                    ".*\\.py$",
                    ".*\\.js$",
                ]

                # Verify response shows KG-only results
                assert result_data["success"] is True
                assert result_data["destination"] == "neo4j"
                assert result_data["storage_summary"]["qdrant_documents"] == 0
                assert result_data["storage_summary"]["neo4j_nodes"] == 35

    @pytest.mark.asyncio
    async def test_index_github_repository_intelligent_routing_with_complex_patterns(
        self, mock_context
    ):
        """Test intelligent routing with complex regex patterns."""
        with patch(
            "src.tools.github_tools.validate_github_url", return_value=(True, None)
        ):
            with patch(
                "src.services.unified_indexing_service.UnifiedIndexingService"
            ) as mock_service_class:
                mock_service = AsyncMock()
                mock_service_class.return_value = mock_service

                # Mock response
                mock_response = Mock()
                mock_response.success = True
                mock_response.repo_url = "https://github.com/user/complex-repo"
                mock_response.repo_name = "complex-repo"
                mock_response.destination = "both"
                mock_response.files_processed = 12
                mock_response.success_rate = 100.0
                mock_response.processing_time_seconds = 15.0
                mock_response.qdrant_documents = 25
                mock_response.neo4j_nodes = 18
                mock_response.cross_system_links_created = 8
                mock_response.performance_summary = {
                    "processing_speed": "0.8 files/sec"
                }
                mock_response.error_summary = {}
                mock_response.file_results = []

                mock_service.process_repository_unified.return_value = mock_response

                # Test with complex regex patterns
                complex_rag_patterns = [
                    ".*/(README|CHANGELOG|CONTRIBUTING)\\.(md|rst|txt)$",
                    ".*/docs/.*\\.(md|mdx)$",
                    ".*\\.config\\.(json|yaml|yml)$",
                ]
                complex_kg_patterns = [
                    ".*/(src|lib)/.*\\.(py|js|ts|tsx)$",
                    ".*test.*\\.(py|js|ts)$",
                    ".*/(spec|__tests__)/.*$",
                ]

                result = await index_github_repository(
                    mock_context,
                    "https://github.com/user/complex-repo",
                    destination="both",
                    enable_intelligent_routing=True,
                    force_rag_patterns=complex_rag_patterns,
                    force_kg_patterns=complex_kg_patterns,
                )
                result_data = json.loads(result)

                # Verify complex patterns were passed correctly
                call_args = mock_service.process_repository_unified.call_args[0][0]
                assert (
                    call_args.routing_config.force_rag_patterns == complex_rag_patterns
                )
                assert call_args.routing_config.force_kg_patterns == complex_kg_patterns

                # Verify response includes complex patterns
                assert result_data["success"] is True
                routing_params = result_data["request_parameters"][
                    "intelligent_routing"
                ]
                assert routing_params["force_rag_patterns"] == complex_rag_patterns
                assert routing_params["force_kg_patterns"] == complex_kg_patterns

    @pytest.mark.asyncio
    async def test_index_github_repository_intelligent_routing_backward_compatibility(
        self, mock_context
    ):
        """Test that intelligent routing maintains backward compatibility."""
        with patch(
            "src.tools.github_tools.validate_github_url", return_value=(True, None)
        ):
            with patch(
                "src.services.unified_indexing_service.UnifiedIndexingService"
            ) as mock_service_class:
                mock_service = AsyncMock()
                mock_service_class.return_value = mock_service

                # Mock response
                mock_response = Mock()
                mock_response.success = True
                mock_response.repo_url = "https://github.com/user/legacy-repo"
                mock_response.repo_name = "legacy-repo"
                mock_response.destination = "both"
                mock_response.files_processed = 6
                mock_response.success_rate = 100.0
                mock_response.processing_time_seconds = 10.0
                mock_response.qdrant_documents = 15
                mock_response.neo4j_nodes = 12
                mock_response.cross_system_links_created = 10
                mock_response.performance_summary = {
                    "processing_speed": "0.6 files/sec"
                }
                mock_response.error_summary = {}
                mock_response.file_results = []

                mock_service.process_repository_unified.return_value = mock_response

                # Test calling function without new parameters (backward compatibility)
                result = await index_github_repository(
                    mock_context,
                    "https://github.com/user/legacy-repo",
                    destination="both",
                    file_types=[".py", ".md"],
                    max_files=20,
                    chunk_size=2000,
                    max_size_mb=300,
                    # NOT providing intelligent routing parameters - should use defaults
                )
                result_data = json.loads(result)

                # Verify defaults were applied
                call_args = mock_service.process_repository_unified.call_args[0][0]
                assert (
                    call_args.routing_config.enable_intelligent_routing is True
                )  # Default
                assert call_args.routing_config.force_rag_patterns == []  # Default
                assert call_args.routing_config.force_kg_patterns == []  # Default

                # Verify response includes default intelligent routing
                assert result_data["success"] is True
                routing_params = result_data["request_parameters"][
                    "intelligent_routing"
                ]
                assert routing_params["enabled"] is True
                assert routing_params["force_rag_patterns"] == []
                assert routing_params["force_kg_patterns"] == []

                # Verify all original parameters still work
                assert result_data["request_parameters"]["destination"] == "both"
                assert result_data["request_parameters"]["file_types"] == [".py", ".md"]
                assert result_data["request_parameters"]["max_files"] == 20
                assert result_data["request_parameters"]["chunk_size"] == 2000
                assert result_data["request_parameters"]["max_size_mb"] == 300
