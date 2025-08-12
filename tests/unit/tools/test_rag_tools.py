"""
Tests for RAG tools.
"""

import pytest
import json
from unittest.mock import Mock, patch
from src.tools.rag_tools import (
    get_available_sources,
    perform_rag_query,
    search_code_examples,
)


class TestRAGTools:
    """Test cases for RAG tools."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock MCP context."""
        context = Mock()
        context.request_context = Mock()
        context.request_context.lifespan_context = Mock()
        context.request_context.lifespan_context.qdrant_client = Mock()
        return context

    @pytest.mark.asyncio
    async def test_get_available_sources_success(self, mock_context):
        """Test get_available_sources with successful execution."""
        # Mock the Qdrant client response
        mock_sources = [
            {
                "source_id": "github.com/user/repo1",
                "summary": "Repository 1",
                "total_words": 1000,
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-01T00:00:00Z",
            },
            {
                "source_id": "github.com/user/repo2",
                "summary": "Repository 2",
                "total_words": 2000,
                "created_at": "2023-01-02T00:00:00Z",
                "updated_at": "2023-01-02T00:00:00Z",
            },
        ]
        mock_context.request_context.lifespan_context.qdrant_client.get_available_sources.return_value = mock_sources

        result = await get_available_sources(mock_context)
        result_data = json.loads(result)

        assert result_data["success"] is True
        assert result_data["total_sources"] == 2
        assert len(result_data["sources"]) == 2
        assert result_data["sources"][0]["source_id"] == "github.com/user/repo1"

    @pytest.mark.asyncio
    async def test_get_available_sources_exception(self, mock_context):
        """Test get_available_sources with exception handling."""
        mock_context.request_context.lifespan_context.qdrant_client.get_available_sources.side_effect = Exception(
            "Database error"
        )

        result = await get_available_sources(mock_context)
        result_data = json.loads(result)

        assert result_data["success"] is False
        assert "Database error" in result_data["error"]

    @pytest.mark.asyncio
    async def test_perform_rag_query_success(self, mock_context):
        """Test perform_rag_query with successful execution."""
        # Mock the RagService and its response
        mock_results = [
            {
                "id": "doc1",
                "content": "Document 1 content",
                "similarity": 0.95,
                "source": "github.com/user/repo",
            },
            {
                "id": "doc2",
                "content": "Document 2 content",
                "similarity": 0.85,
                "source": "github.com/user/repo",
            },
        ]

        with patch("src.services.rag_service.RagService") as mock_rag_service_class:
            mock_rag_service = Mock()
            mock_rag_service.search_with_reranking.return_value = mock_results
            mock_rag_service_class.return_value = mock_rag_service

            result = await perform_rag_query(
                mock_context,
                query="test query",
                source="github.com/user/repo",
                match_count=5,
            )
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["query"] == "test query"
            assert result_data["match_count"] == 2
            assert len(result_data["results"]) == 2

    @pytest.mark.asyncio
    async def test_perform_rag_query_with_file_id(self, mock_context):
        """Test perform_rag_query with file_id filtering."""
        mock_results = [
            {
                "id": "doc1",
                "content": "Document 1 content",
                "similarity": 0.95,
                "file_id": "repo:path/file.md",
            }
        ]

        with patch("src.services.rag_service.RagService") as mock_rag_service_class:
            mock_rag_service = Mock()
            mock_rag_service.search_with_reranking.return_value = mock_results
            mock_rag_service_class.return_value = mock_rag_service

            result = await perform_rag_query(
                mock_context,
                query="test query",
                file_id="repo:path/file.md",
                match_count=5,
            )
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["filters_applied"]["file_id"] == "repo:path/file.md"

    @pytest.mark.asyncio
    async def test_perform_rag_query_exception(self, mock_context):
        """Test perform_rag_query with exception handling."""
        with patch("src.services.rag_service.RagService") as mock_rag_service_class:
            mock_rag_service_class.side_effect = Exception("Search error")

            result = await perform_rag_query(
                mock_context, query="test query", match_count=5
            )
            result_data = json.loads(result)

            assert result_data["success"] is False
            assert "Search error" in result_data["error"]
            assert result_data["query"] == "test query"

    @pytest.mark.asyncio
    async def test_search_code_examples_success(self, mock_context):
        """Test search_code_examples with successful execution."""
        # Mock the RagService and its response
        mock_results = [
            {
                "id": "code1",
                "content": "def example():\n    return 'code'",
                "summary": "Example function",
                "similarity": 0.92,
                "source": "github.com/user/repo",
            },
            {
                "id": "code2",
                "content": "class Example:\n    pass",
                "summary": "Example class",
                "similarity": 0.82,
                "source": "github.com/user/repo",
            },
        ]

        with patch("src.services.rag_service.RagService") as mock_rag_service_class:
            mock_rag_service = Mock()
            mock_rag_service.search_with_reranking.return_value = mock_results
            mock_rag_service_class.return_value = mock_rag_service

            result = await search_code_examples(
                mock_context,
                query="example code",
                source_id="github.com/user/repo",
                match_count=5,
            )
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["query"] == "example code"
            assert result_data["match_count"] == 2
            assert len(result_data["results"]) == 2
            assert result_data["results"][0]["summary"] == "Example function"

    @pytest.mark.asyncio
    async def test_search_code_examples_with_file_id(self, mock_context):
        """Test search_code_examples with file_id filtering."""
        mock_results = [
            {
                "id": "code1",
                "content": "def example():\n    return 'code'",
                "summary": "Example function",
                "similarity": 0.92,
                "file_id": "repo:path/file.py",
            }
        ]

        with patch("src.services.rag_service.RagService") as mock_rag_service_class:
            mock_rag_service = Mock()
            mock_rag_service.search_with_reranking.return_value = mock_results
            mock_rag_service_class.return_value = mock_rag_service

            result = await search_code_examples(
                mock_context,
                query="example code",
                file_id="repo:path/file.py",
                match_count=5,
            )
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["filters_applied"]["file_id"] == "repo:path/file.py"

    @pytest.mark.asyncio
    async def test_search_code_examples_exception(self, mock_context):
        """Test search_code_examples with exception handling."""
        with patch("src.services.rag_service.RagService") as mock_rag_service_class:
            mock_rag_service_class.side_effect = Exception("Code search error")

            result = await search_code_examples(
                mock_context, query="example code", match_count=5
            )
            result_data = json.loads(result)

            assert result_data["success"] is False
            assert "Code search error" in result_data["error"]
            assert result_data["query"] == "example code"
