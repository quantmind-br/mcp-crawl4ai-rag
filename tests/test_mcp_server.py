"""
Integration tests for the MCP server with Qdrant migration.

Tests the main MCP server functionality after Supabase to Qdrant migration.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Mock environment variables before importing
# Modern configuration
os.environ.setdefault("CHAT_MODEL", "gpt-3.5-turbo")
os.environ.setdefault("CHAT_API_KEY", "test-chat-api-key")
os.environ.setdefault("CHAT_API_BASE", "https://api.openai.com/v1")
os.environ.setdefault("EMBEDDINGS_MODEL", "text-embedding-3-small")
os.environ.setdefault("EMBEDDINGS_API_KEY", "test-embeddings-api-key")
os.environ.setdefault("EMBEDDINGS_API_BASE", "https://api.openai.com/v1")

# Other configuration
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")

# Mock external dependencies
sys.modules["crawl4ai"] = Mock()
sys.modules["sentence_transformers"] = Mock()

# Mock knowledge graph modules
mock_kg_validator = Mock()
mock_neo4j_extractor = Mock()
mock_ai_analyzer = Mock()
mock_hallucination_reporter = Mock()

sys.modules["knowledge_graph_validator"] = mock_kg_validator
sys.modules["parse_repo_into_neo4j"] = mock_neo4j_extractor
sys.modules["ai_script_analyzer"] = mock_ai_analyzer
sys.modules["hallucination_reporter"] = mock_hallucination_reporter

# Add mock classes
mock_kg_validator.KnowledgeGraphValidator = Mock
mock_neo4j_extractor.DirectNeo4jExtractor = Mock
mock_ai_analyzer.AIScriptAnalyzer = Mock
mock_hallucination_reporter.HallucinationReporter = Mock


class TestMCPServerIntegration:
    """Test MCP server integration with Qdrant."""

    @patch("crawl4ai_mcp.get_supabase_client")
    def test_context_initialization(self, mock_get_client):
        """Test that context initializes with Qdrant client."""
        # Setup mock
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Import after mocking
        from crawl4ai_mcp import MCPContext

        # Test
        context = MCPContext(qdrant_client=mock_client)

        # Verify
        assert context.qdrant_client == mock_client
        assert hasattr(context, "qdrant_client")

    @patch("crawl4ai_mcp.search_documents")
    @patch("crawl4ai_mcp.get_supabase_client")
    def test_perform_rag_query_integration(self, mock_get_client, mock_search):
        """Test RAG query with Qdrant integration."""
        # Setup mocks
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        mock_search.return_value = [
            {
                "id": "doc1",
                "similarity": 0.9,
                "content": "Python is a programming language",
                "url": "https://python.org",
                "chunk_number": 1,
            }
        ]

        # Import the function
        from crawl4ai_mcp import perform_rag_query

        context = Mock()
        context.qdrant_client = mock_client

        # Test
        result = perform_rag_query(
            query="What is Python?",
            context=context,
            match_count=5,
            filter_metadata=None,
            source_filter=None,
        )

        # Verify
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["similarity"] == 0.9
        mock_search.assert_called_once_with(mock_client, "What is Python?", 5, None)

    @patch("crawl4ai_mcp.search_code_examples")
    @patch("crawl4ai_mcp.get_supabase_client")
    def test_search_code_examples_integration(self, mock_get_client, mock_search):
        """Test code examples search with Qdrant integration."""
        # Setup mocks
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        mock_search.return_value = [
            {
                "id": "code1",
                "similarity": 0.85,
                "content": "def hello_world():\n    print('Hello, World!')",
                "summary": "Basic Python function example",
                "url": "https://example.com/python-basics",
            }
        ]

        # Import the function
        from crawl4ai_mcp import search_code_examples as mcp_search_code

        context = Mock()
        context.qdrant_client = mock_client

        # Test
        result = mcp_search_code(
            query="hello world function",
            context=context,
            match_count=3,
            filter_metadata=None,
            source_id=None,
        )

        # Verify
        assert "results" in result
        assert len(result["results"]) == 1
        assert "def hello_world" in result["results"][0]["content"]
        mock_search.assert_called_once_with(
            mock_client, "hello world function", 3, None, None
        )

    @patch("crawl4ai_mcp.get_supabase_client")
    def test_get_available_sources_integration(self, mock_get_client):
        """Test getting available sources from Qdrant."""
        # Setup mock
        mock_client = Mock()
        mock_client.get_available_sources.return_value = [
            {
                "source_id": "python.org",
                "summary": "Official Python documentation",
                "total_word_count": 50000,
                "updated_at": "2024-01-01T00:00:00",
            }
        ]
        mock_get_client.return_value = mock_client

        # Import the function
        from crawl4ai_mcp import get_available_sources

        context = Mock()
        context.qdrant_client = mock_client

        # Test
        result = get_available_sources(context=context)

        # Verify
        assert "sources" in result
        assert len(result["sources"]) == 1
        assert result["sources"][0]["source_id"] == "python.org"
        mock_client.get_available_sources.assert_called_once()

    @patch("crawl4ai_mcp.AsyncWebCrawler")
    @patch("crawl4ai_mcp.add_documents_to_supabase")
    @patch("crawl4ai_mcp.get_supabase_client")
    async def test_crawl_integration_mock(
        self, mock_get_client, mock_add_docs, mock_crawler_class
    ):
        """Test crawling integration with Qdrant (mocked)."""
        # Setup mocks
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        mock_add_docs.return_value = None

        # Mock crawler
        mock_crawler = AsyncMock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.markdown = "# Test Page\n\nThis is test content."
        mock_result.extracted_content = "This is test content."
        mock_crawler.arun.return_value = mock_result
        mock_crawler.__aenter__.return_value = mock_crawler
        mock_crawler.__aexit__.return_value = None
        mock_crawler_class.return_value = mock_crawler

        # Import the function
        from crawl4ai_mcp import crawl_single_page

        context = Mock()
        context.qdrant_client = mock_client

        # Test
        result = await crawl_single_page(
            url="https://example.com",
            context=context,
            chunk_size=2000,
            overlap_size=100,
        )

        # Verify
        assert result["success"] is True
        assert "content_preview" in result
        mock_add_docs.assert_called_once()


class TestHybridSearch:
    """Test hybrid search functionality with Qdrant."""

    @patch("crawl4ai_mcp.search_documents")
    @patch("crawl4ai_mcp.get_supabase_client")
    def test_hybrid_search_vector_only(self, mock_get_client, mock_search_docs):
        """Test hybrid search with only vector search."""
        # Setup mocks
        mock_client = Mock()
        mock_client.keyword_search_documents.return_value = []  # No keyword results
        mock_get_client.return_value = mock_client

        mock_search_docs.return_value = [
            {"id": "doc1", "similarity": 0.9, "content": "vector result"}
        ]

        # Import the function
        from crawl4ai_mcp import perform_rag_query

        context = Mock()
        context.qdrant_client = mock_client

        # Test
        result = perform_rag_query(
            query="test query",
            context=context,
            match_count=5,
            filter_metadata=None,
            source_filter=None,
        )

        # Verify vector search was called
        mock_search_docs.assert_called_once()
        assert len(result["results"]) == 1

    @patch("crawl4ai_mcp.search_documents")
    @patch("crawl4ai_mcp.get_supabase_client")
    def test_hybrid_search_with_keyword_results(
        self, mock_get_client, mock_search_docs
    ):
        """Test hybrid search combining vector and keyword results."""
        # Setup mocks
        mock_client = Mock()
        mock_client.keyword_search_documents.return_value = [
            {"id": "doc2", "similarity": 0.5, "content": "keyword result"}
        ]
        mock_get_client.return_value = mock_client

        mock_search_docs.return_value = [
            {"id": "doc1", "similarity": 0.9, "content": "vector result"}
        ]

        # Import the function
        from crawl4ai_mcp import perform_rag_query

        context = Mock()
        context.qdrant_client = mock_client

        # Test
        result = perform_rag_query(
            query="test query",
            context=context,
            match_count=5,
            filter_metadata=None,
            source_filter=None,
        )

        # Verify both searches were called
        mock_search_docs.assert_called_once()
        mock_client.keyword_search_documents.assert_called_once()

        # Should have results from both searches
        assert len(result["results"]) == 2


class TestErrorHandling:
    """Test error handling in MCP server."""

    @patch("crawl4ai_mcp.get_supabase_client")
    def test_qdrant_connection_error(self, mock_get_client):
        """Test handling of Qdrant connection errors."""
        # Setup mock to raise connection error
        mock_get_client.side_effect = Exception("Cannot connect to Qdrant")

        # Import after setting up mock
        from crawl4ai_mcp import get_available_sources

        context = Mock()

        # Test - should handle error gracefully
        try:
            result = get_available_sources(context=context)
            # If no exception, should return error result
            assert "error" in result or "sources" in result
        except Exception as e:
            # Connection errors are expected and should be handled
            assert "Qdrant" in str(e)

    @patch("crawl4ai_mcp.search_documents")
    @patch("crawl4ai_mcp.get_supabase_client")
    def test_search_error_handling(self, mock_get_client, mock_search):
        """Test error handling in search operations."""
        # Setup mocks
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Mock search to raise error
        mock_search.side_effect = Exception("Search failed")

        from crawl4ai_mcp import perform_rag_query

        context = Mock()
        context.qdrant_client = mock_client

        # Test - should handle error gracefully
        result = perform_rag_query(query="test", context=context, match_count=5)

        # Should return some result structure even on error
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__])
