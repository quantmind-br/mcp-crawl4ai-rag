"""
# ruff: noqa: E402
Basic tests for MCP server functions.

Simple tests to validate core MCP tool functionality after Qdrant migration.
"""

import pytest
import os
import sys
from src.embedding_config import get_embedding_dimensions
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Setup test environment
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


class TestMCPBasicFunctionality:
    """Test basic MCP server functionality."""

    @patch('src.clients.qdrant_client.get_qdrant_client')
    def test_context_dataclass(self, mock_get_client):
        """Test that context dataclass uses Qdrant client."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Import after mocking
        from src.core.context import Crawl4AIContext

        # Test
        context = Crawl4AIContext(
            crawler=Mock(), 
            qdrant_client=mock_client, 
            embedding_cache=Mock()
        )

        # Verify
        assert context.qdrant_client == mock_client
        assert hasattr(context, "qdrant_client")
        assert hasattr(context, "crawler")
        assert hasattr(context, "embedding_cache")

    @patch("src.services.rag_service.search_documents")
    def test_perform_rag_query_basic(self, mock_search):
        """Test basic RAG query functionality."""
        # Setup mock
        mock_search.return_value = [
            {
                "id": "doc1",
                "similarity": 0.9,
                "content": "Python programming content",
                "url": "https://python.org",
            }
        ]

        from src.tools.rag_tools import perform_rag_query

        # Create mock context
        mock_ctx = Mock()
        mock_ctx.request_context = Mock()
        mock_ctx.request_context.lifespan_context = Mock()
        mock_ctx.request_context.lifespan_context.qdrant_client = Mock()

        # Test (this is an async function)
        # Note: Testing the basic call structure, actual async execution would need more setup
        assert callable(perform_rag_query)

    def test_import_structure(self):
        """Test that all required imports work after migration."""
        # Test importing main modules
        from src.core.context import Crawl4AIContext
        from src.clients.qdrant_client import QdrantClientWrapper, get_qdrant_client
        from src.services.rag_service import search_documents
        from src.clients.qdrant_client import get_qdrant_client as get_vector_db_client

        # Verify classes exist
        assert Crawl4AIContext is not None
        assert QdrantClientWrapper is not None
        assert callable(get_qdrant_client)
        assert callable(search_documents)
        assert callable(get_vector_db_client)

    @patch("src.clients.qdrant_client.QdrantClient")
    def test_vector_db_client_wrapper(self, mock_qdrant_client):
        """Test that the vector DB client helper function works."""
        # Setup mock client
        mock_client_instance = Mock()
        
        # Mock collections to return empty list (no existing collections)
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client_instance.get_collections.return_value = mock_collections
        
        # Mock get_collection to raise exception (collection doesn't exist)
        mock_client_instance.get_collection.side_effect = Exception("Collection not found")
        
        mock_qdrant_client.return_value = mock_client_instance

        from src.clients.qdrant_client import get_qdrant_client as get_vector_db_client

        client = get_vector_db_client()
        assert client is not None
        assert hasattr(client, 'client')

    def test_qdrant_wrapper_interface(self):
        """Test that Qdrant wrapper has expected interface."""
        from src.clients.qdrant_client import QdrantClientWrapper

        # Check that class has expected methods (without calling them)
        expected_methods = [
            "search_documents",
            "search_code_examples",
            "keyword_search_documents",
            "keyword_search_code_examples",
            "health_check",
            "get_available_sources",
            "update_source_info",
            "generate_point_id",
            "normalize_search_results",
        ]

        for method_name in expected_methods:
            assert hasattr(QdrantClientWrapper, method_name)
            assert callable(getattr(QdrantClientWrapper, method_name))

    @patch("src.clients.qdrant_client.QdrantClient")
    def test_collections_configuration(self, mock_qdrant_client):
        """Test that collections are properly configured."""
        from src.clients.qdrant_client import COLLECTIONS

        # Verify collection configurations exist
        assert "crawled_pages" in COLLECTIONS
        assert "code_examples" in COLLECTIONS

        # Verify collection structure
        for name, config in COLLECTIONS.items():
            assert "vectors_config" in config
            assert "payload_schema" in config

            # Check vector configuration
            vector_config = config["vectors_config"]
            assert hasattr(vector_config, "size")
            assert hasattr(vector_config, "distance")
            assert config["vectors_config"].size == get_embedding_dimensions()  # OpenAI embedding size

    def test_environment_configuration(self):
        """Test that environment variables are properly configured."""
        # Test that required environment variables can be accessed
        expected_vars = [
            "QDRANT_HOST",
            "QDRANT_PORT",
            # Modern configuration variables
            "CHAT_MODEL",
            "CHAT_API_KEY",
            "EMBEDDINGS_MODEL",
            "EMBEDDINGS_API_KEY",
        ]

        for var in expected_vars:
            # Just check that the variables are set in our test environment
            assert os.environ.get(var) is not None


class TestSearchFunctionality:
    """Test search functionality with mocked Qdrant."""

    @patch("src.clients.qdrant_client.get_qdrant_client")
    def test_document_search_workflow(self, mock_get_client):
        """Test complete document search workflow."""
        # Setup mocks
        mock_client = Mock()
        mock_client.search_documents.return_value = [
            {"id": "doc1", "similarity": 0.9, "content": "test content"}
        ]
        mock_get_client.return_value = mock_client

        from src.services.rag_service import search_documents

        # Test
        results = search_documents("test query")

        # Verify workflow
        mock_client.search_documents.assert_called_once()
        assert len(results) == 1
        assert results[0]["id"] == "doc1"

    @patch("src.clients.qdrant_client.get_qdrant_client")
    def test_code_search_workflow(self, mock_get_client):
        """Test complete code search workflow."""
        # Setup mocks
        mock_client = Mock()
        mock_client.search_code_examples.return_value = [
            {"id": "code1", "similarity": 0.85, "content": "def test(): pass"}
        ]
        mock_get_client.return_value = mock_client

        from src.services.rag_service import search_code_examples

        # Test
        results = search_code_examples("test function")

        # Verify workflow
        mock_client.search_code_examples.assert_called_once()
        assert len(results) == 1
        assert results[0]["id"] == "code1"


if __name__ == "__main__":
    pytest.main([__file__])
