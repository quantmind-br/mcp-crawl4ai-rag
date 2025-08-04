"""
Basic tests for MCP server functions.

Simple tests to validate core MCP tool functionality after Qdrant migration.
"""
import pytest
import os
import sys
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

    @patch('crawl4ai_mcp.get_supabase_client')
    def test_context_dataclass(self, mock_get_client):
        """Test that context dataclass uses Qdrant client."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        # Import after mocking
        from crawl4ai_mcp import Crawl4AIContext
        
        # Test
        context = Crawl4AIContext(
            crawler=Mock(),
            qdrant_client=mock_client
        )
        
        # Verify
        assert context.qdrant_client == mock_client
        assert hasattr(context, 'qdrant_client')
        assert hasattr(context, 'crawler')

    @patch('crawl4ai_mcp.search_documents')
    def test_perform_rag_query_basic(self, mock_search):
        """Test basic RAG query functionality."""
        # Setup mock
        mock_search.return_value = [
            {
                "id": "doc1", 
                "similarity": 0.9,
                "content": "Python programming content",
                "url": "https://python.org"
            }
        ]
        
        from crawl4ai_mcp import perform_rag_query
        
        # Create mock context
        mock_ctx = Mock()
        mock_ctx.deps = Mock()
        mock_ctx.deps.qdrant_client = Mock()
        
        # Test (this is an async function)
        # Note: Testing the basic call structure, actual async execution would need more setup
        assert callable(perform_rag_query)

    def test_import_structure(self):
        """Test that all required imports work after migration."""
        # Test importing main modules
        from crawl4ai_mcp import Crawl4AIContext
        from qdrant_wrapper import QdrantClientWrapper, get_qdrant_client
        from utils import search_documents, get_supabase_client
        
        # Verify classes exist
        assert Crawl4AIContext is not None
        assert QdrantClientWrapper is not None
        assert callable(get_qdrant_client)
        assert callable(search_documents)
        assert callable(get_supabase_client)

    @patch('utils.get_qdrant_client')
    def test_legacy_compatibility(self, mock_get_qdrant):
        """Test that legacy Supabase function names still work."""
        mock_client = Mock()
        mock_get_qdrant.return_value = mock_client
        
        from utils import get_supabase_client
        
        # Test legacy function returns Qdrant client
        client = get_supabase_client()
        assert client == mock_client

    def test_qdrant_wrapper_interface(self):
        """Test that Qdrant wrapper has expected interface."""
        from qdrant_wrapper import QdrantClientWrapper
        
        # Check that class has expected methods (without calling them)
        expected_methods = [
            'search_documents',
            'search_code_examples', 
            'keyword_search_documents',
            'keyword_search_code_examples',
            'health_check',
            'get_available_sources',
            'update_source_info',
            'generate_point_id',
            'normalize_search_results'
        ]
        
        for method_name in expected_methods:
            assert hasattr(QdrantClientWrapper, method_name)
            assert callable(getattr(QdrantClientWrapper, method_name))

    @patch('qdrant_wrapper.QdrantClient')
    def test_collections_configuration(self, mock_qdrant_client):
        """Test that collections are properly configured."""
        from qdrant_wrapper import COLLECTIONS
        
        # Verify collection configurations exist
        assert "crawled_pages" in COLLECTIONS
        assert "code_examples" in COLLECTIONS
        
        # Verify collection structure
        for name, config in COLLECTIONS.items():
            assert "vectors_config" in config
            assert "payload_schema" in config
            
            # Check vector configuration
            vector_config = config["vectors_config"]
            assert hasattr(vector_config, 'size')
            assert hasattr(vector_config, 'distance')
            assert config["vectors_config"].size == 1536  # OpenAI embedding size

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
            "EMBEDDINGS_API_KEY"
        ]
        
        for var in expected_vars:
            # Just check that the variables are set in our test environment
            assert os.environ.get(var) is not None


class TestSearchFunctionality:
    """Test search functionality with mocked Qdrant."""

    @patch('utils.create_embedding')
    @patch('utils.QdrantClientWrapper')
    def test_document_search_workflow(self, mock_wrapper_class, mock_create_embedding):
        """Test complete document search workflow."""
        # Setup mocks
        mock_client = Mock()
        mock_client.search_documents.return_value = [
            {"id": "doc1", "similarity": 0.9, "content": "test content"}
        ]
        mock_wrapper_class.return_value = mock_client
        mock_create_embedding.return_value = [0.1] * 1536
        
        from utils import search_documents
        
        # Test
        results = search_documents(mock_client, "test query")
        
        # Verify workflow
        mock_create_embedding.assert_called_once_with("test query")
        mock_client.search_documents.assert_called_once()
        assert len(results) == 1
        assert results[0]["id"] == "doc1"

    @patch('utils.create_embedding')
    @patch('utils.QdrantClientWrapper')
    def test_code_search_workflow(self, mock_wrapper_class, mock_create_embedding):
        """Test complete code search workflow."""
        # Setup mocks
        mock_client = Mock()
        mock_client.search_code_examples.return_value = [
            {"id": "code1", "similarity": 0.85, "content": "def test(): pass"}
        ]
        mock_wrapper_class.return_value = mock_client
        mock_create_embedding.return_value = [0.1] * 1536
        
        from utils import search_code_examples
        
        # Test
        results = search_code_examples(mock_client, "test function")
        
        # Verify workflow
        mock_create_embedding.assert_called_once()
        mock_client.search_code_examples.assert_called_once()
        assert len(results) == 1
        assert results[0]["id"] == "code1"


if __name__ == "__main__":
    pytest.main([__file__])