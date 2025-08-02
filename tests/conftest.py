"""
Pytest configuration and fixtures for Crawl4AI MCP RAG tests.
"""
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for all tests
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables."""
    test_env = {
        # Backward compatibility
        "OPENAI_API_KEY": "test-openai-key",
        "MODEL_CHOICE": "gpt-3.5-turbo",  # Keep for backward compatibility testing
        
        # New flexible configuration
        "CHAT_MODEL": "gpt-3.5-turbo",
        "CHAT_API_KEY": "test-chat-api-key",
        "CHAT_API_BASE": "https://api.openai.com/v1",
        "EMBEDDINGS_MODEL": "text-embedding-3-small",
        "EMBEDDINGS_API_KEY": "test-embeddings-api-key", 
        "EMBEDDINGS_API_BASE": "https://api.openai.com/v1",
        
        # Other configuration
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "test-password",
        "USE_CONTEXTUAL_EMBEDDINGS": "false"
    }
    
    # Set test environment variables
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield
    
    # Restore original environment
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def mock_qdrant_client():
    """Provide a mock Qdrant client for testing."""
    mock_client = Mock()
    
    # Setup default behaviors
    mock_client.search_documents.return_value = []
    mock_client.search_code_examples.return_value = []
    mock_client.keyword_search_documents.return_value = []
    mock_client.keyword_search_code_examples.return_value = []
    mock_client.get_available_sources.return_value = []
    mock_client.health_check.return_value = {"status": "healthy"}
    mock_client.update_source_info.return_value = None
    mock_client.upsert_points.return_value = None
    mock_client.add_documents_to_qdrant.return_value = []
    mock_client.add_code_examples_to_qdrant.return_value = []
    
    return mock_client


@pytest.fixture
def sample_documents():
    """Provide sample document data for testing."""
    return [
        {
            "id": "doc1",
            "similarity": 0.95,
            "content": "Python is a high-level programming language.",
            "url": "https://python.org/docs/tutorial",
            "chunk_number": 1,
            "source_id": "python.org",
            "metadata": {"category": "tutorial"}
        },
        {
            "id": "doc2", 
            "similarity": 0.87,
            "content": "JavaScript is used for web development.",
            "url": "https://developer.mozilla.org/js",
            "chunk_number": 1,
            "source_id": "developer.mozilla.org",
            "metadata": {"category": "reference"}
        }
    ]


@pytest.fixture
def sample_code_examples():
    """Provide sample code example data for testing."""
    return [
        {
            "id": "code1",
            "similarity": 0.92,
            "content": "def hello_world():\n    print('Hello, World!')",
            "summary": "Basic Python function that prints Hello World",
            "url": "https://python.org/examples/hello",
            "chunk_number": 1,
            "source_id": "python.org",
            "metadata": {"language": "python", "difficulty": "beginner"}
        },
        {
            "id": "code2",
            "similarity": 0.84,
            "content": "function greet(name) {\n    console.log(`Hello, ${name}!`);\n}",
            "summary": "JavaScript function for greeting with a name parameter",
            "url": "https://developer.mozilla.org/js/examples",
            "chunk_number": 1,
            "source_id": "developer.mozilla.org",
            "metadata": {"language": "javascript", "difficulty": "beginner"}
        }
    ]


@pytest.fixture
def sample_sources():
    """Provide sample source data for testing."""
    return [
        {
            "source_id": "python.org",
            "summary": "Official Python documentation and tutorials",
            "total_word_count": 125000,
            "updated_at": "2024-01-15T10:30:00Z"
        },
        {
            "source_id": "developer.mozilla.org",
            "summary": "Mozilla Developer Network web development resources",
            "total_word_count": 89000,
            "updated_at": "2024-01-10T14:20:00Z"
        }
    ]


@pytest.fixture
def mock_openai_response():
    """Provide mock OpenAI API responses."""
    def create_mock_response(embeddings_data=None, chat_content="Test response"):
        if embeddings_data is None:
            embeddings_data = [[0.1] * 1536, [0.2] * 1536]
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=emb) for emb in embeddings_data]
        
        # For chat completions
        mock_chat_response = Mock()
        mock_chat_response.choices = [Mock(message=Mock(content=chat_content))]
        
        return mock_response, mock_chat_response
    
    return create_mock_response


@pytest.fixture
def mock_crawl4ai_result():
    """Provide mock Crawl4AI results."""
    mock_result = Mock()
    mock_result.success = True
    mock_result.markdown = """
    # Test Page
    
    This is a test page with some content.
    
    ```python
    def example_function():
        return "Hello, World!"
    ```
    
    More content here for testing.
    """
    mock_result.extracted_content = "This is a test page with some content. More content here for testing."
    mock_result.metadata = {"title": "Test Page", "description": "A test page"}
    
    return mock_result