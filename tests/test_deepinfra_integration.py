"""
Integration tests for DeepInfra embedding provider.

Tests DeepInfra API configuration, embedding creation, error handling,
and integration with the existing flexible API system.
"""
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from utils import (
    get_embeddings_client, create_embedding, create_embeddings_batch
)
from embedding_config import get_embedding_dimensions, validate_embeddings_config
from qdrant_wrapper import QdrantClientWrapper, get_collections_config


class TestDeepInfraConfiguration:
    """Test DeepInfra-specific configuration."""
    
    def setup_method(self):
        """Clean environment setup for each test."""
        self.original_env = {
            key: os.environ.get(key) for key in [
                'EMBEDDINGS_MODEL', 'EMBEDDINGS_API_KEY', 'EMBEDDINGS_API_BASE',
                'EMBEDDINGS_DIMENSIONS'
            ]
        }
        
        # Clear environment for clean testing
        for key in self.original_env:
            if key in os.environ:
                del os.environ[key]

    def teardown_method(self):
        """Restore environment after each test."""
        for key in self.original_env:
            if key in os.environ:
                del os.environ[key]
        
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value

    def test_qwen3_embeddings_configuration(self):
        """Test Qwen3-Embedding-0.6B configuration."""
        os.environ['EMBEDDINGS_MODEL'] = 'Qwen/Qwen3-Embedding-0.6B'
        os.environ['EMBEDDINGS_API_KEY'] = 'deepinfra-key'
        os.environ['EMBEDDINGS_API_BASE'] = 'https://api.deepinfra.com/v1/openai'
        
        client = get_embeddings_client()
        assert client.api_key == 'deepinfra-key'
        assert str(client.base_url).rstrip('/') == 'https://api.deepinfra.com/v1/openai'

    def test_deepinfra_dimension_detection(self):
        """Test automatic dimension detection for DeepInfra models."""
        # Test Qwen3 model
        os.environ['EMBEDDINGS_MODEL'] = 'Qwen/Qwen3-Embedding-0.6B'
        os.environ.pop('EMBEDDINGS_DIMENSIONS', None)
        assert get_embedding_dimensions() == 1024
        
        # Test BGE model
        os.environ['EMBEDDINGS_MODEL'] = 'BAAI/bge-large-en-v1.5'
        assert get_embedding_dimensions() == 1024
        
        # Test smaller model
        os.environ['EMBEDDINGS_MODEL'] = 'BAAI/bge-small-en-v1.5'
        assert get_embedding_dimensions() == 384

    def test_explicit_dimensions_override(self):
        """Test explicit EMBEDDINGS_DIMENSIONS configuration."""
        os.environ['EMBEDDINGS_MODEL'] = 'Qwen/Qwen3-Embedding-0.6B'
        os.environ['EMBEDDINGS_DIMENSIONS'] = '512'  # Custom dimension
        assert get_embedding_dimensions() == 512

    def test_dimension_validation(self):
        """Test dimension validation with various inputs."""
        # Valid positive integer
        os.environ['EMBEDDINGS_DIMENSIONS'] = '1024'
        assert get_embedding_dimensions() == 1024
        
        # Invalid negative number
        os.environ['EMBEDDINGS_DIMENSIONS'] = '-1'
        with pytest.raises(ValueError, match="must be positive"):
            get_embedding_dimensions()
        
        # Invalid non-integer
        os.environ['EMBEDDINGS_DIMENSIONS'] = 'invalid'
        with pytest.raises(ValueError, match="must be a valid integer"):
            get_embedding_dimensions()

    @patch('src.utils.get_embeddings_client')
    def test_qwen3_embedding_creation(self, mock_get_client):
        """Test embedding creation with Qwen3 model."""
        # Mock DeepInfra client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1024)]  # Qwen3 dimension
        mock_client.embeddings.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        os.environ['EMBEDDINGS_MODEL'] = 'Qwen/Qwen3-Embedding-0.6B'
        os.environ['EMBEDDINGS_API_KEY'] = 'test-key'
        
        # Test embedding creation
        embeddings = create_embeddings_batch(['test text'])
        
        # Verify model and dimensions
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1024
        call_args = mock_client.embeddings.create.call_args
        assert call_args[1]['model'] == 'Qwen/Qwen3-Embedding-0.6B'

    def test_config_validation_success(self):
        """Test successful configuration validation."""
        os.environ['EMBEDDINGS_API_KEY'] = 'valid-key'
        os.environ['EMBEDDINGS_MODEL'] = 'Qwen/Qwen3-Embedding-0.6B'
        
        # Should not raise exception
        assert validate_embeddings_config() is True

    def test_config_validation_missing_key(self):
        """Test validation failure with missing API key."""
        # Ensure no API key is set
        os.environ.pop('EMBEDDINGS_API_KEY', None)
        
        with pytest.raises(ValueError, match="No API key configured"):
            validate_embeddings_config()


class TestDeepInfraCollectionConfig:
    """Test collection configuration with DeepInfra dimensions."""
    
    def setup_method(self):
        """Setup for collection tests."""
        self.original_env = {
            key: os.environ.get(key) for key in [
                'EMBEDDINGS_MODEL', 'EMBEDDINGS_DIMENSIONS'
            ]
        }

    def teardown_method(self):
        """Cleanup after tests."""
        for key in self.original_env:
            if key in os.environ:
                del os.environ[key]
        
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value

    def test_dynamic_collection_config_qwen3(self):
        """Test collection configuration with Qwen3 dimensions."""
        os.environ['EMBEDDINGS_MODEL'] = 'Qwen/Qwen3-Embedding-0.6B'
        
        config = get_collections_config()
        
        # Verify both collections have correct dimensions
        assert config['crawled_pages']['vectors_config'].size == 1024
        assert config['code_examples']['vectors_config'].size == 1024
        
        # Verify payload schemas are preserved
        assert 'url' in config['crawled_pages']['payload_schema']
        assert 'content' in config['crawled_pages']['payload_schema']
        assert 'summary' in config['code_examples']['payload_schema']

    def test_dynamic_collection_config_custom_dims(self):
        """Test collection configuration with custom dimensions."""
        os.environ['EMBEDDINGS_DIMENSIONS'] = '768'
        
        config = get_collections_config()
        
        # Verify custom dimensions are used
        assert config['crawled_pages']['vectors_config'].size == 768
        assert config['code_examples']['vectors_config'].size == 768


class TestDimensionValidationIntegration:
    """Test dimension validation and collection recreation."""
    
    def setup_method(self):
        """Setup for integration tests."""
        self.original_env = {
            key: os.environ.get(key) for key in [
                'EMBEDDINGS_MODEL', 'EMBEDDINGS_DIMENSIONS'
            ]
        }

    def teardown_method(self):
        """Cleanup after tests."""
        for key in self.original_env:
            if key in os.environ:
                del os.environ[key]
        
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value

    @patch('qdrant_wrapper.QdrantClient')
    def test_dimension_mismatch_detection(self, mock_qdrant_client):
        """Test detection of dimension mismatches."""
        # Setup mock client
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        
        # Mock existing collection with 1536 dimensions
        mock_collection_info = Mock()
        mock_collection_info.config.params.vectors.size = 1536
        mock_client.get_collection.return_value = mock_collection_info
        mock_client.collection_exists.return_value = True
        
        # Configure for 1024 dimensions (Qwen3)
        os.environ['EMBEDDINGS_MODEL'] = 'Qwen/Qwen3-Embedding-0.6B'
        
        # Create wrapper and test validation
        wrapper = QdrantClientWrapper()
        from qdrant_client.models import VectorParams, Distance
        
        validation = wrapper._validate_collection_dimensions(
            'test_collection', 
            VectorParams(size=1024, distance=Distance.COSINE)
        )
        
        # Should detect mismatch
        assert validation['needs_recreation'] is True
        assert validation['current_size'] == 1536
        assert validation['expected_size'] == 1024

    @patch('qdrant_wrapper.QdrantClient')
    def test_collection_recreation_flow(self, mock_qdrant_client):
        """Test complete collection recreation flow."""
        # Setup mock client
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = True
        
        # Configure environment
        os.environ['EMBEDDINGS_MODEL'] = 'Qwen/Qwen3-Embedding-0.6B'
        
        # Create wrapper
        wrapper = QdrantClientWrapper()
        
        # Test recreation
        from qdrant_client.models import VectorParams, Distance
        vectors_config = VectorParams(size=1024, distance=Distance.COSINE)
        wrapper._recreate_collection_safely('test_collection', vectors_config)
        
        # Verify deletion and creation were called
        mock_client.delete_collection.assert_called_once_with('test_collection')
        mock_client.create_collection.assert_called_once()
        
        # Verify creation call arguments
        create_call = mock_client.create_collection.call_args
        assert create_call[1]['collection_name'] == 'test_collection'
        assert create_call[1]['vectors_config'].size == 1024


class TestBackwardCompatibility:
    """Test backward compatibility with existing OpenAI configurations."""
    
    def setup_method(self):
        """Setup for compatibility tests."""
        self.original_env = {
            key: os.environ.get(key) for key in [
                'EMBEDDINGS_MODEL', 'EMBEDDINGS_DIMENSIONS'
            ]
        }

    def teardown_method(self):
        """Cleanup after tests."""
        for key in self.original_env:
            if key in os.environ:
                del os.environ[key]
        
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value

    def test_openai_model_detection(self):
        """Test OpenAI model dimension detection still works."""
        os.environ['EMBEDDINGS_MODEL'] = 'text-embedding-3-small'
        os.environ.pop('EMBEDDINGS_DIMENSIONS', None)
        assert get_embedding_dimensions() == 1536
        
        os.environ['EMBEDDINGS_MODEL'] = 'text-embedding-3-large'
        assert get_embedding_dimensions() == 3072

    def test_unknown_model_fallback(self):
        """Test fallback behavior for unknown models."""
        os.environ['EMBEDDINGS_MODEL'] = 'unknown-model'
        os.environ.pop('EMBEDDINGS_DIMENSIONS', None)
        assert get_embedding_dimensions() == 1536  # Default fallback

    @patch('src.utils.get_embeddings_client')
    def test_fallback_embedding_dimensions(self, mock_get_client):
        """Test fallback embedding creation uses correct dimensions."""
        # Configure for Qwen3
        os.environ['EMBEDDINGS_MODEL'] = 'Qwen/Qwen3-Embedding-0.6B'
        
        # Mock client failure
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API error")
        mock_get_client.return_value = mock_client
        
        # Test fallback embedding
        embedding = create_embedding("test text")
        
        # Should use Qwen3 dimensions (1024) for fallback
        assert len(embedding) == 1024
        assert all(v == 0.0 for v in embedding)