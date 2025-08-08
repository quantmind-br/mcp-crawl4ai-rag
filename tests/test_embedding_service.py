"""
Unit tests for the EmbeddingService class.

Tests the embedding service functionality including dense and sparse vector generation,
caching behavior, error handling, and contextual embedding generation.
"""

import os
import pytest
from unittest.mock import Mock, patch

# Import the service and related classes
from src.services.embedding_service import (
    EmbeddingService,
    SparseVectorEncoder,
    get_embedding_service,
    create_embedding,
    create_embeddings_batch,
    create_sparse_embedding,
    create_sparse_embeddings_batch,
    generate_contextual_embedding,
)
from src.sparse_vector_types import SparseVectorConfig


class TestSparseVectorEncoder:
    """Test the SparseVectorEncoder singleton class."""

    def setup_method(self):
        """Reset singleton state before each test."""
        SparseVectorEncoder._instance = None
        SparseVectorEncoder._encoder = None
        SparseVectorEncoder._trained = False

    def test_singleton_pattern(self):
        """Test that SparseVectorEncoder follows singleton pattern."""
        encoder1 = SparseVectorEncoder()
        encoder2 = SparseVectorEncoder()
        assert encoder1 is encoder2

    @patch("fastembed.SparseTextEmbedding")
    def test_encoder_lazy_loading(self, mock_sparse_embedding):
        """Test that encoder is lazily loaded on first use."""
        mock_encoder_instance = Mock()
        mock_sparse_embedding.return_value = mock_encoder_instance

        encoder = SparseVectorEncoder()

        # Encoder should not be initialized yet
        assert encoder._encoder is None

        # Call _ensure_encoder to trigger lazy loading
        encoder._ensure_encoder()

        # Now encoder should be initialized
        assert encoder._encoder is mock_encoder_instance
        mock_sparse_embedding.assert_called_once_with(model_name="Qdrant/bm25")

    @patch("fastembed.SparseTextEmbedding")
    def test_encode_empty_text(self, mock_sparse_embedding):
        """Test encoding empty or whitespace-only text."""
        encoder = SparseVectorEncoder()

        # Test empty string
        result = encoder.encode("")
        assert result.indices == []
        assert result.values == []

        # Test whitespace-only string
        result = encoder.encode("   ")
        assert result.indices == []
        assert result.values == []

    @patch("fastembed.SparseTextEmbedding")
    def test_encode_success(self, mock_sparse_embedding):
        """Test successful sparse vector encoding."""
        # Setup mock
        mock_encoder_instance = Mock()
        mock_sparse_result = Mock()
        mock_sparse_result.indices = Mock()
        mock_sparse_result.indices.tolist.return_value = [1, 5, 10]
        mock_sparse_result.values = Mock()
        mock_sparse_result.values.tolist.return_value = [0.5, 0.3, 0.8]

        mock_encoder_instance.embed.return_value = [mock_sparse_result]
        mock_sparse_embedding.return_value = mock_encoder_instance

        encoder = SparseVectorEncoder()
        result = encoder.encode("test text")

        # Verify result
        assert result.indices == [1, 5, 10]
        assert result.values == [0.5, 0.3, 0.8]

        # Verify encoder was trained
        assert encoder._trained is True

    @patch("fastembed.SparseTextEmbedding")
    def test_encode_batch_success(self, mock_sparse_embedding):
        """Test successful batch sparse vector encoding."""
        # Setup mock
        mock_encoder_instance = Mock()
        mock_sparse_result1 = Mock()
        mock_sparse_result1.indices = Mock()
        mock_sparse_result1.indices.tolist.return_value = [1, 2]
        mock_sparse_result1.values = Mock()
        mock_sparse_result1.values.tolist.return_value = [0.5, 0.3]

        mock_sparse_result2 = Mock()
        mock_sparse_result2.indices = Mock()
        mock_sparse_result2.indices.tolist.return_value = [3, 4]
        mock_sparse_result2.values = Mock()
        mock_sparse_result2.values.tolist.return_value = [0.7, 0.9]

        mock_encoder_instance.embed.return_value = [
            mock_sparse_result1,
            mock_sparse_result2,
        ]
        mock_sparse_embedding.return_value = mock_encoder_instance

        encoder = SparseVectorEncoder()
        results = encoder.encode_batch(["text1", "text2"])

        # Verify results
        assert len(results) == 2
        assert results[0].indices == [1, 2]
        assert results[0].values == [0.5, 0.3]
        assert results[1].indices == [3, 4]
        assert results[1].values == [0.7, 0.9]

    @patch("fastembed.SparseTextEmbedding")
    def test_encode_error_handling(self, mock_sparse_embedding):
        """Test error handling in sparse vector encoding."""
        # Setup mock to raise exception
        mock_encoder_instance = Mock()
        mock_encoder_instance.embed.side_effect = Exception("Encoding failed")
        mock_sparse_embedding.return_value = mock_encoder_instance

        encoder = SparseVectorEncoder()
        result = encoder.encode("test text")

        # Should return empty sparse vector on error
        assert result.indices == []
        assert result.values == []

    def test_encode_fastembed_not_available(self):
        """Test behavior when FastEmbed is not available."""
        encoder = SparseVectorEncoder()

        with patch(
            "fastembed.SparseTextEmbedding",
            side_effect=ImportError("FastEmbed not found"),
        ):
            with pytest.raises(
                ImportError, match="FastEmbed is required for sparse vectors"
            ):
                encoder.encode("test text")


class TestEmbeddingService:
    """Test the EmbeddingService class."""

    def setup_method(self):
        """Setup test environment."""
        # Reset singleton state
        SparseVectorEncoder._instance = None
        SparseVectorEncoder._encoder = None
        SparseVectorEncoder._trained = False

    @patch("src.services.embedding_service.get_embedding_cache")
    def test_init(self, mock_get_cache):
        """Test EmbeddingService initialization."""
        mock_cache = Mock()
        mock_get_cache.return_value = mock_cache

        service = EmbeddingService()

        assert service._cache is mock_cache
        assert isinstance(service._sparse_encoder, SparseVectorEncoder)

    def test_create_embedding_success(self):
        """Test successful single embedding creation."""
        service = EmbeddingService()
        service._cache = None  # Disable cache for this test

        # Mock the create_embeddings_batch method to return a successful result
        with patch.object(
            service, "create_embeddings_batch", return_value=[[0.1, 0.2, 0.3]]
        ):
            result = service.create_embedding("test text")

        assert result == [0.1, 0.2, 0.3]

    @patch("src.services.embedding_service.get_embedding_dimensions")
    def test_create_embedding_error(self, mock_get_dimensions):
        """Test embedding creation error handling."""
        # Setup mocks
        mock_get_dimensions.return_value = 1536

        service = EmbeddingService()
        service._cache = None  # Disable cache for this test

        # Mock the create_embeddings_batch method to raise an exception
        with patch.object(
            service, "create_embeddings_batch", side_effect=Exception("API Error")
        ):
            result = service.create_embedding("test text")

        # Should return zero embedding on error
        assert result == [0.0] * 1536

    @patch("src.services.embedding_service.get_embedding_dimensions")
    def test_create_embedding_empty_result(self, mock_get_dimensions):
        """Test embedding creation when batch returns empty result."""
        # Setup mocks
        mock_get_dimensions.return_value = 1536

        service = EmbeddingService()
        service._cache = None  # Disable cache for this test

        # Mock the create_embeddings_batch method to return empty list
        with patch.object(service, "create_embeddings_batch", return_value=[]):
            result = service.create_embedding("test text")

        # Should return zero embedding when no embeddings returned
        assert result == [0.0] * 1536

    @patch("fastembed.SparseTextEmbedding")
    def test_create_sparse_embedding(self, mock_sparse_embedding):
        """Test sparse embedding creation."""
        # Setup mock
        mock_encoder_instance = Mock()
        mock_sparse_result = Mock()
        mock_sparse_result.indices = Mock()
        mock_sparse_result.indices.tolist.return_value = [1, 5, 10]
        mock_sparse_result.values = Mock()
        mock_sparse_result.values.tolist.return_value = [0.5, 0.3, 0.8]

        mock_encoder_instance.embed.return_value = [mock_sparse_result]
        mock_sparse_embedding.return_value = mock_encoder_instance

        service = EmbeddingService()
        result = service.create_sparse_embedding("test text")

        assert result.indices == [1, 5, 10]
        assert result.values == [0.5, 0.3, 0.8]

    @patch.dict(os.environ, {"USE_HYBRID_SEARCH": "false"})
    @patch("src.services.embedding_service.get_embeddings_client")
    def test_create_embeddings_batch_dense_only(self, mock_get_client):
        """Test batch embedding creation with dense vectors only."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2]), Mock(embedding=[0.3, 0.4])]
        mock_client.embeddings.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        service = EmbeddingService()
        service._cache = None  # Disable cache for this test

        result = service.create_embeddings_batch(["text1", "text2"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]

    @patch.dict(os.environ, {"USE_HYBRID_SEARCH": "true"})
    @patch("src.services.embedding_service.get_embeddings_client")
    @patch("fastembed.SparseTextEmbedding")
    def test_create_embeddings_batch_hybrid(
        self, mock_sparse_embedding, mock_get_client
    ):
        """Test batch embedding creation with hybrid search (dense + sparse)."""
        # Setup dense embedding mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2]), Mock(embedding=[0.3, 0.4])]
        mock_client.embeddings.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Setup sparse embedding mock
        mock_encoder_instance = Mock()
        mock_sparse_result1 = Mock()
        mock_sparse_result1.indices = Mock()
        mock_sparse_result1.indices.tolist.return_value = [1, 2]
        mock_sparse_result1.values = Mock()
        mock_sparse_result1.values.tolist.return_value = [0.5, 0.3]

        mock_sparse_result2 = Mock()
        mock_sparse_result2.indices = Mock()
        mock_sparse_result2.indices.tolist.return_value = [3, 4]
        mock_sparse_result2.values = Mock()
        mock_sparse_result2.values.tolist.return_value = [0.7, 0.9]

        mock_encoder_instance.embed.return_value = [
            mock_sparse_result1,
            mock_sparse_result2,
        ]
        mock_sparse_embedding.return_value = mock_encoder_instance

        service = EmbeddingService()
        service._cache = None  # Disable cache for this test

        dense_vectors, sparse_vectors = service.create_embeddings_batch(
            ["text1", "text2"]
        )

        assert dense_vectors == [[0.1, 0.2], [0.3, 0.4]]
        assert len(sparse_vectors) == 2
        assert sparse_vectors[0].indices == [1, 2]
        assert sparse_vectors[0].values == [0.5, 0.3]

    @patch.dict(os.environ, {"USE_HYBRID_SEARCH": "false"})
    @patch("src.services.embedding_service.get_embeddings_client")
    def test_create_embeddings_batch_with_cache(self, mock_get_client):
        """Test batch embedding creation with cache hits and misses."""
        # Setup API mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.3, 0.4])]  # Only one new embedding
        mock_client.embeddings.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Setup cache mock
        mock_cache = Mock()
        mock_cache.get_batch.return_value = {"text1": [0.1, 0.2]}  # Cache hit for text1

        service = EmbeddingService()
        service._cache = mock_cache

        result = service.create_embeddings_batch(["text1", "text2"])

        # Should use cached embedding for text1 and API for text2
        assert result == [[0.1, 0.2], [0.3, 0.4]]

        # Verify cache was called
        mock_cache.get_batch.assert_called_once()
        mock_cache.set_batch.assert_called_once()

    @patch.dict(os.environ, {"USE_HYBRID_SEARCH": "false"})
    def test_create_embeddings_batch_empty_input(self):
        """Test batch embedding creation with empty input."""
        service = EmbeddingService()

        result = service.create_embeddings_batch([])
        assert result == []

    @patch.dict(os.environ, {"USE_HYBRID_SEARCH": "true"})
    def test_create_embeddings_batch_empty_input_hybrid(self):
        """Test batch embedding creation with empty input in hybrid mode."""
        service = EmbeddingService()

        result = service.create_embeddings_batch([])
        assert result == ([], [])

    @patch("src.services.embedding_service.get_chat_client")
    def test_generate_contextual_embedding_success(self, mock_get_client):
        """Test successful contextual embedding generation."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "This chunk discusses testing methodology."
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        service = EmbeddingService()

        result_text, success = service.generate_contextual_embedding(
            "Full document about testing", "This is a test chunk"
        )

        assert success is True
        assert "This chunk discusses testing methodology." in result_text
        assert "This is a test chunk" in result_text

    @patch("src.services.embedding_service.get_chat_client")
    def test_generate_contextual_embedding_api_error(self, mock_get_client):
        """Test contextual embedding generation with API error."""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client

        service = EmbeddingService()

        result_text, success = service.generate_contextual_embedding(
            "Full document", "Test chunk"
        )

        assert success is False
        assert result_text == "Test chunk"

    @patch("src.services.embedding_service.get_chat_client")
    @patch.dict(os.environ, {"CHAT_MODEL": "", "CHAT_FALLBACK_MODEL": ""})
    def test_generate_contextual_embedding_no_model(self, mock_get_client):
        """Test contextual embedding generation with no model configured but API fails."""
        # Even with no model configured, it will use gpt-4o-mini as default
        # So we test the case where the API call fails
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client

        service = EmbeddingService()

        result_text, success = service.generate_contextual_embedding(
            "Full document", "Test chunk"
        )

        assert success is False
        assert result_text == "Test chunk"

    @patch("src.services.embedding_service.get_chat_client")
    def test_generate_contextual_embedding_empty_response(self, mock_get_client):
        """Test contextual embedding generation with empty API response."""
        # Setup mock with empty response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = []
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        service = EmbeddingService()

        result_text, success = service.generate_contextual_embedding(
            "Full document", "Test chunk"
        )

        assert success is False
        assert result_text == "Test chunk"

    def test_process_chunk_with_context(self):
        """Test process_chunk_with_context method."""
        service = EmbeddingService()

        # Mock the generate_contextual_embedding method
        with patch.object(service, "generate_contextual_embedding") as mock_generate:
            mock_generate.return_value = ("contextual text", True)

            result = service.process_chunk_with_context(("url", "content", "full_doc"))

            assert result == ("contextual text", True)
            mock_generate.assert_called_once_with("full_doc", "content")


class TestGlobalFunctions:
    """Test global convenience functions."""

    def setup_method(self):
        """Reset global service instance."""
        import src.services.embedding_service

        src.services.embedding_service._embedding_service = None

    def test_get_embedding_service_singleton(self):
        """Test that get_embedding_service returns singleton instance."""
        service1 = get_embedding_service()
        service2 = get_embedding_service()

        assert service1 is service2
        assert isinstance(service1, EmbeddingService)

    @patch("src.services.embedding_service.get_embedding_service")
    def test_create_embedding_convenience(self, mock_get_service):
        """Test create_embedding convenience function."""
        mock_service = Mock()
        mock_service.create_embedding.return_value = [0.1, 0.2, 0.3]
        mock_get_service.return_value = mock_service

        result = create_embedding("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_service.create_embedding.assert_called_once_with("test text")

    @patch("src.services.embedding_service.get_embedding_service")
    def test_create_sparse_embedding_convenience(self, mock_get_service):
        """Test create_sparse_embedding convenience function."""
        mock_service = Mock()
        mock_sparse_vector = SparseVectorConfig(indices=[1, 2], values=[0.5, 0.3])
        mock_service.create_sparse_embedding.return_value = mock_sparse_vector
        mock_get_service.return_value = mock_service

        result = create_sparse_embedding("test text")

        assert result == mock_sparse_vector
        mock_service.create_sparse_embedding.assert_called_once_with("test text")

    @patch("src.services.embedding_service.get_embedding_service")
    def test_create_embeddings_batch_convenience(self, mock_get_service):
        """Test create_embeddings_batch convenience function."""
        mock_service = Mock()
        mock_service.create_embeddings_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_get_service.return_value = mock_service

        result = create_embeddings_batch(["text1", "text2"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_service.create_embeddings_batch.assert_called_once_with(["text1", "text2"])

    @patch("src.services.embedding_service.get_embedding_service")
    def test_create_sparse_embeddings_batch_convenience(self, mock_get_service):
        """Test create_sparse_embeddings_batch convenience function."""
        mock_service = Mock()
        mock_sparse_vectors = [
            SparseVectorConfig(indices=[1, 2], values=[0.5, 0.3]),
            SparseVectorConfig(indices=[3, 4], values=[0.7, 0.9]),
        ]
        mock_service.create_sparse_embeddings_batch.return_value = mock_sparse_vectors
        mock_get_service.return_value = mock_service

        result = create_sparse_embeddings_batch(["text1", "text2"])

        assert result == mock_sparse_vectors
        mock_service.create_sparse_embeddings_batch.assert_called_once_with(
            ["text1", "text2"]
        )

    @patch("src.services.embedding_service.get_embedding_service")
    def test_generate_contextual_embedding_convenience(self, mock_get_service):
        """Test generate_contextual_embedding convenience function."""
        mock_service = Mock()
        mock_service.generate_contextual_embedding.return_value = (
            "contextual text",
            True,
        )
        mock_get_service.return_value = mock_service

        result = generate_contextual_embedding("full doc", "chunk")

        assert result == ("contextual text", True)
        mock_service.generate_contextual_embedding.assert_called_once_with(
            "full doc", "chunk"
        )


class TestErrorHandling:
    """Test error handling scenarios."""

    @patch("src.services.embedding_service.get_embeddings_client")
    def test_api_retry_logic(self, mock_get_client):
        """Test API retry logic with exponential backoff."""
        # Setup mock to fail twice then succeed
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2])]

        mock_client.embeddings.create.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            mock_response,  # Success on third try
        ]
        mock_get_client.return_value = mock_client

        service = EmbeddingService()
        service._cache = None  # Disable cache

        with patch("time.sleep") as mock_sleep:  # Mock sleep to speed up test
            result = service._create_embeddings_api_call(["test text"])

        assert result == [[0.1, 0.2]]
        assert mock_client.embeddings.create.call_count == 3
        assert mock_sleep.call_count == 2  # Two retries

    @patch("src.services.embedding_service.get_embeddings_client")
    @patch("src.services.embedding_service.get_embedding_dimensions")
    def test_individual_fallback_on_batch_failure(
        self, mock_get_dimensions, mock_get_client
    ):
        """Test fallback to individual embedding creation when batch fails."""
        mock_get_dimensions.return_value = 2

        # Setup mock to fail on batch but succeed on individual calls
        mock_client = Mock()

        # Batch calls fail
        batch_side_effects = [Exception("Batch failed")] * 3

        # Individual calls succeed
        individual_responses = [
            Mock(data=[Mock(embedding=[0.1, 0.2])]),
            Mock(data=[Mock(embedding=[0.3, 0.4])]),
        ]

        mock_client.embeddings.create.side_effect = (
            batch_side_effects + individual_responses
        )
        mock_get_client.return_value = mock_client

        service = EmbeddingService()
        service._cache = None  # Disable cache

        with patch("time.sleep"):  # Mock sleep to speed up test
            result = service._create_embeddings_api_call(["text1", "text2"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        # 3 batch attempts + 2 individual calls
        assert mock_client.embeddings.create.call_count == 5

    @patch("src.services.embedding_service.get_embeddings_client")
    @patch("src.services.embedding_service.get_embedding_dimensions")
    def test_complete_api_failure_fallback(self, mock_get_dimensions, mock_get_client):
        """Test complete API failure with zero embedding fallback."""
        mock_get_dimensions.return_value = 2

        # Setup mock to always fail
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("Complete failure")
        mock_get_client.return_value = mock_client

        service = EmbeddingService()
        service._cache = None  # Disable cache

        with patch("time.sleep"):  # Mock sleep to speed up test
            result = service._create_embeddings_api_call(["text1", "text2"])

        # Should return zero embeddings as fallback
        assert result == [[0.0, 0.0], [0.0, 0.0]]
