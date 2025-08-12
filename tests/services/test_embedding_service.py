"""
Tests for the embedding service.
"""

import pytest
import os
from unittest.mock import Mock, patch
from src.services.embedding_service import (
    EmbeddingService,
    SparseVectorEncoder,
    get_embedding_service,
    create_embedding,
    create_embeddings_batch,
    create_sparse_embedding,
    create_sparse_embeddings_batch,
    generate_contextual_embedding,
    health_check_gpu_acceleration,
)
from src.sparse_vector_types import SparseVectorConfig


class TestEmbeddingService:
    """Test cases for the EmbeddingService class."""

    @pytest.fixture
    def embedding_service(self):
        """Create an EmbeddingService instance for testing."""
        return EmbeddingService()

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache for testing."""
        return Mock()

    def test_init(self, embedding_service):
        """Test EmbeddingService initialization."""
        assert embedding_service is not None
        assert hasattr(embedding_service, "_sparse_encoder")
        assert hasattr(embedding_service, "_cache")

    @patch("src.services.embedding_service.get_embedding_cache")
    def test_create_embedding_single_text(self, mock_get_cache, embedding_service):
        """Test creating embedding for a single text."""
        # Mock cache to return None (no cache hit)
        mock_cache = Mock()
        mock_cache.get_batch.return_value = {}
        mock_get_cache.return_value = mock_cache

        # Mock API response
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        with patch.object(
            embedding_service,
            "_create_embeddings_api_call",
            return_value=[mock_embedding],
        ):
            result = embedding_service.create_embedding("test text")
            assert result == mock_embedding

    @patch("src.services.embedding_service.get_embedding_cache")
    def test_create_embedding_with_cache_hit(self, mock_get_cache, embedding_service):
        """Test creating embedding with cache hit."""
        # Mock cache to return cached embedding
        mock_cache = Mock()
        mock_cache.get_batch.return_value = {"test text": [0.5, 0.4, 0.3, 0.2, 0.1]}
        mock_get_cache.return_value = mock_cache

        result = embedding_service.create_embedding("test text")
        assert result == [0.5, 0.4, 0.3, 0.2, 0.1]

    def test_create_sparse_embedding(self, embedding_service):
        """Test creating sparse embedding."""
        with patch.object(embedding_service._sparse_encoder, "encode") as mock_encode:
            mock_encode.return_value = SparseVectorConfig(
                indices=[1, 2, 3], values=[0.1, 0.2, 0.3]
            )
            result = embedding_service.create_sparse_embedding("test text")
            assert isinstance(result, SparseVectorConfig)
            assert result.indices == [1, 2, 3]
            assert result.values == [0.1, 0.2, 0.3]

    @patch("src.services.embedding_service.get_embedding_cache")
    def test_create_embeddings_batch(self, mock_get_cache, embedding_service):
        """Test creating embeddings for multiple texts."""
        # Mock cache to return empty dict (no cache hits)
        mock_cache = Mock()
        mock_cache.get_batch.return_value = {}
        mock_get_cache.return_value = mock_cache

        # Mock API response
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        with patch.object(
            embedding_service,
            "_create_embeddings_api_call",
            return_value=mock_embeddings,
        ):
            result = embedding_service.create_embeddings_batch(["text1", "text2"])
            assert result == mock_embeddings

    @patch("src.services.embedding_service.get_embedding_cache")
    def test_create_embeddings_batch_with_cache(
        self, mock_get_cache, embedding_service
    ):
        """Test creating embeddings with partial cache hits."""
        # Mock cache to return one cached embedding
        mock_cache = Mock()
        mock_cache.get_batch.return_value = {"text1": [0.7, 0.8, 0.9]}
        mock_get_cache.return_value = mock_cache

        # Mock API response for cache miss
        with patch.object(
            embedding_service,
            "_create_embeddings_api_call",
            return_value=[[0.1, 0.2, 0.3]],
        ):
            result = embedding_service.create_embeddings_batch(["text1", "text2"])
            assert len(result) == 2
            assert result[0] == [0.7, 0.8, 0.9]  # From cache
            assert result[1] == [0.1, 0.2, 0.3]  # From API

    def test_create_sparse_embeddings_batch(self, embedding_service):
        """Test creating sparse embeddings for multiple texts."""
        with patch.object(
            embedding_service._sparse_encoder, "encode_batch"
        ) as mock_encode_batch:
            mock_encode_batch.return_value = [
                SparseVectorConfig(indices=[1, 2], values=[0.1, 0.2]),
                SparseVectorConfig(indices=[3, 4], values=[0.3, 0.4]),
            ]
            result = embedding_service.create_sparse_embeddings_batch(
                ["text1", "text2"]
            )
            assert len(result) == 2
            assert all(isinstance(r, SparseVectorConfig) for r in result)

    @patch.dict(os.environ, {"USE_HYBRID_SEARCH": "true"})
    @patch("src.services.embedding_service.get_embedding_cache")
    def test_create_embeddings_batch_hybrid_mode(
        self, mock_get_cache, embedding_service
    ):
        """Test creating embeddings in hybrid mode."""
        # Mock cache to return empty dict (no cache hits)
        mock_cache = Mock()
        mock_cache.get_batch.return_value = {}
        mock_get_cache.return_value = mock_cache

        # Mock API response
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        with patch.object(
            embedding_service,
            "_create_embeddings_api_call",
            return_value=mock_embeddings,
        ):
            with patch.object(
                embedding_service, "create_sparse_embeddings_batch"
            ) as mock_sparse:
                mock_sparse.return_value = [
                    SparseVectorConfig(indices=[1, 2], values=[0.1, 0.2]),
                    SparseVectorConfig(indices=[3, 4], values=[0.3, 0.4]),
                ]
                result = embedding_service.create_embeddings_batch(["text1", "text2"])
                assert isinstance(result, tuple)
                assert len(result) == 2
                assert result[0] == mock_embeddings
                assert len(result[1]) == 2

    def test_generate_contextual_embedding(self, embedding_service):
        """Test generating contextual embedding."""
        with patch("src.services.embedding_service.get_chat_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Context for the chunk"
            mock_response.choices[0].finish_reason = "stop"
            mock_client.chat.completions.create.return_value = mock_response

            result_text, was_contextualized = (
                embedding_service.generate_contextual_embedding(
                    "This is a full document with lots of content",
                    "This is a specific chunk",
                )
            )

            assert was_contextualized is True
            assert "Context for the chunk" in result_text
            assert "This is a specific chunk" in result_text

    def test_process_chunk_with_context(self, embedding_service):
        """Test processing chunk with context."""
        with patch.object(
            embedding_service, "generate_contextual_embedding"
        ) as mock_generate:
            mock_generate.return_value = ("contextual text", True)
            result = embedding_service.process_chunk_with_context(
                ("url", "content", "full_doc")
            )
            assert result == ("contextual text", True)


class TestSparseVectorEncoder:
    """Test cases for the SparseVectorEncoder class."""

    def test_singleton_pattern(self):
        """Test that SparseVectorEncoder follows singleton pattern."""
        encoder1 = SparseVectorEncoder()
        encoder2 = SparseVectorEncoder()
        assert encoder1 is encoder2

    def test_encode_empty_text(self):
        """Test encoding empty text."""
        encoder = SparseVectorEncoder()
        result = encoder.encode("")
        assert isinstance(result, SparseVectorConfig)
        assert result.indices == []
        assert result.values == []

    def test_encode_batch_empty_list(self):
        """Test encoding empty list of texts."""
        encoder = SparseVectorEncoder()
        result = encoder.encode_batch([])
        assert result == []


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_get_embedding_service(self):
        """Test get_embedding_service function."""
        service1 = get_embedding_service()
        service2 = get_embedding_service()
        assert service1 is service2  # Should be singleton

    def test_create_embedding_function(self):
        """Test create_embedding convenience function."""
        with patch(
            "src.services.embedding_service.get_embedding_service"
        ) as mock_get_service:
            mock_service = Mock()
            mock_get_service.return_value = mock_service
            mock_service.create_embedding.return_value = [0.1, 0.2, 0.3]

            result = create_embedding("test text")
            assert result == [0.1, 0.2, 0.3]
            mock_service.create_embedding.assert_called_once_with("test text")

    def test_create_embeddings_batch_function(self):
        """Test create_embeddings_batch convenience function."""
        with patch(
            "src.services.embedding_service.get_embedding_service"
        ) as mock_get_service:
            mock_service = Mock()
            mock_get_service.return_value = mock_service
            mock_service.create_embeddings_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]

            result = create_embeddings_batch(["text1", "text2"])
            assert result == [[0.1, 0.2], [0.3, 0.4]]
            mock_service.create_embeddings_batch.assert_called_once_with(
                ["text1", "text2"]
            )

    def test_create_sparse_embedding_function(self):
        """Test create_sparse_embedding convenience function."""
        with patch(
            "src.services.embedding_service.get_embedding_service"
        ) as mock_get_service:
            mock_service = Mock()
            mock_get_service.return_value = mock_service
            mock_service.create_sparse_embedding.return_value = SparseVectorConfig(
                indices=[1], values=[0.5]
            )

            result = create_sparse_embedding("test text")
            assert isinstance(result, SparseVectorConfig)
            assert result.indices == [1]
            assert result.values == [0.5]
            mock_service.create_sparse_embedding.assert_called_once_with("test text")

    def test_create_sparse_embeddings_batch_function(self):
        """Test create_sparse_embeddings_batch convenience function."""
        with patch(
            "src.services.embedding_service.get_embedding_service"
        ) as mock_get_service:
            mock_service = Mock()
            mock_get_service.return_value = mock_service
            mock_service.create_sparse_embeddings_batch.return_value = [
                SparseVectorConfig(indices=[1], values=[0.5])
            ]

            result = create_sparse_embeddings_batch(["text1"])
            assert len(result) == 1
            assert isinstance(result[0], SparseVectorConfig)
            mock_service.create_sparse_embeddings_batch.assert_called_once_with(
                ["text1"]
            )

    def test_generate_contextual_embedding_function(self):
        """Test generate_contextual_embedding convenience function."""
        with patch(
            "src.services.embedding_service.get_embedding_service"
        ) as mock_get_service:
            mock_service = Mock()
            mock_get_service.return_value = mock_service
            mock_service.generate_contextual_embedding.return_value = (
                "contextual text",
                True,
            )

            result = generate_contextual_embedding("full document", "chunk")
            assert result == ("contextual text", True)
            mock_service.generate_contextual_embedding.assert_called_once_with(
                "full document", "chunk"
            )


class TestHealthCheck:
    """Test cases for health check functions."""

    def test_health_check_gpu_acceleration_no_torch(self):
        """Test health check when torch is not available."""
        result = health_check_gpu_acceleration()
        assert result["gpu_available"] is False
        assert result["device_name"] == "CPU"
        assert result["test_passed"] is False

    @patch("src.services.embedding_service.torch")
    def test_health_check_gpu_acceleration_cuda_available(self, mock_torch):
        """Test health check when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_torch.cuda.get_device_properties.return_value.total_memory = (
            8 * 1024**3
        )  # 8GB

        # Mock tensor operations
        mock_tensor = Mock()
        mock_torch.randn.return_value = mock_tensor
        mock_tensor.__matmul__ = Mock()

        result = health_check_gpu_acceleration()
        assert result["gpu_available"] is True
        assert result["device_name"] == "Test GPU"
        assert result["test_passed"] is True

    @patch("src.services.embedding_service.torch")
    def test_health_check_gpu_acceleration_mps_available(self, mock_torch):
        """Test health check when MPS is available."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        # Mock tensor operations
        mock_tensor = Mock()
        mock_torch.randn.return_value = mock_tensor
        mock_tensor.sum = Mock()

        result = health_check_gpu_acceleration()
        assert result["gpu_available"] is True
        assert "Apple Silicon" in result["device_name"]
        assert result["test_passed"] is True
