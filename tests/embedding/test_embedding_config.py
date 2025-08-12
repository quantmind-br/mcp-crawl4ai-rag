import pytest
from unittest.mock import patch
from src.embedding_config import (
    get_embedding_dimensions,
    validate_embeddings_config,
    reset_embeddings_config,
)


class TestEmbeddingConfig:
    def setup_method(self):
        """Reset the singleton before each test"""
        reset_embeddings_config()

    def test_get_embedding_dimensions_default(self):
        """Test get_embedding_dimensions returns correct dimensions for default model"""
        with patch.dict(
            "os.environ", {"EMBEDDINGS_MODEL": "text-embedding-3-small"}, clear=True
        ):
            dimensions = get_embedding_dimensions()
            assert isinstance(dimensions, int)
            assert dimensions == 1536

    def test_get_embedding_dimensions_explicit(self):
        """Test get_embedding_dimensions with explicit dimensions"""
        with patch.dict("os.environ", {"EMBEDDINGS_DIMENSIONS": "512"}, clear=True):
            dimensions = get_embedding_dimensions()
            assert dimensions == 512

    def test_get_embedding_dimensions_invalid_explicit(self):
        """Test get_embedding_dimensions raises ValueError for invalid explicit dimensions"""
        with patch.dict("os.environ", {"EMBEDDINGS_DIMENSIONS": "invalid"}, clear=True):
            with pytest.raises(ValueError):
                get_embedding_dimensions()

    def test_get_embedding_dimensions_zero_explicit(self):
        """Test get_embedding_dimensions raises ValueError for zero dimensions"""
        with patch.dict("os.environ", {"EMBEDDINGS_DIMENSIONS": "0"}, clear=True):
            with pytest.raises(ValueError):
                get_embedding_dimensions()

    def test_get_embedding_dimensions_negative_explicit(self):
        """Test get_embedding_dimensions raises ValueError for negative dimensions"""
        with patch.dict("os.environ", {"EMBEDDINGS_DIMENSIONS": "-100"}, clear=True):
            with pytest.raises(ValueError):
                get_embedding_dimensions()

    def test_get_embedding_dimensions_unknown_model(self):
        """Test get_embedding_dimensions raises ValueError for unknown model"""
        with patch.dict(
            "os.environ", {"EMBEDDINGS_MODEL": "unknown-model"}, clear=True
        ):
            with pytest.raises(ValueError):
                get_embedding_dimensions()

    def test_validate_embeddings_config_valid(self):
        """Test validate_embeddings_config with valid configuration"""
        with patch.dict(
            "os.environ",
            {
                "EMBEDDINGS_API_KEY": "test-key",
                "EMBEDDINGS_MODEL": "text-embedding-3-small",
            },
            clear=True,
        ):
            assert validate_embeddings_config() is True

    def test_validate_embeddings_config_missing_api_key(self):
        """Test validate_embeddings_config raises ValueError when API key is missing"""
        with patch.dict(
            "os.environ", {"EMBEDDINGS_MODEL": "text-embedding-3-small"}, clear=True
        ):
            with pytest.raises(ValueError):
                validate_embeddings_config()

    def test_validate_embeddings_config_invalid_model(self):
        """Test validate_embeddings_config raises ValueError for invalid model"""
        with patch.dict(
            "os.environ",
            {"EMBEDDINGS_API_KEY": "test-key", "EMBEDDINGS_MODEL": "unknown-model"},
            clear=True,
        ):
            with pytest.raises(ValueError):
                validate_embeddings_config()
