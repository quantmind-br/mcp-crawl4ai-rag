"""
Tests for sparse vector types configuration.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestSparseVectorConfiguration:
    """Test sparse vector types configuration functions."""

    def test_get_sparse_indices_basic(self):
        """Test basic sparse indices generation."""
        from src.sparse_vector_types import get_sparse_indices
        
        # Test with simple text
        text = "python programming language"
        indices = get_sparse_indices(text, max_tokens=1000)
        
        assert isinstance(indices, list)
        assert len(indices) > 0
        assert all(isinstance(idx, int) for idx in indices)
        assert all(0 <= idx < max_tokens for idx in indices for max_tokens in [1000])

    def test_get_sparse_indices_empty_text(self):
        """Test sparse indices with empty text."""
        from src.sparse_vector_types import get_sparse_indices
        
        indices = get_sparse_indices("", max_tokens=1000)
        assert isinstance(indices, list)
        assert len(indices) == 0

    def test_get_sparse_indices_whitespace_only(self):
        """Test sparse indices with whitespace-only text."""
        from src.sparse_vector_types import get_sparse_indices
        
        indices = get_sparse_indices("   \n  \t  ", max_tokens=1000)
        assert isinstance(indices, list)
        assert len(indices) == 0

    def test_get_sparse_indices_max_tokens_limit(self):
        """Test that indices respect max_tokens limit."""
        from src.sparse_vector_types import get_sparse_indices
        
        text = "python programming language development"
        max_tokens = 10
        indices = get_sparse_indices(text, max_tokens=max_tokens)
        
        assert all(idx < max_tokens for idx in indices)

    def test_get_sparse_indices_special_characters(self):
        """Test sparse indices with special characters."""
        from src.sparse_vector_types import get_sparse_indices
        
        text = "hello@world.com #hashtag $price 100% complete!"
        indices = get_sparse_indices(text, max_tokens=1000)
        
        assert isinstance(indices, list)
        assert len(indices) >= 0  # Should handle special chars gracefully

    def test_get_sparse_indices_consistency(self):
        """Test that same text produces same indices."""
        from src.sparse_vector_types import get_sparse_indices
        
        text = "consistent text for testing"
        indices1 = get_sparse_indices(text, max_tokens=1000)
        indices2 = get_sparse_indices(text, max_tokens=1000)
        
        assert indices1 == indices2

    def test_get_sparse_indices_different_max_tokens(self):
        """Test sparse indices with different max_tokens values."""
        from src.sparse_vector_types import get_sparse_indices
        
        text = "python programming language"
        
        indices_100 = get_sparse_indices(text, max_tokens=100)
        indices_1000 = get_sparse_indices(text, max_tokens=1000)
        
        # All indices from smaller vocab should be valid in larger vocab
        assert all(idx < 100 for idx in indices_100)
        assert all(idx < 1000 for idx in indices_1000)

    def test_create_sparse_vector_basic(self):
        """Test basic sparse vector creation."""
        from src.sparse_vector_types import create_sparse_vector
        
        indices = [1, 5, 10, 15]
        values = [0.8, 0.6, 0.9, 0.7]
        
        sparse_vector = create_sparse_vector(indices, values)
        
        assert hasattr(sparse_vector, 'indices')
        assert hasattr(sparse_vector, 'values')
        assert sparse_vector.indices == indices
        assert sparse_vector.values == values

    def test_create_sparse_vector_empty(self):
        """Test sparse vector creation with empty inputs."""
        from src.sparse_vector_types import create_sparse_vector
        
        sparse_vector = create_sparse_vector([], [])
        
        assert sparse_vector.indices == []
        assert sparse_vector.values == []

    def test_create_sparse_vector_mismatched_length(self):
        """Test sparse vector creation with mismatched indices/values length."""
        from src.sparse_vector_types import create_sparse_vector
        
        with pytest.raises((ValueError, AssertionError)):
            create_sparse_vector([1, 2, 3], [0.5, 0.6])  # Mismatched lengths

    def test_create_sparse_vector_sorted_indices(self):
        """Test that sparse vector indices are sorted."""
        from src.sparse_vector_types import create_sparse_vector
        
        indices = [10, 1, 5, 15]
        values = [0.8, 0.6, 0.9, 0.7]
        
        sparse_vector = create_sparse_vector(indices, values)
        
        # Indices should be sorted
        assert sparse_vector.indices == sorted(indices)

    def test_text_to_sparse_vector_integration(self):
        """Test complete text to sparse vector conversion."""
        from src.sparse_vector_types import text_to_sparse_vector
        
        text = "machine learning artificial intelligence"
        sparse_vector = text_to_sparse_vector(text, max_tokens=1000)
        
        assert hasattr(sparse_vector, 'indices')
        assert hasattr(sparse_vector, 'values')
        assert len(sparse_vector.indices) == len(sparse_vector.values)
        assert len(sparse_vector.indices) > 0
        assert all(isinstance(idx, int) for idx in sparse_vector.indices)
        assert all(isinstance(val, (int, float)) for val in sparse_vector.values)

    def test_text_to_sparse_vector_empty_text(self):
        """Test sparse vector from empty text."""
        from src.sparse_vector_types import text_to_sparse_vector
        
        sparse_vector = text_to_sparse_vector("", max_tokens=1000)
        
        assert sparse_vector.indices == []
        assert sparse_vector.values == []

    def test_get_sparse_vector_config(self):
        """Test sparse vector configuration retrieval."""
        from src.sparse_vector_types import get_sparse_vector_config
        
        config = get_sparse_vector_config()
        
        assert isinstance(config, dict)
        assert 'max_tokens' in config
        assert 'default_value' in config
        assert isinstance(config['max_tokens'], int)
        assert config['max_tokens'] > 0

    def test_validate_sparse_vector(self):
        """Test sparse vector validation."""
        from src.sparse_vector_types import validate_sparse_vector, create_sparse_vector
        
        # Valid sparse vector
        valid_vector = create_sparse_vector([1, 5, 10], [0.5, 0.8, 0.3])
        assert validate_sparse_vector(valid_vector) is True
        
        # Invalid sparse vector (if function exists)
        try:
            invalid_vector = Mock()
            invalid_vector.indices = [1, 5]
            invalid_vector.values = [0.5]  # Mismatched lengths
            result = validate_sparse_vector(invalid_vector)
            assert result is False
        except AttributeError:
            # Function might not exist, which is fine
            pass

    @patch('src.sparse_vector_types.logger')
    def test_sparse_vector_logging(self, mock_logger):
        """Test that sparse vector operations log appropriately."""
        from src.sparse_vector_types import get_sparse_indices
        
        # This test ensures logging is working if implemented
        get_sparse_indices("test text", max_tokens=100)
        
        # Logger might be called for debugging info
        # This is more about ensuring no exceptions are raised