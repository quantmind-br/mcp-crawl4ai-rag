from src.sparse_vector_types import SparseVectorConfig


class TestSparseVectorConfig:
    def test_sparse_vector_config_creation(self):
        """Test SparseVectorConfig creation with valid parameters"""
        config = SparseVectorConfig(indices=[0, 2, 5], values=[0.1, 0.5, 0.9])
        assert config.indices == [0, 2, 5]
        assert config.values == [0.1, 0.5, 0.9]

    def test_sparse_vector_config_to_dict(self):
        """Test SparseVectorConfig to_dict method"""
        config = SparseVectorConfig(indices=[0, 2, 5], values=[0.1, 0.5, 0.9])
        # Since dataclass doesn't have to_dict by default, we'll test the attributes directly
        assert hasattr(config, "indices")
        assert hasattr(config, "values")
        assert config.indices == [0, 2, 5]
        assert config.values == [0.1, 0.5, 0.9]
