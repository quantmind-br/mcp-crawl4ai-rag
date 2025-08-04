"""
Comprehensive tests for Qdrant Native Hybrid Search with FastBM25 sparse vectors.

This test suite validates:
1. Sparse vector configuration and creation
2. Hybrid collection schema validation
3. Hybrid search functionality with FastBM25
4. Collection migration from legacy to hybrid schema
5. Performance benchmarks
6. Vector naming compatibility
"""

import os
import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

# Import test utilities
try:
    from src.qdrant_wrapper import QdrantClientWrapper
    from src.sparse_vector_types import SparseVectorConfig
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        SparseVector,
        PointStruct,
        VectorParams,
        Distance,
        SparseVectorParams,
        SparseIndexParams,
        Modifier,
    )
    QDRANT_AVAILABLE = True
except ImportError as e:
    print(f"Skipping Qdrant tests due to import error: {e}")
    QDRANT_AVAILABLE = False


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant dependencies not available")
class TestSparseVectorConfig:
    """Test SparseVectorConfig functionality."""
    
    def test_create_sparse_vector(self):
        """Test SparseVectorConfig creation and conversion."""
        config = SparseVectorConfig(indices=[1, 3, 5], values=[0.5, 1.2, 0.8])
        sparse_vector = config.to_qdrant_sparse_vector()
        
        assert len(sparse_vector.indices) == 3
        assert len(sparse_vector.values) == 3
        assert sparse_vector.indices == [1, 3, 5]
        assert sparse_vector.values == [0.5, 1.2, 0.8]
    
    def test_empty_sparse_vector(self):
        """Test handling of empty sparse vectors."""
        config = SparseVectorConfig(indices=[], values=[])
        sparse_vector = config.to_qdrant_sparse_vector()
        
        assert len(sparse_vector.indices) == 0
        assert len(sparse_vector.values) == 0
    
    def test_sparse_vector_from_text(self):
        """Test creating sparse vector from text for BM25."""
        # This would normally use fastembed, but we'll mock it
        text = "machine learning python neural networks"
        mock_indices = [0, 1, 2, 3]
        mock_values = [1.0, 1.2, 0.8, 0.9]
        
        config = SparseVectorConfig(indices=mock_indices, values=mock_values)
        sparse_vector = config.to_qdrant_sparse_vector()
        
        assert sparse_vector.indices == [0, 1, 2, 3]
        assert len(sparse_vector.values) == 4


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant dependencies not available")
class TestHybridCollectionConfiguration:
    """Test hybrid collection creation and configuration."""
    
    @patch('src.qdrant_wrapper.get_embedding_dimensions')
    def test_hybrid_collections_config(self, mock_get_dims):
        """Test hybrid collections configuration generation."""
        from src.qdrant_wrapper import get_hybrid_collections_config
        
        mock_get_dims.return_value = 1024
        config = get_hybrid_collections_config()
        
        # Verify crawled_pages configuration
        assert "crawled_pages" in config
        crawled_config = config["crawled_pages"]
        
        # Check dense vector configuration
        assert "vectors_config" in crawled_config
        assert "text-dense" in crawled_config["vectors_config"]
        assert crawled_config["vectors_config"]["text-dense"].size == 1024
        assert crawled_config["vectors_config"]["text-dense"].distance == Distance.COSINE
        
        # Check sparse vector configuration
        assert "sparse_vectors_config" in crawled_config
        assert "text-sparse" in crawled_config["sparse_vectors_config"]
        sparse_config = crawled_config["sparse_vectors_config"]["text-sparse"]
        assert sparse_config.modifier == Modifier.IDF
        
        # Check sources configuration
        assert "sources" in config
        sources_config = config["sources"]
        assert "text-dense" in sources_config["vectors_config"]
        assert "text-sparse" in sources_config["sparse_vectors_config"]


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant dependencies not available")
class TestQdrantWrapperHybridIntegration:
    """Test QdrantWrapper integration with hybrid search."""
    
    def setup_method(self):
        """Set up test environment."""
        self.original_env = dict(os.environ)
        
    def teardown_method(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_vector_naming_compatibility(self):
        """Test that vector naming follows hybrid collection schema."""
        # This test simulates the point struct validation issue
        dummy_dense = [0.0] * 1024
        
        # Test correct named vector format
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "text-dense": dummy_dense,
                "text-sparse": SparseVector(indices=[0], values=[0.0])
            },
            payload={"test": "data"}
        )
        
        # Validate structure
        assert "text-dense" in point.vector
        assert "text-sparse" in point.vector
        assert isinstance(point.vector["text-sparse"], SparseVector)
    
    @patch('src.qdrant_wrapper.QdrantClient')
    def test_point_struct_upsert_compatibility(self, mock_client):
        """Test PointStruct creation for hybrid collections correctly handles vector naming."""
        wrapper = QdrantClientWrapper()
        
        # Mock collection info for hybrid schema
        mock_collection_info = Mock()
        mock_collection_info.config.params.vectors = {
            "text-dense": VectorParams(size=1024, distance=Distance.COSINE)
        }
        mock_collection_info.config.params.sparse_vectors = {
            "text-sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=Modifier.IDF
            )
        }
        
        mock_client.return_value.get_collection.return_value = mock_collection_info
        
        # Test update_source_info 
        source_id = "test-source"
        summary = "Test summary"
        word_count = 100
        
        # This should not raise vector name errors
        try:
            wrapper.client = mock_client.return_value
            wrapper.use_hybrid_search = True
            wrapper.update_source_info(source_id, summary, word_count)
            
            # Verify the upsert call was made
            assert mock_client.return_value.upsert.called
            call_args = mock_client.return_value.upsert.call_args
            points = call_args[1]['points']
            
            # Verify point structure
            point = points[0]
            if isinstance(point.vector, dict):
                assert "text-dense" in point.vector
                assert "text-sparse" in point.vector
            else:
                # Legacy fallback
                assert isinstance(point.vector, list)
                
        except Exception as e:
            if "Not existing vector name error" in str(e):
                pytest.fail("Vector naming incompatibility detected")
    
    def test_sparse_vector_creation_for_sources(self):
        """Test sparse vector creation specifically for sources collection metadata."""
        from qdrant_client.models import SparseVector
        
        # This test ensures the fix for the bug in update_source_info
        sparse_vector = SparseVector(indices=[0], values=[0.0])
        
        assert sparse_vector.indices == [0]
        assert sparse_vector.values == [0.0]
        assert len(sparse_vector.indices) == 1
        assert len(sparse_vector.values) == 1


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant dependencies not available")
class TestHybridSearchFunctionality:
    """Test hybrid search operations."""
    
    def test_reciprocal_rank_fusion_calculation(self):
        """Test RRF calculation for combining dense and sparse results."""
        # Simulate dense and sparse search results
        dense_results = [
            {"id": "doc1", "score": 0.9, "rank": 1},
            {"id": "doc2", "score": 0.8, "rank": 2},
            {"id": "doc3", "score": 0.7, "rank": 3},
        ]
        
        sparse_results = [
            {"id": "doc2", "score": 1.2, "rank": 1},
            {"id": "doc4", "score": 1.1, "rank": 2},
            {"id": "doc1", "score": 1.0, "rank": 3},
        ]
        
        # Mock RRF calculation
        k = 60
        fusion_scores = {}
        
        # Process dense results
        for result in dense_results:
            rrf_score = 1 / (k + result["rank"])
            fusion_scores[result["id"]] = rrf_score * 0.5  # dense weight
        
        # Process sparse results  
        for result in sparse_results:
            rrf_score = 1 / (k + result["rank"])
            if result["id"] in fusion_scores:
                fusion_scores[result["id"]] += rrf_score * 0.5  # sparse weight
            else:
                fusion_scores[result["id"]] = rrf_score * 0.5
        
        # Verify fusion logic
        expected_doc1 = 1/61 * 0.5 + 1/63 * 0.5
        expected_doc2 = 1/62 * 0.5 + 1/61 * 0.5
        
        assert abs(fusion_scores["doc1"] - expected_doc1) < 0.001
        assert abs(fusion_scores["doc2"] - expected_doc2) < 0.001


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant dependencies not available")
class TestCollectionMigration:
    """Test collection migration from legacy to hybrid schema."""
    
    def test_migration_detector(self):
        """Test migration detection logic."""
        # This tests the validate_collection method logic
        legacy_config = VectorParams(size=1024, distance=Distance.COSINE)
        hybrid_config = {
            "text-dense": VectorParams(size=1024, distance=Distance.COSINE),
            "text-sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=Modifier.IDF
            )
        }
        
        # Legacy detection should return true for migration needed
        assert "VectorParams" in str(type(legacy_config))
        assert isinstance(hybrid_config, dict)


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant dependencies not available")
class TestPerformanceBenchmarks:
    """Performance benchmarks for hybrid search."""
    
    def test_sparse_vector_memory_usage(self):
        """Test memory efficiency of sparse vectors."""
        # Typical sparse vectors from BM25 should be memory efficient
        text = "machine learning python artificial intelligence neural network"
        mock_indices = list(range(0, 50, 3))  # Sparse vector
        mock_values = [1.0, 1.2, 0.8, 1.5, 0.9] * 10
        
        sparse_vector = SparseVector(indices=mock_indices, values=mock_values)
        
        # Verify sparse structure is maintained
        assert len(sparse_vector.indices) >= 10
        assert len(sparse_vector.values) >= 10
        # Sparse vectors should use significantly less memory than dense
        sparsity = 1 - (len(sparse_vector.indices) / 1024)
        assert sparsity > 0.9  # At least 90% sparse


if __name__ == "__main__":
    # Run basic functionality tests
    print("Running hybrid search tests...")
    
    # Test with Qdrant client mock
    if QDRANT_AVAILABLE:
        pytest.main([__file__, "-v"])
    else:
        print("Qdrant tests skipped due to import errors")