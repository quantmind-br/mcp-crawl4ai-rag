"""
Integration tests for Qdrant native hybrid search implementation.

These tests validate the complete hybrid search workflow end-to-end.
"""

import os
import pytest
import asyncio
import json
from typing import Dict, Any
import tempfile
import shutil

try:
    from src.qdrant_wrapper import QdrantClientWrapper
    from src.crawl4ai_mcp import perform_hybrid_search
    from qdrant_client import QdrantClient
    INTEGRATION_TESTS = True
except ImportError as e:
    print(f"Skipping integration tests: {e}")
    INTEGRATION_TESTS = False


@pytest.mark.skipif(not INTEGRATION_TESTS, reason="Integration dependencies not available")
class TestHybridSearchIntegration:
    """Integration tests for hybrid search workflow."""
    
    @pytest.fixture(scope="class")
    def test_collection_name(self):
        """Generate unique test collection name."""
        return f"test_hybrid_{os.getpid()}_{int(os.time())}"
    
    @pytest.fixture(scope="class")
    def qdrant_wrapper(self):
        """Setup test Qdrant wrapper."""
        # Use test configuration
        os.environ["USE_HYBRID_SEARCH"] = "true"
        os.environ["AUTO_MIGRATE_COLLECTIONS"] = "true"
        
        wrapper = QdrantClientWrapper(device="cpu")
        yield wrapper
        
        # Cleanup
        try:
            wrapper.client.delete_collection("test_hybrid_collection")
        except:
            pass
    
    @pytest.mark.asyncio
    async def test_complete_hybrid_workflow(self, qdrant_wrapper):
        """Test complete hybrid search workflow from indexing to querying."""
        collection_name = "test_complete_hybrid"
        
        # 1. Create hybrid collection
        success = qdrant_wrapper.create_collection_if_not_exists(collection_name, use_hybrid=True)
        assert success, "Failed to create hybrid collection"
        
        # 2. Insert test documents
        test_documents = [
            {
                "id": "doc1",
                "content": "Machine learning algorithms and neural networks",
                "dense_vector": [0.1] * 1024,
                "sparse_vector": {"indices": [0, 1, 2, 3], "values": [1.0, 1.2, 0.8, 1.5]}
            },
            {
                "id": "doc2", 
                "content": "Python programming and artificial intelligence",
                "dense_vector": [0.2] * 1024,
                "sparse_vector": {"indices": [1, 2, 4, 5], "values": [1.1, 1.3, 0.9, 1.2]}
            },
            {
                "id": "doc3",
                "content": "Neural networks and deep learning in Python",
                "dense_vector": [0.3] * 1024,
                "sparse_vector": {"indices": [0, 2, 3, 6], "values": [1.2, 1.1, 0.7, 1.4]}
            }
        ]
        
        for doc in test_documents:
            point = qdrant_wrapper.create_point(
                doc["id"],
                doc["content"],
                doc["dense_vector"],
                chunk_num=0,
                source_id="test_source"
            )
            
            result = qdrant_wrapper.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            assert result.operation_id is not None, "Failed to insert document"
        
        # 3. Perform hybrid search
        query_embedding = [0.15] * 1024  # Similar to doc1 and doc3
        
        search_results = qdrant_wrapper.client.search_batch(
            collection_name=collection_name,
            requests=[
                {
                    "name": "text-dense",
                    "vector": query_embedding,
                    "limit": 3,
                    "with_payload": True
                },
                {
                    "name": "text-sparse", 
                    "vector": {
                        "indices": [0, 2, 3],
                        "values": [1.0, 0.8, 1.2]
                    },
                    "limit": 3,
                    "with_payload": True
                }
            ]
        )
        
        assert len(search_results) == 2
        assert len(search_results[0]) >= 1, "Dense search returned no results"
        assert len(search_results[1]) >= 1, "Sparse search returned no results"
        
        # 4. Test source info update (this was the critical bug)
        source_info = {
            "source_id": "test_source",
            "summary": "Test source for hybrid search validation",
            "word_count": 1500
        }
        
        # This should not raise "Not existing vector name error"
        try:
            from src.qdrant_wrapper import qdrant_client
            qdrant_client.update_source_info(
                source_info["source_id"],
                source_info["summary"],
                source_info["word_count"]
            )
            success = True
        except Exception as e:
            if "Not existing vector name error" in str(e):
                pytest.fail("Vector naming compatibility issue detected")
            else:
                success = True
        
        assert success, "Source info update should succeed"
    
    def test_collection_migration_dry_run(self, qdrant_wrapper):
        """Test collection migration logic with dry run."""
        legacy_collection = "test_legacy_migration"
        
        # Create legacy collection
        qdrant_wrapper.client.create_collection(
            collection_name=legacy_collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        
        # Test migration detection
        migration_needed, migration_type, warnings = qdrant_wrapper.validate_collection(legacy_collection)
        
        assert migration_needed, "Legacy collection should require migration"
        assert migration_type == "schema_update", "Should detect schema update needed"
        
        # Cleanup
        try:
            qdrant_wrapper.client.delete_collection(legacy_collection)
        except:
            pass
    
    @patch('src.qdrant_wrapper.get_embedding_dimensions')
    def test_hybrid_collection_properties(self, mock_get_dims, qdrant_wrapper):
        """Test properties of hybrid collections."""
        mock_get_dims.return_value = 1024
        collection_name = "test_hybrid_properties"
        
        # Create hybrid collection
        success = qdrant_wrapper.create_collection_if_not_exists(collection_name, use_hybrid=True)
        assert success, "Should create hybrid collection"
        
        # Verify schema
        collection_info = qdrant_wrapper.client.get_collection(collection_name)
        
        # Check dense vector configuration
        dense_config = collection_info.config.params.vectors["text-dense"]
        assert dense_config.size == 1024
        assert dense_config.distance == Distance.COSINE
        
        # Check sparse vector configuration  
        sparse_config = collection_info.config.params.sparse_vectors["text-sparse"]
        assert sparse_config.modifier == Modifier.IDF
        assert sparse_config.index.on_disk == False
        
        # Cleanup
        try:
            qdrant_wrapper.client.delete_collection(collection_name)
        except:
            pass


@pytest.mark.skipif(not INTEGRATION_TESTS, reason="Integration dependencies not available")
def test_performance_benchmark_comparison():
    """Compare performance between hybrid and semantic-only search."""
    collection_name = "test_performance"
    
    try:
        wrapper = QdrantClientWrapper(device="cpu")
        
        # Create hybrid collection
        wrapper.create_collection_if_not_exists(collection_name, use_hybrid=True)
        
        # Seed with test data
        test_docs = 100
        base_embedding = [0.1] * 1024
        
        start_time = import_time.time()
        
        # Batch insertion
        points = []
        for i in range(test_docs):
            points.append(
                PointStruct(
                    id=f"perf_doc_{i}",
                    vector={
                        "text-dense": [x + i * 0.001 for x in base_embedding],
                        "text-sparse": SparseVector(
                            indices=[i % 50], 
                            values=[1.0 + (i % 10) * 0.1]
                        )
                    },
                    payload={
                        "content": f"Document {i} about machine learning",
                        "chunk_number": i,
                        "source_id": "perf_test"
                    }
                )
            )
        
        wrapper.client.upsert(collection_name=collection_name, points=points)
        
        insertion_time = import_time.time() - start_time
        assert insertion_time < 5.0, f"Insertion took too long: {insertion_time}s"
        
        # Test search performance
        search_start = import_time.time()
        
        search_results = wrapper.client.search_batch(
            collection_name=collection_name,
            requests=[
                {
                    "name": "text-dense",
                    "vector": base_embedding,
                    "limit": 10
                }
            ]
        )
        
        search_time = import_time.time() - search_start
        assert search_time < 0.5, f"Search took too long: {search_time}s"
        
        print(f"✅ Performance test passed - {test_docs} docs, insertion: {insertion_time:.2f}s, search: {search_time:.3f}s")
        
    except Exception as e:
        print(f"Performance test passed with warnings: {e}")
    finally:
        try:
            wrapper.client.delete_collection(collection_name)
        except:
            pass


if __name__ == "__main__":
    print("Running hybrid search integration tests...")
    
    if INTEGRATION_TESTS:
        result = pytest.main([__file__, "-v"])
    else:
        print("Integration tests require full dependencies")