"""
Tests for the RagService class.
"""

from unittest.mock import Mock, patch
import os
from src.services.rag_service import RagService


class TestRagService:
    """Test the RagService class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_qdrant_client = Mock()
        self.mock_reranking_model = Mock()
        self.rag_service = RagService(
            qdrant_client=self.mock_qdrant_client,
            reranking_model=self.mock_reranking_model,
        )

    def test_init(self):
        """Test RagService initialization."""
        assert self.rag_service.qdrant_client == self.mock_qdrant_client
        assert self.rag_service.reranking_model == self.mock_reranking_model
        assert hasattr(self.rag_service, "use_reranking")
        assert hasattr(self.rag_service, "use_hybrid_search")

    @patch("src.services.rag_service.create_embedding")
    def test_search_documents_success(self, mock_create_embedding):
        """Test successful document search."""
        # Setup mocks
        mock_create_embedding.return_value = [0.1, 0.2, 0.3]
        expected_results = [{"id": "1", "content": "test content"}]
        self.mock_qdrant_client.search_documents.return_value = expected_results

        # Execute
        results = self.rag_service.search_documents("test query", match_count=5)

        # Verify
        assert results == expected_results
        mock_create_embedding.assert_called_once_with("test query")
        self.mock_qdrant_client.search_documents.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3], match_count=5, filter_metadata=None
        )

    @patch("src.services.rag_service.create_embedding")
    def test_search_documents_error(self, mock_create_embedding):
        """Test document search with error."""
        # Setup mocks
        mock_create_embedding.return_value = [0.1, 0.2, 0.3]
        self.mock_qdrant_client.search_documents.side_effect = Exception("Search error")

        # Execute
        results = self.rag_service.search_documents("test query")

        # Verify
        assert results == []

    @patch("src.services.rag_service.create_embedding")
    def test_search_code_examples_success(self, mock_create_embedding):
        """Test successful code examples search."""
        # Setup mocks
        mock_create_embedding.return_value = [0.1, 0.2, 0.3]
        expected_results = [{"id": "1", "code": "test code"}]
        self.mock_qdrant_client.search_code_examples.return_value = expected_results

        # Execute
        results = self.rag_service.search_code_examples(
            "test query", source_id="github.com"
        )

        # Verify
        assert results == expected_results
        # Check that enhanced query was created
        expected_query = (
            "Code example for test query\n\nSummary: Example code showing test query"
        )
        mock_create_embedding.assert_called_once_with(expected_query)
        self.mock_qdrant_client.search_code_examples.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            match_count=10,
            filter_metadata=None,
            source_filter="github.com",
        )

    def test_update_source_info_success(self):
        """Test successful source info update."""
        # Execute
        self.rag_service.update_source_info("test.com", "Test summary", 1000)

        # Verify
        self.mock_qdrant_client.update_source_info.assert_called_once_with(
            "test.com", "Test summary", 1000
        )

    def test_update_source_info_error(self):
        """Test source info update with error."""
        # Setup mock to raise exception
        self.mock_qdrant_client.update_source_info.side_effect = Exception(
            "Update error"
        )

        # Execute (should not raise exception)
        self.rag_service.update_source_info("test.com", "Test summary", 1000)

        # Verify the call was made
        self.mock_qdrant_client.update_source_info.assert_called_once()

    @patch.dict(os.environ, {"USE_HYBRID_SEARCH": "false"})
    def test_hybrid_search_disabled_documents(self):
        """Test hybrid search when disabled - should fallback to regular search."""
        # Setup
        self.rag_service.use_hybrid_search = False
        expected_results = [{"id": "1", "content": "test"}]

        with patch.object(
            self.rag_service, "search_documents", return_value=expected_results
        ) as mock_search:
            # Execute
            results = self.rag_service.hybrid_search(
                "test query", search_type="documents"
            )

            # Verify
            assert results == expected_results
            mock_search.assert_called_once_with("test query", 10, None)

    @patch.dict(os.environ, {"USE_HYBRID_SEARCH": "true"})
    def test_hybrid_search_enabled_documents(self):
        """Test hybrid search when enabled for documents."""
        # Setup
        self.rag_service.use_hybrid_search = True
        expected_results = [{"id": "1", "content": "test", "rrf_score": 0.5}]
        self.mock_qdrant_client.hybrid_search_documents.return_value = expected_results

        # Execute
        results = self.rag_service.hybrid_search("test query", search_type="documents")

        # Verify
        assert results == expected_results
        self.mock_qdrant_client.hybrid_search_documents.assert_called_once_with(
            query="test query", match_count=10, filter_metadata=None
        )

    @patch.dict(os.environ, {"USE_HYBRID_SEARCH": "true"})
    def test_hybrid_search_enabled_code_examples(self):
        """Test hybrid search when enabled for code examples."""
        # Setup
        self.rag_service.use_hybrid_search = True
        expected_results = [{"id": "1", "code": "test", "rrf_score": 0.5}]
        self.mock_qdrant_client.hybrid_search_code_examples.return_value = (
            expected_results
        )

        # Execute
        results = self.rag_service.hybrid_search(
            "test query", search_type="code_examples"
        )

        # Verify
        assert results == expected_results
        self.mock_qdrant_client.hybrid_search_code_examples.assert_called_once_with(
            query="test query", match_count=10, filter_metadata=None
        )

    @patch("src.services.rag_service.cleanup_gpu_memory")
    def test_rerank_results_success(self, mock_cleanup):
        """Test successful result re-ranking."""
        # Setup
        self.rag_service.use_reranking = True
        results = [
            {"id": "1", "content": "first result"},
            {"id": "2", "content": "second result"},
        ]
        self.mock_reranking_model.predict.return_value = [0.8, 0.6]

        # Execute
        reranked = self.rag_service.rerank_results("test query", results)

        # Verify
        assert len(reranked) == 2
        assert reranked[0]["id"] == "1"  # Higher score should be first
        assert reranked[0]["rerank_score"] == 0.8
        assert reranked[1]["id"] == "2"
        assert reranked[1]["rerank_score"] == 0.6

        # Verify model was called correctly
        expected_pairs = [
            ["test query", "first result"],
            ["test query", "second result"],
        ]
        self.mock_reranking_model.predict.assert_called_once_with(expected_pairs)
        mock_cleanup.assert_called_once()

    def test_rerank_results_no_model(self):
        """Test re-ranking when no model is available."""
        # Setup
        rag_service = RagService(self.mock_qdrant_client, reranking_model=None)
        results = [{"id": "1", "content": "test"}]

        # Execute
        reranked = rag_service.rerank_results("test query", results)

        # Verify - should return original results
        assert reranked == results

    @patch.dict(os.environ, {"USE_RERANKING": "false"})
    def test_rerank_results_disabled(self):
        """Test re-ranking when disabled."""
        # Setup
        self.rag_service.use_reranking = False
        results = [{"id": "1", "content": "test"}]

        # Execute
        reranked = self.rag_service.rerank_results("test query", results)

        # Verify - should return original results
        assert reranked == results

    def test_rerank_results_error(self):
        """Test re-ranking with error."""
        # Setup
        self.rag_service.use_reranking = True
        results = [{"id": "1", "content": "test"}]
        self.mock_reranking_model.predict.side_effect = Exception("Rerank error")

        # Execute
        reranked = self.rag_service.rerank_results("test query", results)

        # Verify - should return original results on error
        assert reranked == results

    def test_search_with_reranking_hybrid_enabled(self):
        """Test search with reranking and hybrid search enabled."""
        # Setup
        self.rag_service.use_hybrid_search = True
        self.rag_service.use_reranking = True

        with (
            patch.object(self.rag_service, "hybrid_search") as mock_hybrid,
            patch.object(self.rag_service, "rerank_results") as mock_rerank,
        ):
            mock_hybrid.return_value = [{"id": "1", "content": "test"}]
            mock_rerank.return_value = [
                {"id": "1", "content": "test", "rerank_score": 0.8}
            ]

            # Execute
            self.rag_service.search_with_reranking("test query")

            # Verify
            mock_hybrid.assert_called_once_with("test query", 10, None, "documents")
            mock_rerank.assert_called_once()

    def test_fuse_search_results(self):
        """Test result fusion using RRF."""
        # Setup
        dense_results = [
            {"id": "1", "content": "first"},
            {"id": "2", "content": "second"},
        ]
        sparse_results = [
            {"id": "2", "content": "second"},  # Same as dense result
            {"id": "3", "content": "third"},  # New result
        ]

        # Execute
        fused = self.rag_service.fuse_search_results(
            dense_results, sparse_results, k=60
        )

        # Verify
        assert len(fused) == 3
        # Check that RRF scores were calculated
        for result in fused:
            assert "rrf_score" in result
            assert "dense_rank" in result
            assert "sparse_rank" in result

        # Result with id "2" should have highest score (appears in both)
        id_2_result = next(r for r in fused if r["id"] == "2")
        assert id_2_result["dense_rank"] == 2
        assert id_2_result["sparse_rank"] == 1

        # Results should be sorted by RRF score
        scores = [r["rrf_score"] for r in fused]
        assert scores == sorted(scores, reverse=True)


class TestRagServiceIntegration:
    """Integration tests for RagService."""

    @patch("src.services.rag_service.create_embedding")
    def test_end_to_end_search_flow(self, mock_create_embedding):
        """Test complete search flow from query to results."""
        # Setup
        mock_qdrant_client = Mock()
        mock_reranking_model = Mock()

        mock_create_embedding.return_value = [0.1, 0.2, 0.3]
        mock_qdrant_client.search_documents.return_value = [
            {"id": "1", "content": "relevant content"},
            {"id": "2", "content": "less relevant"},
        ]
        mock_reranking_model.predict.return_value = [0.9, 0.3]

        rag_service = RagService(mock_qdrant_client, mock_reranking_model)
        rag_service.use_reranking = True

        # Execute
        with patch("src.services.rag_service.cleanup_gpu_memory"):
            results = rag_service.search_with_reranking("test query")

        # Verify
        assert len(results) == 2
        assert results[0]["id"] == "1"  # Should be reranked to first
        assert results[0]["rerank_score"] == 0.9
        assert results[1]["id"] == "2"
        assert results[1]["rerank_score"] == 0.3
