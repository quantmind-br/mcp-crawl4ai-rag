"""
Tests for the RAG service.
"""

import pytest
import os
from unittest.mock import Mock, patch
from src.services.rag_service import (
    RagService,
    add_documents_to_vector_db,
    add_code_examples_to_vector_db,
    search_documents,
    search_code_examples,
    update_source_info,
)


class TestRagService:
    """Test cases for the RagService class."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client for testing."""
        return Mock()

    @pytest.fixture
    def mock_reranking_model(self):
        """Create a mock reranking model for testing."""
        return Mock()

    @pytest.fixture
    def rag_service(self, mock_qdrant_client, mock_reranking_model):
        """Create a RagService instance for testing."""
        return RagService(mock_qdrant_client, mock_reranking_model)

    def test_init(self, mock_qdrant_client, mock_reranking_model):
        """Test RagService initialization."""
        service = RagService(mock_qdrant_client, mock_reranking_model)
        assert service.qdrant_client == mock_qdrant_client
        assert service.reranking_model == mock_reranking_model

    @patch.dict(os.environ, {"USE_RERANKING": "false", "USE_HYBRID_SEARCH": "false"})
    def test_search_documents(self, rag_service):
        """Test searching for documents."""
        # Mock the embedding service
        with patch(
            "src.services.rag_service.create_embedding"
        ) as mock_create_embedding:
            mock_create_embedding.return_value = [0.1, 0.2, 0.3]

            # Mock Qdrant client response
            mock_results = [
                {"id": "doc1", "content": "Document 1", "similarity": 0.9},
                {"id": "doc2", "content": "Document 2", "similarity": 0.8},
            ]
            rag_service.qdrant_client.search_documents.return_value = mock_results

            results = rag_service.search_documents("test query", match_count=5)

            assert len(results) == 2
            assert results[0]["id"] == "doc1"
            assert results[1]["id"] == "doc2"
            mock_create_embedding.assert_called_once_with("test query")

    @patch.dict(os.environ, {"USE_RERANKING": "false", "USE_HYBRID_SEARCH": "false"})
    def test_search_code_examples(self, rag_service):
        """Test searching for code examples."""
        # Mock the embedding service
        with patch(
            "src.services.rag_service.create_embedding"
        ) as mock_create_embedding:
            mock_create_embedding.return_value = [0.1, 0.2, 0.3]

            # Mock Qdrant client response
            mock_results = [
                {"id": "code1", "content": "Code example 1", "similarity": 0.9},
                {"id": "code2", "content": "Code example 2", "similarity": 0.8},
            ]
            rag_service.qdrant_client.search_code_examples.return_value = mock_results

            results = rag_service.search_code_examples("test query", match_count=5)

            assert len(results) == 2
            assert results[0]["id"] == "code1"
            assert results[1]["id"] == "code2"
            mock_create_embedding.assert_called_once()

    @patch.dict(os.environ, {"USE_HYBRID_SEARCH": "true"})
    def test_hybrid_search_documents(self, rag_service):
        """Test hybrid search for documents."""
        # Mock Qdrant client response
        mock_results = [
            {"id": "doc1", "content": "Document 1", "score": 0.95},
            {"id": "doc2", "content": "Document 2", "score": 0.85},
        ]
        rag_service.qdrant_client.hybrid_search_documents.return_value = mock_results

        results = rag_service.hybrid_search(
            "test query", match_count=5, search_type="documents"
        )

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[1]["id"] == "doc2"

    @patch.dict(os.environ, {"USE_HYBRID_SEARCH": "true"})
    def test_hybrid_search_code_examples(self, rag_service):
        """Test hybrid search for code examples."""
        # Mock Qdrant client response
        mock_results = [
            {"id": "code1", "content": "Code example 1", "score": 0.92},
            {"id": "code2", "content": "Code example 2", "score": 0.82},
        ]
        rag_service.qdrant_client.hybrid_search_code_examples.return_value = (
            mock_results
        )

        results = rag_service.hybrid_search(
            "test query", match_count=5, search_type="code_examples"
        )

        assert len(results) == 2
        assert results[0]["id"] == "code1"
        assert results[1]["id"] == "code2"

    @patch.dict(os.environ, {"USE_RERANKING": "true"})
    def test_rerank_results(self, rag_service):
        """Test reranking search results."""
        # Mock reranking model
        mock_scores = [0.95, 0.85, 0.75]
        rag_service.reranking_model.predict.return_value = mock_scores

        # Test results
        test_results = [
            {"id": "doc1", "content": "Document 1"},
            {"id": "doc2", "content": "Document 2"},
            {"id": "doc3", "content": "Document 3"},
        ]

        reranked_results = rag_service.rerank_results("test query", test_results)

        assert len(reranked_results) == 3
        assert reranked_results[0]["rerank_score"] == 0.95
        assert reranked_results[1]["rerank_score"] == 0.85
        assert reranked_results[2]["rerank_score"] == 0.75
        assert reranked_results[0]["id"] == "doc1"  # Should be sorted by score

    @patch.dict(os.environ, {"USE_RERANKING": "false"})
    def test_rerank_results_disabled(self, rag_service):
        """Test reranking when disabled."""
        test_results = [
            {"id": "doc1", "content": "Document 1"},
            {"id": "doc2", "content": "Document 2"},
        ]

        # Should return results unchanged when reranking is disabled
        reranked_results = rag_service.rerank_results("test query", test_results)

        assert reranked_results == test_results
        # Model should not be called
        rag_service.reranking_model.predict.assert_not_called()

    def test_fuse_search_results(self, rag_service):
        """Test fusing dense and sparse search results."""
        # Test data
        dense_results = [
            {"id": "doc1", "content": "Document 1", "similarity": 0.9},
            {"id": "doc2", "content": "Document 2", "similarity": 0.8},
        ]

        sparse_results = [
            {"id": "doc2", "content": "Document 2", "similarity": 0.85},
            {"id": "doc3", "content": "Document 3", "similarity": 0.75},
        ]

        fused_results = rag_service.fuse_search_results(
            dense_results, sparse_results, k=60
        )

        assert len(fused_results) == 3
        # doc2 should have combined score from both results
        # doc1 and doc3 should have scores from single sources
        # Results should be sorted by RRF score

    def test_update_source_info(self, rag_service):
        """Test updating source information."""
        rag_service.update_source_info("test-source", "Test summary", 1000)
        rag_service.qdrant_client.update_source_info.assert_called_once_with(
            "test-source", "Test summary", 1000
        )

    @patch.dict(os.environ, {"USE_RERANKING": "true", "USE_HYBRID_SEARCH": "false"})
    def test_search_with_reranking(self, rag_service):
        """Test search with reranking."""
        # Mock dependencies
        with patch.object(rag_service, "search_documents") as mock_search:
            with patch.object(rag_service, "rerank_results") as mock_rerank:
                mock_search.return_value = [
                    {"id": "doc1", "content": "Document 1"},
                    {"id": "doc2", "content": "Document 2"},
                ]
                mock_rerank.return_value = [
                    {"id": "doc1", "content": "Document 1", "rerank_score": 0.95},
                    {"id": "doc2", "content": "Document 2", "rerank_score": 0.85},
                ]

                results = rag_service.search_with_reranking(
                    "test query", search_type="documents"
                )

                assert len(results) == 2
                assert results[0]["rerank_score"] == 0.95
                mock_search.assert_called_once()
                mock_rerank.assert_called_once()


class TestAddDocumentsToVectorDB:
    """Test cases for add_documents_to_vector_db function."""

    @patch("src.services.rag_service.create_embeddings_batch")
    @patch("src.services.rag_service.process_chunk_with_context")
    def test_add_documents_to_vector_db(
        self, mock_process_chunk, mock_create_embeddings
    ):
        """Test adding documents to vector database."""
        # Mock Qdrant client
        mock_client = Mock()

        # Mock embedding functions
        mock_create_embeddings.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_process_chunk.return_value = (
            ("contextual content 1", True),
            ("contextual content 2", True),
        )

        # Mock point batches from client
        mock_points_batch = [
            {
                "id": 1,
                "content": "content 1",
                "payload": {"url": "http://example.com/1", "chunk_number": 1},
            },
            {
                "id": 2,
                "content": "content 2",
                "payload": {"url": "http://example.com/2", "chunk_number": 2},
            },
        ]
        mock_client.add_documents_to_qdrant.return_value = [mock_points_batch]

        # Test data
        urls = ["http://example.com/1", "http://example.com/2"]
        chunk_numbers = [1, 2]
        contents = ["content 1", "content 2"]
        metadatas = [{"source": "example.com"}, {"source": "example.com"}]
        url_to_full_document = {
            "http://example.com/1": "full document 1",
            "http://example.com/2": "full document 2",
        }

        # Call function
        with patch.dict(
            os.environ,
            {"USE_CONTEXTUAL_EMBEDDINGS": "true", "USE_HYBRID_SEARCH": "false"},
        ):
            add_documents_to_vector_db(
                mock_client,
                urls,
                chunk_numbers,
                contents,
                metadatas,
                url_to_full_document,
            )

        # Verify calls
        mock_client.add_documents_to_qdrant.assert_called_once()
        mock_create_embeddings.assert_called_once()
        mock_client.upsert_points.assert_called_once()

    def test_add_documents_to_vector_db_empty_urls(self):
        """Test adding documents with empty URLs list."""
        mock_client = Mock()

        # Should not raise an exception and should not call any methods
        add_documents_to_vector_db(mock_client, [], [], [], [], {})

        # No calls should be made to the client
        mock_client.add_documents_to_qdrant.assert_not_called()
        mock_client.upsert_points.assert_not_called()


class TestAddCodeExamplesToVectorDB:
    """Test cases for add_code_examples_to_vector_db function."""

    @patch("src.services.rag_service.create_embeddings_batch")
    def test_add_code_examples_to_vector_db(self, mock_create_embeddings):
        """Test adding code examples to vector database."""
        # Mock Qdrant client
        mock_client = Mock()

        # Mock embedding functions
        mock_create_embeddings.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        # Mock point batches from client
        mock_points_batch = [
            {
                "id": 1,
                "combined_text": "code example 1 with summary 1",
                "payload": {"url": "http://example.com/code1", "chunk_number": 1},
            },
            {
                "id": 2,
                "combined_text": "code example 2 with summary 2",
                "payload": {"url": "http://example.com/code2", "chunk_number": 2},
            },
        ]
        mock_client.add_code_examples_to_qdrant.return_value = [mock_points_batch]

        # Test data
        urls = ["http://example.com/code1", "http://example.com/code2"]
        chunk_numbers = [1, 2]
        code_examples = ["code example 1", "code example 2"]
        summaries = ["summary 1", "summary 2"]
        metadatas = [{"source": "example.com"}, {"source": "example.com"}]

        # Call function
        with patch.dict(os.environ, {"USE_HYBRID_SEARCH": "false"}):
            add_code_examples_to_vector_db(
                mock_client, urls, chunk_numbers, code_examples, summaries, metadatas
            )

        # Verify calls
        mock_client.add_code_examples_to_qdrant.assert_called_once()
        mock_create_embeddings.assert_called_once()
        mock_client.upsert_points.assert_called_once()

    def test_add_code_examples_to_vector_db_empty_urls(self):
        """Test adding code examples with empty URLs list."""
        mock_client = Mock()

        # Should not raise an exception and should not call any methods
        add_code_examples_to_vector_db(mock_client, [], [], [], [], [])

        # No calls should be made to the client
        mock_client.add_code_examples_to_qdrant.assert_not_called()
        mock_client.upsert_points.assert_not_called()


class TestStandaloneFunctions:
    """Test cases for standalone functions."""

    def test_search_documents_standalone(self):
        """Test standalone search_documents function."""
        mock_qdrant_client = Mock()
        mock_reranker = Mock()

        with patch("src.services.rag_service.RagService") as mock_rag_service_class:
            mock_rag_service_instance = Mock()
            mock_rag_service_class.return_value = mock_rag_service_instance
            mock_rag_service_instance.search_with_reranking.return_value = [
                {"id": "doc1", "content": "Document 1"}
            ]

            results = search_documents(
                mock_qdrant_client, "test query", reranker=mock_reranker
            )

            assert len(results) == 1
            assert results[0]["id"] == "doc1"
            mock_rag_service_class.assert_called_once_with(
                mock_qdrant_client, reranking_model=mock_reranker
            )

    def test_search_code_examples_standalone(self):
        """Test standalone search_code_examples function."""
        mock_qdrant_client = Mock()
        mock_reranker = Mock()

        with patch("src.services.rag_service.RagService") as mock_rag_service_class:
            mock_rag_service_instance = Mock()
            mock_rag_service_class.return_value = mock_rag_service_instance
            mock_rag_service_instance.search_with_reranking.return_value = [
                {"id": "code1", "content": "Code example 1"}
            ]

            results = search_code_examples(
                mock_qdrant_client, "test query", reranker=mock_reranker
            )

            assert len(results) == 1
            assert results[0]["id"] == "code1"
            mock_rag_service_class.assert_called_once_with(
                mock_qdrant_client, reranking_model=mock_reranker
            )

    def test_update_source_info_standalone(self):
        """Test standalone update_source_info function."""
        mock_qdrant_client = Mock()

        with patch("src.services.rag_service.RagService") as mock_rag_service_class:
            mock_rag_service_instance = Mock()
            mock_rag_service_class.return_value = mock_rag_service_instance

            update_source_info(mock_qdrant_client, "test-source", "Test summary", 1000)

            mock_rag_service_class.assert_called_once_with(mock_qdrant_client)
            mock_rag_service_instance.update_source_info.assert_called_once_with(
                "test-source", "Test summary", 1000
            )
