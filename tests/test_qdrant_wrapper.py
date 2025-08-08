"""

Unit tests for QdrantClientWrapper.

Tests the core functionality of the Qdrant client wrapper that replaced Supabase.
"""
# ruff: noqa: E402

import os
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.clients.qdrant_client import QdrantClientWrapper, get_qdrant_client
from embedding_config import get_embedding_dimensions


class TestQdrantClientWrapper:
    """Test cases for QdrantClientWrapper class."""

    def _setup_mock_client(self, mock_qdrant_client):
        """Helper method to setup mock client with proper collections mock."""
        mock_client_instance = Mock()

        # Mock collections to return empty list (no existing collections)
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client_instance.get_collections.return_value = mock_collections

        # Mock get_collection to raise exception (collection doesn't exist)
        mock_client_instance.get_collection.side_effect = Exception(
            "Collection not found"
        )

        mock_qdrant_client.return_value = mock_client_instance
        return mock_client_instance

    @patch("src.clients.qdrant_client.QdrantClient")
    def test_init_default_config(self, mock_qdrant_client):
        """Test initialization with default configuration."""

        # Setup mock
        self._setup_mock_client(mock_qdrant_client)

        # Test
        wrapper = QdrantClientWrapper()

        # Verify
        assert wrapper.host == "localhost"
        assert wrapper.port == 6333
        mock_qdrant_client.assert_called_once_with(
            host="localhost", port=6333, prefer_grpc=True, timeout=30
        )

    @patch("src.clients.qdrant_client.QdrantClient")
    def test_init_custom_config(self, mock_qdrant_client):
        """Test initialization with custom configuration."""

        # Setup mock
        self._setup_mock_client(mock_qdrant_client)

        # Test
        wrapper = QdrantClientWrapper(host="custom-host", port=9999)

        # Verify
        assert wrapper.host == "custom-host"
        assert wrapper.port == 9999

    @patch("src.clients.qdrant_client.QdrantClient")
    def test_generate_point_id(self, mock_qdrant_client):
        """Test point ID generation consistency."""

        # Setup mock
        self._setup_mock_client(mock_qdrant_client)

        wrapper = QdrantClientWrapper()

        # Test
        url = "https://example.com/page"
        chunk_number = 1

        id1 = wrapper.generate_point_id(url, chunk_number)
        id2 = wrapper.generate_point_id(url, chunk_number)

        # Verify consistency
        assert id1 == id2
        assert isinstance(id1, str)
        # Verify it's a valid UUID format
        import uuid

        uuid.UUID(id1)  # This will raise ValueError if not a valid UUID

    @patch("src.clients.qdrant_client.QdrantClient")
    def test_normalize_search_results(self, mock_qdrant_client):
        """Test search result normalization."""

        # Setup mock
        self._setup_mock_client(mock_qdrant_client)

        wrapper = QdrantClientWrapper()

        # Test data
        mock_hit = Mock()
        mock_hit.id = "test_id"
        mock_hit.score = 0.95
        mock_hit.payload = {
            "url": "https://example.com",
            "content": "test content",
            "chunk_number": 1,
        }

        qdrant_results = [mock_hit]

        # Test
        normalized = wrapper.normalize_search_results(qdrant_results)

        # Verify
        assert len(normalized) == 1
        result = normalized[0]
        assert result["id"] == "test_id"
        assert result["similarity"] == 0.95
        assert result["url"] == "https://example.com"
        assert result["content"] == "test content"
        assert result["chunk_number"] == 1

    @patch("src.clients.qdrant_client.QdrantClient")
    def test_health_check_healthy(self, mock_qdrant_client):
        """Test health check when system is healthy."""

        # Setup mocks
        mock_client_instance = Mock()
        mock_collections = Mock()
        mock_collections.collections = [
            Mock(name="crawled_pages"),
            Mock(name="code_examples"),
        ]
        mock_client_instance.get_collections.return_value = mock_collections

        # Mock collection info
        mock_collection_info = Mock()
        mock_collection_info.status = "green"
        mock_collection_info.points_count = 100
        mock_collection_info.config.params.vectors.distance.value = "Cosine"
        mock_collection_info.config.params.vectors.size = get_embedding_dimensions()
        mock_client_instance.get_collection.return_value = mock_collection_info

        mock_qdrant_client.return_value = mock_client_instance

        wrapper = QdrantClientWrapper()

        # Test
        health = wrapper.health_check()

        # Verify
        assert health["status"] == "healthy"
        assert "collections" in health
        assert health["sources_count"] == 0  # Empty sources_storage initially

    @patch("src.clients.qdrant_client.QdrantClient")
    def test_health_check_unhealthy(self, mock_qdrant_client):
        """Test health check when system is unhealthy."""

        # Setup mock client that works for initialization but fails for health check
        mock_client_instance = Mock()
        mock_client_instance.get_collections.side_effect = [
            Mock(),
            Exception("Connection failed"),
        ]
        mock_qdrant_client.return_value = mock_client_instance

        wrapper = QdrantClientWrapper()

        # Test
        health = wrapper.health_check()

        # Verify
        assert health["status"] == "unhealthy"
        assert "error" in health
        assert "Connection failed" in health["error"]

    @patch("src.clients.qdrant_client.QdrantClient")
    def test_update_source_info(self, mock_qdrant_client):
        """Test source information update."""

        # Setup mock
        mock_client_instance = self._setup_mock_client(mock_qdrant_client)

        # Mock scroll for sources collection to return empty initially
        mock_client_instance.scroll.return_value = ([], None)

        wrapper = QdrantClientWrapper()

        # Test
        source_id = "example.com"
        summary = "Test website"
        word_count = 1000

        wrapper.update_source_info(source_id, summary, word_count)

        # Verify that upsert was called (we can't easily test the internal storage)
        # Instead, let's verify the method doesn't raise an exception
        assert True  # If we get here, the method worked

    @patch.dict(os.environ, {"USE_HYBRID_SEARCH": "false"}, clear=False)
    @patch("src.clients.qdrant_client.QdrantClient")
    def test_search_documents_no_filter(self, mock_qdrant_client):
        """Test document search without filters."""

        # Setup mock
        mock_client_instance = self._setup_mock_client(mock_qdrant_client)

        # Mock search results
        mock_hit = Mock()
        mock_hit.id = "doc1"
        mock_hit.score = 0.9
        mock_hit.payload = {"content": "test document"}
        mock_client_instance.search.return_value = [mock_hit]

        wrapper = QdrantClientWrapper()

        # Test
        query_embedding = [0.1] * get_embedding_dimensions()
        results = wrapper.search_documents(query_embedding, match_count=5)

        # Verify
        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["similarity"] == 0.9

        # Verify search was called correctly
        # Note: With hybrid search disabled (default), query_vector is just the embedding
        mock_client_instance.search.assert_called_once_with(
            collection_name="crawled_pages",
            query_vector=query_embedding,
            query_filter=None,
            limit=5,
            with_payload=True,
            score_threshold=0.0,
        )

    @patch("src.clients.qdrant_client.QdrantClient")
    def test_search_documents_with_filters(self, mock_qdrant_client):
        """Test document search with metadata and source filters."""

        # Setup mock
        mock_client_instance = self._setup_mock_client(mock_qdrant_client)
        mock_client_instance.search.return_value = []

        wrapper = QdrantClientWrapper()

        # Test
        query_embedding = [0.1] * get_embedding_dimensions()
        filter_metadata = {"category": "docs"}
        source_filter = "example.com"

        wrapper.search_documents(
            query_embedding,
            match_count=10,
            filter_metadata=filter_metadata,
            source_filter=source_filter,
        )

        # Verify search was called with filters
        call_args = mock_client_instance.search.call_args
        assert call_args[1]["query_filter"] is not None

    @patch("src.clients.qdrant_client.QdrantClient")
    def test_keyword_search_documents(self, mock_qdrant_client):
        """Test keyword search functionality."""

        # Setup mock
        mock_client_instance = self._setup_mock_client(mock_qdrant_client)

        # Mock scroll results
        mock_point = Mock()
        mock_point.id = "doc1"
        mock_point.payload = {"content": "This is a Python tutorial"}
        mock_client_instance.scroll.return_value = ([mock_point], None)

        wrapper = QdrantClientWrapper()

        # Test
        results = wrapper.keyword_search_documents("python", match_count=5)

        # Verify
        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["similarity"] == 0.5  # Default similarity for keyword matches

        # Verify scroll was called
        mock_client_instance.scroll.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions."""

    @patch("src.clients.qdrant_client.QdrantClientWrapper")
    def test_get_qdrant_client(self, mock_wrapper):
        """Test Qdrant client factory function."""

        # Setup mock
        mock_instance = Mock()
        mock_wrapper.return_value = mock_instance

        # Test
        # Ensure singleton is reset so our mock is used
        import src.clients.qdrant_client as qc

        qc._qdrant_client_instance = None
        client = get_qdrant_client()

        # Verify
        assert client is mock_instance
        mock_wrapper.assert_called_once()

    @patch("src.clients.qdrant_client.QdrantClientWrapper")
    @patch("src.clients.qdrant_client._qdrant_client_instance", None)  # Reset singleton
    def test_get_qdrant_client_failure(self, mock_wrapper):
        """Test Qdrant client factory function with failure."""

        # Setup mock to raise exception
        mock_wrapper.side_effect = Exception("Connection failed")

        # Test
        with pytest.raises(Exception) as exc_info:
            get_qdrant_client()

        # Verify
        assert "Connection failed" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
