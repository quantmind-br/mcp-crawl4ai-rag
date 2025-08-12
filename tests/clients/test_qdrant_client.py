"""
Tests for the Qdrant client wrapper.
"""

import pytest
import os
from unittest.mock import Mock, patch
from src.clients.qdrant_client import (
    QdrantClientWrapper,
    get_qdrant_client,
    get_collections_config,
    get_hybrid_collections_config,
    get_active_collections_config,
)


class TestQdrantClientWrapper:
    """Test cases for the QdrantClientWrapper class."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        return Mock()

    @pytest.fixture
    def qdrant_wrapper(self, mock_qdrant_client):
        """Create a QdrantClientWrapper instance with mocked dependencies."""
        with patch("src.clients.qdrant_client.QdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client
            wrapper = QdrantClientWrapper()
            wrapper.client = mock_qdrant_client
            return wrapper

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.clients.qdrant_client.QdrantClient") as mock_client_class:
                with patch(
                    "src.clients.qdrant_client.QdrantClientWrapper._ensure_collections_exist"
                ) as mock_ensure:
                    mock_client = Mock()
                    mock_client_class.return_value = mock_client
                    mock_client.get_collections.return_value = Mock()
                    mock_ensure.return_value = None  # No exception

                    wrapper = QdrantClientWrapper()

                    assert wrapper.host == "localhost"
                    assert wrapper.port == 6333
                    assert wrapper.use_hybrid_search is False

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        with patch.dict(
            os.environ,
            {
                "QDRANT_HOST": "test-host",
                "QDRANT_PORT": "7000",
                "USE_HYBRID_SEARCH": "true",
            },
        ):
            with patch("src.clients.qdrant_client.QdrantClient") as mock_client_class:
                with patch(
                    "src.clients.qdrant_client.QdrantClientWrapper._ensure_collections_exist"
                ) as mock_ensure:
                    mock_client = Mock()
                    mock_client_class.return_value = mock_client
                    mock_client.get_collections.return_value = Mock()
                    mock_ensure.return_value = None  # No exception

                    wrapper = QdrantClientWrapper()

                assert wrapper.host == "test-host"
                assert wrapper.port == 7000
                assert wrapper.use_hybrid_search is True

    def test_create_client_success(self):
        """Test successful client creation."""
        with patch("src.clients.qdrant_client.QdrantClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get_collections.return_value = Mock()
            mock_client_class.return_value = mock_client

            wrapper = QdrantClientWrapper()
            assert wrapper.client == mock_client

    def test_create_client_failure(self):
        """Test client creation failure."""
        with patch("src.clients.qdrant_client.QdrantClient") as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")

            with pytest.raises(ConnectionError, match="Cannot connect to Qdrant"):
                QdrantClientWrapper()

    def test_collection_exists_true(self, qdrant_wrapper):
        """Test collection existence check when collection exists."""
        qdrant_wrapper.client.get_collection.return_value = Mock()

        result = qdrant_wrapper._collection_exists("test_collection")

        assert result is True
        qdrant_wrapper.client.get_collection.assert_called_once_with("test_collection")

    def test_collection_exists_false(self, qdrant_wrapper):
        """Test collection existence check when collection doesn't exist."""
        qdrant_wrapper.client.get_collection.side_effect = Exception("Not found")

        result = qdrant_wrapper._collection_exists("test_collection")

        assert result is False

    def test_generate_point_id(self, qdrant_wrapper):
        """Test point ID generation."""
        url = "https://example.com/test"
        chunk_number = 5

        point_id = qdrant_wrapper.generate_point_id(url, chunk_number)

        assert isinstance(point_id, str)
        assert len(point_id) > 0

    def test_normalize_search_results(self, qdrant_wrapper):
        """Test search results normalization."""
        mock_hit = Mock()
        mock_hit.id = "test-id"
        mock_hit.score = 0.95
        mock_hit.payload = {"content": "test content", "url": "https://example.com"}

        qdrant_results = [mock_hit]
        normalized = qdrant_wrapper.normalize_search_results(qdrant_results)

        assert len(normalized) == 1
        assert normalized[0]["id"] == "test-id"
        assert normalized[0]["similarity"] == 0.95
        assert normalized[0]["content"] == "test content"
        assert normalized[0]["url"] == "https://example.com"

    def test_search_documents_success(self, qdrant_wrapper):
        """Test successful document search."""
        mock_results = [Mock()]
        qdrant_wrapper.client.search.return_value = mock_results
        qdrant_wrapper.normalize_search_results = Mock(return_value=[{"id": "test"}])

        results = qdrant_wrapper.search_documents([0.1, 0.2, 0.3], match_count=5)

        assert len(results) == 1
        assert results[0]["id"] == "test"
        qdrant_wrapper.client.search.assert_called_once()

    def test_search_documents_with_filter(self, qdrant_wrapper):
        """Test document search with metadata filter."""
        mock_results = [Mock()]
        qdrant_wrapper.client.search.return_value = mock_results
        qdrant_wrapper.normalize_search_results = Mock(return_value=[{"id": "test"}])

        filter_metadata = {"category": "test"}
        results = qdrant_wrapper.search_documents(
            [0.1, 0.2, 0.3], match_count=5, filter_metadata=filter_metadata
        )

        assert len(results) == 1
        qdrant_wrapper.client.search.assert_called_once()

    def test_search_documents_error(self, qdrant_wrapper):
        """Test document search with error handling."""
        qdrant_wrapper.client.search.side_effect = Exception("Search failed")

        results = qdrant_wrapper.search_documents([0.1, 0.2, 0.3])

        assert results == []

    def test_search_code_examples_success(self, qdrant_wrapper):
        """Test successful code examples search."""
        mock_results = [Mock()]
        qdrant_wrapper.client.search.return_value = mock_results
        qdrant_wrapper.normalize_search_results = Mock(return_value=[{"id": "test"}])

        results = qdrant_wrapper.search_code_examples([0.1, 0.2, 0.3], match_count=5)

        assert len(results) == 1
        assert results[0]["id"] == "test"
        qdrant_wrapper.client.search.assert_called_once()

    def test_search_code_examples_error(self, qdrant_wrapper):
        """Test code examples search with error handling."""
        qdrant_wrapper.client.search.side_effect = Exception("Search failed")

        results = qdrant_wrapper.search_code_examples([0.1, 0.2, 0.3])

        assert results == []

    def test_upsert_points(self, qdrant_wrapper):
        """Test upserting points."""
        mock_points = [Mock()]

        qdrant_wrapper.upsert_points("test_collection", mock_points)

        qdrant_wrapper.client.upsert.assert_called_once_with(
            collection_name="test_collection", points=mock_points, wait=True
        )

    def test_get_available_sources(self, qdrant_wrapper):
        """Test getting available sources."""
        mock_point = Mock()
        mock_point.payload = {
            "source_id": "test-source",
            "summary": "Test source",
            "total_words": 1000,
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
        }

        qdrant_wrapper.client.scroll.return_value = ([mock_point], None)

        sources = qdrant_wrapper.get_available_sources()

        assert len(sources) == 1
        assert sources[0]["source_id"] == "test-source"
        assert sources[0]["summary"] == "Test source"

    def test_get_available_sources_error(self, qdrant_wrapper):
        """Test getting available sources with error handling."""
        qdrant_wrapper.client.scroll.side_effect = Exception("Scroll failed")

        sources = qdrant_wrapper.get_available_sources()

        assert sources == []

    def test_update_source_info(self, qdrant_wrapper):
        """Test updating source information."""
        with patch(
            "src.clients.qdrant_client.get_embedding_dimensions", return_value=1536
        ):
            with patch(
                "src.clients.qdrant_client.uuid.uuid5", return_value="test-uuid"
            ):
                qdrant_wrapper.update_source_info("test-source", "Test summary", 1000)

                qdrant_wrapper.client.upsert.assert_called_once()

    def test_update_source_info_error(self, qdrant_wrapper):
        """Test updating source information with error handling."""
        qdrant_wrapper.client.upsert.side_effect = Exception("Upsert failed")

        with patch(
            "src.clients.qdrant_client.get_embedding_dimensions", return_value=1536
        ):
            qdrant_wrapper.update_source_info("test-source", "Test summary", 1000)
            # Should not raise exception, just log error

    def test_health_check_healthy(self, qdrant_wrapper):
        """Test health check when healthy."""
        mock_collections = Mock()
        mock_collections.collections = [Mock(name="test_collection")]
        qdrant_wrapper.client.get_collections.return_value = mock_collections

        mock_collection_info = Mock()
        mock_collection_info.status = "green"
        mock_collection_info.points_count = 100
        mock_collection_info.config.params.vectors.distance.value = "Cosine"
        mock_collection_info.config.params.vectors.size = 1536
        qdrant_wrapper.client.get_collection.return_value = mock_collection_info

        qdrant_wrapper.get_available_sources = Mock(
            return_value=[{"source_id": "test"}]
        )

        health = qdrant_wrapper.health_check()

        assert health["status"] == "healthy"
        assert "collections" in health
        assert health["sources_count"] == 1

    def test_health_check_unhealthy(self, qdrant_wrapper):
        """Test health check when unhealthy."""
        qdrant_wrapper.client.get_collections.side_effect = Exception(
            "Connection failed"
        )

        health = qdrant_wrapper.health_check()

        assert health["status"] == "unhealthy"
        assert "error" in health


class TestConfigurationFunctions:
    """Test cases for configuration functions."""

    def test_get_collections_config(self):
        """Test getting collections configuration."""
        with patch(
            "src.clients.qdrant_client.get_embedding_dimensions", return_value=1536
        ):
            config = get_collections_config()

            assert "crawled_pages" in config
            assert "code_examples" in config
            assert "sources" in config
            assert config["crawled_pages"]["vectors_config"].size == 1536

    def test_get_hybrid_collections_config(self):
        """Test getting hybrid collections configuration."""
        with patch(
            "src.clients.qdrant_client.get_embedding_dimensions", return_value=1536
        ):
            config = get_hybrid_collections_config()

            assert "crawled_pages" in config
            assert "vectors_config" in config["crawled_pages"]
            assert "sparse_vectors_config" in config["crawled_pages"]
            assert "text-dense" in config["crawled_pages"]["vectors_config"]

    def test_get_active_collections_config_legacy(self):
        """Test getting active collections configuration in legacy mode."""
        with patch.dict(os.environ, {"USE_HYBRID_SEARCH": "false"}):
            with patch(
                "src.clients.qdrant_client.get_embedding_dimensions", return_value=1536
            ):
                config = get_active_collections_config()

                assert "crawled_pages" in config
                assert isinstance(
                    config["crawled_pages"]["vectors_config"], Mock
                ) or hasattr(config["crawled_pages"]["vectors_config"], "size")

    def test_get_active_collections_config_hybrid(self):
        """Test getting active collections configuration in hybrid mode."""
        with patch.dict(os.environ, {"USE_HYBRID_SEARCH": "true"}):
            with patch(
                "src.clients.qdrant_client.get_embedding_dimensions", return_value=1536
            ):
                config = get_active_collections_config()

                assert "crawled_pages" in config
                assert isinstance(config["crawled_pages"]["vectors_config"], dict)
                assert "text-dense" in config["crawled_pages"]["vectors_config"]


class TestSingletonPattern:
    """Test cases for the singleton pattern."""

    def test_get_qdrant_client_singleton(self):
        """Test that get_qdrant_client returns the same instance."""
        with patch("src.clients.qdrant_client.QdrantClient") as mock_client_class:
            with patch(
                "src.clients.qdrant_client.QdrantClientWrapper._ensure_collections_exist"
            ) as mock_ensure:
                mock_client = Mock()
                mock_client.get_collections.return_value = Mock()
                mock_client_class.return_value = mock_client
                mock_ensure.return_value = None  # No exception

                # Reset singleton instance
                import src.clients.qdrant_client as qc_module

                qc_module._qdrant_client_instance = None

                client1 = get_qdrant_client()
                client2 = get_qdrant_client()

                assert client1 is client2

    def test_get_qdrant_client_recreate_on_failure(self):
        """Test that get_qdrant_client creates new instance when previous one fails."""
        with patch("src.clients.qdrant_client.QdrantClient") as mock_client_class:
            with patch(
                "src.clients.qdrant_client.QdrantClientWrapper._ensure_collections_exist"
            ) as mock_ensure:
                with patch(
                    "src.clients.qdrant_client.QdrantClientWrapper._create_client"
                ) as mock_create_client:
                    mock_client1 = Mock()
                    mock_client2 = Mock()
                    mock_create_client.side_effect = [mock_client1, mock_client2]
                    mock_ensure.return_value = None  # No exception

                    # Reset singleton instance
                    import src.clients.qdrant_client as qc_module

                    qc_module._qdrant_client_instance = None

                    client1 = get_qdrant_client()

                    # Make the first client unhealthy
                    mock_client1.get_collections.side_effect = Exception(
                        "Connection lost"
                    )

                    client2 = get_qdrant_client()

                    assert client1 is not client2
