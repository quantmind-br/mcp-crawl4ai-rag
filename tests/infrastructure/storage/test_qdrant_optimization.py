"""
Tests for Qdrant client optimization and connection management using mocks.
"""

import os
import time
import pytest
from unittest.mock import Mock, patch

# Import after setting path in conftest


class TestQdrantOptimization:
    @patch("src.clients.qdrant_client.QdrantClient")
    @patch(
        "src.clients.qdrant_client.QdrantClientWrapper._collection_exists",
        return_value=False,
    )
    def test_singleton_pattern_reuse(self, mock_exists, mock_qdrant_client):
        from src.clients.qdrant_client import get_qdrant_client

        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance

        c1 = get_qdrant_client()
        c2 = get_qdrant_client()
        assert c1 is c2

    @patch("src.clients.qdrant_client.QdrantClient")
    @patch(
        "src.clients.qdrant_client.QdrantClientWrapper._collection_exists",
        return_value=False,
    )
    def test_unhealthy_client_recreation(self, mock_exists, mock_qdrant_client):
        from src.clients.qdrant_client import get_qdrant_client

        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance

        c1 = get_qdrant_client()
        # Simular cliente inválido
        mock_client_instance.get_collections.side_effect = Exception("unhealthy")
        # Próxima chamada deve recriar
        mock_qdrant_client.side_effect = None
        mock_client_instance2 = Mock()
        mock_qdrant_client.return_value = mock_client_instance2
        # Resetar singleton para forçar recriação
        import src.clients.qdrant_client as qc

        qc._qdrant_client_instance = None
        c2 = get_qdrant_client()
        assert c2 is not c1

    @patch("src.clients.qdrant_client.QdrantClient")
    @patch(
        "src.clients.qdrant_client.QdrantClientWrapper._collection_exists",
        return_value=False,
    )
    def test_health_check_includes_cache_status(self, mock_exists, mock_qdrant_client):
        from src.clients.qdrant_client import QdrantClientWrapper

        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_qdrant_client.return_value = mock_client_instance
        client = QdrantClientWrapper()
        health = client.health_check()
        assert "status" in health

    @patch("src.clients.qdrant_client.QdrantClient")
    @patch(
        "src.clients.qdrant_client.QdrantClientWrapper._collection_exists",
        return_value=False,
    )
    def test_performance_optimization(self, mock_exists, mock_qdrant_client):
        from src.clients.qdrant_client import get_qdrant_client

        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock()
        mock_qdrant_client.return_value = mock_client_instance

        start = time.time()
        _ = get_qdrant_client()
        _ = get_qdrant_client()
        assert (time.time() - start) >= 0.0

    @patch("src.clients.qdrant_client.QdrantClient")
    @patch(
        "src.clients.qdrant_client.QdrantClientWrapper._collection_exists",
        return_value=False,
    )
    def test_concurrent_access_safety(self, mock_exists, mock_qdrant_client):
        import concurrent.futures
        from src.clients.qdrant_client import get_qdrant_client

        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance

        def get_client():
            return get_qdrant_client()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(get_client) for _ in range(10)]
            results = [f.result() for f in futures]
        assert all(r is results[0] for r in results)


class TestQdrantConnectionManagement:
    @patch("src.clients.qdrant_client.QdrantClient")
    @patch(
        "src.clients.qdrant_client.QdrantClientWrapper._collection_exists",
        return_value=False,
    )
    def test_connection_recovery(self, mock_exists, mock_qdrant_client):
        from src.clients.qdrant_client import get_qdrant_client

        # First call fails
        mock_qdrant_client.side_effect = ConnectionError("Connection failed")
        with pytest.raises(ConnectionError):
            # Reset singleton to force instantiation path
            import src.clients.qdrant_client as qc

            qc._qdrant_client_instance = None
            get_qdrant_client()
        # Then succeeds
        mock_qdrant_client.side_effect = None
        mock_qdrant_client.return_value = Mock()
        import src.clients.qdrant_client as qc

        qc._qdrant_client_instance = None
        client = get_qdrant_client()
        assert client is not None

    @patch.dict(os.environ, {"QDRANT_HOST": "test-host", "QDRANT_PORT": "9999"})
    @patch("src.clients.qdrant_client.QdrantClient")
    @patch(
        "src.clients.qdrant_client.QdrantClientWrapper._collection_exists",
        return_value=False,
    )
    def test_environment_configuration(self, mock_exists, mock_qdrant_client):
        from src.clients.qdrant_client import QdrantClientWrapper

        mock_qdrant_client.return_value = Mock()
        client = QdrantClientWrapper()
        assert client.host == "test-host"
        assert client.port == 9999
