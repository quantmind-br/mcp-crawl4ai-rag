"""
Tests for Qdrant client optimizations and connection management.

Tests the singleton pattern implementation and collection verification caching
to ensure unnecessary reconnections and schema checks are avoided.
"""
import pytest
import os
import time
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestQdrantOptimization:
    """Test Qdrant client optimization features."""

    def setup_method(self):
        """Reset singleton state before each test."""
        # Import and reset singleton state
        from src.qdrant_wrapper import QdrantClientWrapper
        # Reset global state
        import src.qdrant_wrapper as qw
        qw._qdrant_client_instance = None
        QdrantClientWrapper._collections_verified = False

    def teardown_method(self):
        """Clean up after each test."""
        # Reset global state
        import src.qdrant_wrapper as qw
        qw._qdrant_client_instance = None
        from src.qdrant_wrapper import QdrantClientWrapper
        QdrantClientWrapper._collections_verified = False

    @patch('src.qdrant_wrapper.QdrantClient')
    def test_singleton_pattern_reuse(self, mock_qdrant_client):
        """Test that the singleton pattern reuses existing client instances."""
        # Mock successful Qdrant client
        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        
        from src.qdrant_wrapper import get_qdrant_client
        
        # First call should create new instance
        client1 = get_qdrant_client()
        assert mock_qdrant_client.call_count == 1
        
        # Second call should reuse existing instance
        client2 = get_qdrant_client()
        assert mock_qdrant_client.call_count == 1  # No additional calls
        assert client1 is client2  # Same instance

    @patch('src.qdrant_wrapper.QdrantClient')
    def test_collection_verification_caching(self, mock_qdrant_client):
        """Test that collection verification is cached across instances."""
        # Mock successful Qdrant client
        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock()
        mock_client_instance.get_collection.side_effect = Exception("Collection not found")
        mock_qdrant_client.return_value = mock_client_instance
        
        from src.qdrant_wrapper import QdrantClientWrapper
        
        # First instance should verify collections
        client1 = QdrantClientWrapper()
        assert QdrantClientWrapper._collections_verified is True
        
        # Reset singleton to force new instance creation
        import src.qdrant_wrapper as qw
        qw._qdrant_client_instance = None
        
        # Second instance should skip verification
        with patch('src.qdrant_wrapper.QdrantClientWrapper._ensure_collections_exist') as mock_ensure:
            client2 = QdrantClientWrapper()
            mock_ensure.assert_not_called()  # Should not be called due to caching

    @patch('src.qdrant_wrapper.QdrantClient')
    def test_unhealthy_client_recreation(self, mock_qdrant_client):
        """Test that unhealthy clients are recreated."""
        # Mock client that becomes unhealthy
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        
        from src.qdrant_wrapper import get_qdrant_client
        
        # First call creates healthy client
        mock_client_instance.get_collections.return_value = Mock()
        client1 = get_qdrant_client()
        
        # Make client unhealthy
        mock_client_instance.get_collections.side_effect = Exception("Connection lost")
        
        # Next call should create new instance
        mock_new_client = Mock()
        mock_new_client.get_collections.return_value = Mock()
        mock_qdrant_client.return_value = mock_new_client
        
        client2 = get_qdrant_client()
        assert mock_qdrant_client.call_count == 2  # Two instances created

    def test_reset_verification_cache(self):
        """Test that verification cache can be reset."""
        from src.qdrant_wrapper import QdrantClientWrapper
        
        # Set verification flag
        QdrantClientWrapper._collections_verified = True
        
        # Reset cache
        QdrantClientWrapper.reset_verification_cache()
        
        # Should be reset
        assert QdrantClientWrapper._collections_verified is False

    @patch('src.qdrant_wrapper.QdrantClient')
    def test_health_check_includes_cache_status(self, mock_qdrant_client):
        """Test that health check includes verification cache status."""
        # Mock successful Qdrant client
        mock_client_instance = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client_instance.get_collections.return_value = mock_collections
        mock_qdrant_client.return_value = mock_client_instance
        
        from src.qdrant_wrapper import QdrantClientWrapper
        
        client = QdrantClientWrapper()
        health = client.health_check()
        
        assert "collections_verified" in health
        assert health["collections_verified"] is True

    @patch('src.qdrant_wrapper.QdrantClient')
    def test_performance_optimization(self, mock_qdrant_client):
        """Test that optimizations improve performance."""
        # Mock Qdrant client
        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        
        from src.qdrant_wrapper import get_qdrant_client
        
        # Measure time for multiple client retrievals
        start_time = time.time()
        
        # First call (creates client)
        client1 = get_qdrant_client()
        
        # Multiple subsequent calls (should reuse)
        for _ in range(50):
            client = get_qdrant_client()
            assert client is client1  # Same instance
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should be very fast due to singleton pattern
        assert elapsed < 0.05, f"Client retrieval too slow: {elapsed:.3f}s"
        
        # Should only create one Qdrant client instance
        assert mock_qdrant_client.call_count == 1

    @patch('src.qdrant_wrapper.QdrantClient')
    def test_concurrent_access_safety(self, mock_qdrant_client):
        """Test that singleton pattern is safe for concurrent access."""
        import concurrent.futures
        
        # Mock Qdrant client
        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        
        from src.qdrant_wrapper import get_qdrant_client
        
        clients = []
        
        def get_client():
            return get_qdrant_client()
        
        # Create multiple clients concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_client) for _ in range(20)]
            for future in concurrent.futures.as_completed(futures):
                clients.append(future.result())
        
        # All clients should be the same instance
        first_client = clients[0]
        for client in clients:
            assert client is first_client
        
        # Should only create one Qdrant client instance
        assert mock_qdrant_client.call_count == 1


class TestQdrantConnectionManagement:
    """Test Qdrant connection management and lifecycle."""

    def setup_method(self):
        """Reset state before each test."""
        import src.qdrant_wrapper as qw
        qw._qdrant_client_instance = None
        from src.qdrant_wrapper import QdrantClientWrapper
        QdrantClientWrapper._collections_verified = False

    def teardown_method(self):
        """Clean up after each test."""
        import src.qdrant_wrapper as qw
        qw._qdrant_client_instance = None
        from src.qdrant_wrapper import QdrantClientWrapper
        QdrantClientWrapper._collections_verified = False

    @patch('src.qdrant_wrapper.QdrantClient')
    def test_connection_error_handling(self, mock_qdrant_client):
        """Test proper handling of connection errors."""
        # Mock connection failure
        mock_qdrant_client.side_effect = ConnectionError("Cannot connect to Qdrant")
        
        from src.qdrant_wrapper import get_qdrant_client
        
        with pytest.raises(ConnectionError):
            get_qdrant_client()

    @patch('src.qdrant_wrapper.QdrantClient')
    def test_connection_recovery(self, mock_qdrant_client):
        """Test that connections can be recovered after failures."""
        from src.qdrant_wrapper import get_qdrant_client
        
        # First call fails
        mock_qdrant_client.side_effect = ConnectionError("Connection failed")
        
        with pytest.raises(ConnectionError):
            get_qdrant_client()
        
        # Recovery: successful connection
        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock()
        mock_qdrant_client.side_effect = None
        mock_qdrant_client.return_value = mock_client_instance
        
        # Should succeed on retry
        client = get_qdrant_client()
        assert client is not None

    @patch.dict(os.environ, {"QDRANT_HOST": "test-host", "QDRANT_PORT": "9999"})
    @patch('src.qdrant_wrapper.QdrantClient')
    def test_environment_configuration(self, mock_qdrant_client):
        """Test that environment variables are properly used."""
        # Mock successful client
        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        
        from src.qdrant_wrapper import QdrantClientWrapper
        
        client = QdrantClientWrapper()
        
        # Check that custom host and port are used
        assert client.host == "test-host"
        assert client.port == 9999
        
        # Verify Qdrant client was created with correct parameters
        mock_qdrant_client.assert_called_with(
            host="test-host",
            port=9999,
            prefer_grpc=True,
            timeout=30
        )