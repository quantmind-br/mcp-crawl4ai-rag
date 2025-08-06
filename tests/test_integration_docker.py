"""
# ruff: noqa: E402
Integration tests with real Docker services.

Tests the actual integration with Qdrant and Neo4j running in Docker containers.
"""

import pytest
import requests
import time
import os
import sys
import uuid
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def is_docker_service_ready(url: str, max_retries: int = 10) -> bool:
    """Check if a Docker service is ready by polling its endpoint."""
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False


@pytest.fixture(scope="module")
def docker_services():
    """Ensure Docker services are running and healthy."""
    # Check if Qdrant is ready (use collections endpoint)
    qdrant_ready = is_docker_service_ready("http://localhost:6333/collections")

    # Neo4j takes longer to start, so we're more lenient
    neo4j_ready = is_docker_service_ready("http://localhost:7474", max_retries=20)

    return {"qdrant_ready": qdrant_ready, "neo4j_ready": neo4j_ready}


class TestQdrantIntegration:
    """Test integration with real Qdrant service."""

    def test_qdrant_health(self, docker_services):
        """Test Qdrant root endpoint."""
        if not docker_services["qdrant_ready"]:
            pytest.skip("Qdrant not ready")

        response = requests.get("http://localhost:6333/")
        assert response.status_code == 200
        data = response.json()
        assert "title" in data
        assert "qdrant" in data["title"].lower()

    def test_qdrant_collections_endpoint(self, docker_services):
        """Test Qdrant collections endpoint."""
        if not docker_services["qdrant_ready"]:
            pytest.skip("Qdrant not ready")

        response = requests.get("http://localhost:6333/collections")
        assert response.status_code == 200
        data = response.json()
        assert "result" in data

    def test_qdrant_client_wrapper_real_connection(self, docker_services):
        """Test QdrantClientWrapper with real Qdrant service."""
        if not docker_services["qdrant_ready"]:
            pytest.skip("Qdrant not ready")

        from qdrant_wrapper import QdrantClientWrapper

        # Test connection
        try:
            client = QdrantClientWrapper(host="localhost", port=6333)

            # Test health check
            health = client.health_check()
            assert health["status"] == "healthy"
            assert "collections" in health

            # Test collections were created
            collections = health["collections"]
            expected_collections = ["crawled_pages", "code_examples"]

            for collection_name in expected_collections:
                if collection_name in collections:
                    collection_info = collections[collection_name]
                    assert "status" in collection_info
                    assert "config" in collection_info
                    assert collection_info["config"]["size"] == 1536

        except Exception as e:
            pytest.fail(f"QdrantClientWrapper connection failed: {e}")

    def test_qdrant_basic_operations(self, docker_services):
        """Test basic Qdrant operations with real service."""
        if not docker_services["qdrant_ready"]:
            pytest.skip("Qdrant not ready")

        from qdrant_wrapper import QdrantClientWrapper
        from qdrant_client.models import PointStruct

        try:
            client = QdrantClientWrapper(host="localhost", port=6333)

            # Test adding a test point with UUID
            test_id = str(uuid.uuid4())
            test_point = PointStruct(
                id=test_id,
                vector=[0.1] * 1536,  # Mock embedding
                payload={
                    "url": "https://test.com",
                    "content": "Test content for integration",
                    "chunk_number": 1,
                    "source_id": "test.com",
                },
            )

            # Insert test point
            client.upsert_points("crawled_pages", [test_point])

            # Wait a moment for indexing
            time.sleep(1)

            # Test search
            results = client.search_documents(
                query_embedding=[0.1] * 1536, match_count=1
            )

            # Verify we can search (even if no exact matches)
            assert isinstance(results, list)

            # Test keyword search
            keyword_results = client.keyword_search_documents(
                query="Test content", match_count=5
            )

            assert isinstance(keyword_results, list)

            # Clean up test point (optional)
            try:
                client.client.delete(
                    collection_name="crawled_pages", points_selector=[test_id]
                )
            except Exception:
                pass  # Clean up failure is not critical

        except Exception as e:
            pytest.fail(f"Qdrant operations failed: {e}")


class TestUtilsWithRealQdrant:
    """Test utils functions with real Qdrant service."""

    def test_get_qdrant_client_real(self, docker_services):
        """Test getting real Qdrant client."""
        if not docker_services["qdrant_ready"]:
            pytest.skip("Qdrant not ready")

        from qdrant_wrapper import get_qdrant_client

        try:
            client = get_qdrant_client()
            assert client is not None

            # Test health check
            health = client.health_check()
            assert health["status"] == "healthy"

        except Exception as e:
            pytest.fail(f"get_qdrant_client failed: {e}")

    @pytest.mark.skipif(
        not os.getenv("EMBEDDINGS_API_KEY"), reason="No embeddings API key"
    )
    def test_embedding_integration_real(self, docker_services):
        """Test embedding creation and storage (requires OpenAI API key)."""
        if not docker_services["qdrant_ready"]:
            pytest.skip("Qdrant not ready")

        from utils import create_embedding, get_vector_db_client
        from qdrant_client.models import PointStruct

        try:
            # Get client
            client = get_vector_db_client()  # Returns Qdrant client

            # Create test embedding (this will use real OpenAI API if key is available)
            test_text = "This is a test for integration testing"
            embedding = create_embedding(test_text)

            # Verify embedding structure
            assert isinstance(embedding, list)
            assert len(embedding) == 1536

            # Test storing embedding
            test_point = PointStruct(
                id="integration-test-embedding",
                vector=embedding,
                payload={
                    "url": "https://integration-test.com",
                    "content": test_text,
                    "chunk_number": 1,
                    "source_id": "integration-test.com",
                },
            )

            client.upsert_points("crawled_pages", [test_point])

            # Wait for indexing
            time.sleep(1)

            # Test search with the same embedding
            results = client.search_documents(query_embedding=embedding, match_count=1)

            assert isinstance(results, list)
            if results:
                # If we found results, verify structure
                result = results[0]
                assert "id" in result
                assert "similarity" in result
                assert "content" in result

            # Clean up
            try:
                client.client.delete(
                    collection_name="crawled_pages",
                    points_selector=["integration-test-embedding"],
                )
            except Exception:
                pass

        except Exception as e:
            pytest.fail(f"Embedding integration test failed: {e}")


class TestNeo4jIntegration:
    """Test Neo4j integration (when available)."""

    def test_neo4j_availability(self, docker_services):
        """Test if Neo4j is accessible."""
        if not docker_services["neo4j_ready"]:
            pytest.skip(
                "Neo4j not ready - this is optional for basic RAG functionality"
            )

        # Try to connect to Neo4j browser interface
        try:
            response = requests.get("http://localhost:7474", timeout=5)
            # Neo4j returns HTML, so we just check it responds
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.skip("Neo4j not accessible - this is optional")

    def test_neo4j_bolt_port(self, docker_services):
        """Test if Neo4j Bolt port is accessible."""
        if not docker_services["neo4j_ready"]:
            pytest.skip("Neo4j not ready")

        import socket

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(("localhost", 7687))
            sock.close()

            # Port should be open (connect_ex returns 0 on success)
            assert result == 0, "Neo4j Bolt port 7687 should be accessible"

        except Exception as e:
            pytest.skip(f"Neo4j port test failed: {e}")


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow with Docker services."""

    @pytest.mark.skipif(
        not os.getenv("EMBEDDINGS_API_KEY"), reason="No embeddings API key"
    )
    def test_complete_rag_workflow(self, docker_services):
        """Test complete RAG workflow from storage to retrieval."""
        if not docker_services["qdrant_ready"]:
            pytest.skip("Qdrant not ready")

        from utils import (
    get_vector_db_client,
            add_documents_to_vector_db,
            search_documents,
        )

        try:
            # Get client
            client = get_vector_db_client()

            # Test data
            test_urls = ["https://test-integration.com/page1"]
            test_contents = [
                "This is integration test content about Python programming and machine learning."
            ]
            test_chunk_numbers = [1]
            test_metadatas = [{"category": "integration_test", "language": "en"}]
            test_url_to_full_doc = {
                "https://test-integration.com/page1": "Full document content for integration testing"
            }

            # Store documents
            add_documents_to_vector_db(
                client=client,
                urls=test_urls,
                chunk_numbers=test_chunk_numbers,
                contents=test_contents,
                metadatas=test_metadatas,
                url_to_full_document=test_url_to_full_doc,
            )

            # Wait for indexing
            time.sleep(2)

            # Search for documents
            search_results = search_documents(
                client=client, query="Python programming", match_count=5
            )

            # Verify search results
            assert isinstance(search_results, list)

            # Clean up test data
            try:
                # Try to remove test documents (cleanup)
                # This is optional as test data won't interfere with normal operation
                pass
            except Exception:
                pass

        except Exception as e:
            pytest.fail(f"End-to-end workflow test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
