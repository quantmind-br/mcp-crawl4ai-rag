"""
Tests for core context dataclass and dependency injection.
"""

import sys
from pathlib import Path
from unittest.mock import Mock
from dataclasses import fields

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestCrawl4AIContext:
    """Test Crawl4AIContext dataclass."""

    def test_context_dataclass_creation(self):
        """Test that context can be created with required fields."""
        from src.core.context import Crawl4AIContext

        mock_crawler = Mock()
        mock_client = Mock()
        mock_cache = Mock()

        context = Crawl4AIContext(
            crawler=mock_crawler, qdrant_client=mock_client, embedding_cache=mock_cache
        )

        assert context.crawler is mock_crawler
        assert context.qdrant_client is mock_client
        assert context.embedding_cache is mock_cache

    def test_context_optional_fields(self):
        """Test that optional fields default to None."""
        from src.core.context import Crawl4AIContext

        context = Crawl4AIContext(
            crawler=Mock(), qdrant_client=Mock(), embedding_cache=Mock()
        )

        # Optional fields should default to None
        assert context.reranker is None
        assert context.knowledge_validator is None
        assert context.repo_extractor is None
        assert context.embedding_service is None
        assert context.rag_service is None

    def test_context_all_fields_provided(self):
        """Test context with all fields provided."""
        from src.core.context import Crawl4AIContext

        mocks = {
            "crawler": Mock(),
            "qdrant_client": Mock(),
            "embedding_cache": Mock(),
            "reranker": Mock(),
            "knowledge_validator": Mock(),
            "repo_extractor": Mock(),
            "embedding_service": Mock(),
            "rag_service": Mock(),
        }

        context = Crawl4AIContext(**mocks)

        for field_name, mock_obj in mocks.items():
            assert getattr(context, field_name) is mock_obj

    def test_context_dataclass_fields(self):
        """Test that context has expected fields with correct types."""
        from src.core.context import Crawl4AIContext

        expected_fields = {
            "crawler",
            "qdrant_client",
            "embedding_cache",
            "io_executor",
            "cpu_executor", 
            "performance_config",
            "reranker",
            "knowledge_validator",
            "repo_extractor",
            "embedding_service",
            "rag_service",
        }

        actual_fields = {field.name for field in fields(Crawl4AIContext)}
        assert actual_fields == expected_fields

    def test_context_repr(self):
        """Test context string representation."""
        from src.core.context import Crawl4AIContext

        context = Crawl4AIContext(
            crawler=Mock(), qdrant_client=Mock(), embedding_cache=Mock()
        )

        repr_str = repr(context)
        assert "Crawl4AIContext" in repr_str
        assert "crawler=" in repr_str
        assert "qdrant_client=" in repr_str

    def test_context_immutability(self):
        """Test that context fields can be modified (not frozen)."""
        from src.core.context import Crawl4AIContext

        context = Crawl4AIContext(
            crawler=Mock(), qdrant_client=Mock(), embedding_cache=Mock()
        )

        # Should be able to modify fields (context is not frozen)
        new_mock = Mock()
        context.crawler = new_mock
        assert context.crawler is new_mock

    def test_context_field_access(self):
        """Test accessing context fields through attribute access."""
        from src.core.context import Crawl4AIContext

        mock_crawler = Mock()
        mock_client = Mock()
        mock_cache = Mock()

        context = Crawl4AIContext(
            crawler=mock_crawler, qdrant_client=mock_client, embedding_cache=mock_cache
        )

        # Test attribute access works
        assert hasattr(context, "crawler")
        assert hasattr(context, "qdrant_client")
        assert hasattr(context, "embedding_cache")
        assert hasattr(context, "reranker")

        # Test getting attributes
        assert getattr(context, "crawler") is mock_crawler
        assert getattr(context, "qdrant_client") is mock_client
        assert getattr(context, "embedding_cache") is mock_cache
        assert getattr(context, "reranker") is None

    def test_context_type_hints(self):
        """Test that context has proper type hints."""
        from src.core.context import Crawl4AIContext

        # This test ensures the dataclass is properly typed
        # The actual type checking would be done by mypy/pyright
        annotations = getattr(Crawl4AIContext, "__annotations__", {})

        # Should have type annotations for all fields
        expected_annotations = {
            "crawler",
            "qdrant_client",
            "embedding_cache",
            "io_executor",
            "cpu_executor",
            "performance_config",
            "reranker",
            "knowledge_validator",
            "repo_extractor",
            "embedding_service",
            "rag_service",
        }

        actual_annotations = set(annotations.keys())
        assert actual_annotations == expected_annotations
