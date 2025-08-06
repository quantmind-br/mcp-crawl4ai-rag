"""

Integration tests for utils.py with Qdrant wrapper.

Tests the integration between utils functions and QdrantClientWrapper.
"""
# ruff: noqa: E402

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from utils import (
    get_vector_db_client,
    search_documents,
    search_code_examples,
    add_documents_to_vector_db,
    add_code_examples_to_vector_db,
    create_embedding,
    create_embeddings_batch,
    extract_code_blocks,
    update_source_info,
)
from embedding_config import get_embedding_dimensions
from qdrant_wrapper import QdrantClientWrapper


class TestUtilsIntegration:
    """Test integration between utils and Qdrant wrapper."""

    @patch("utils.get_qdrant_client")
    def test_get_vector_db_client_returns_qdrant(self, mock_get_qdrant):
        """Test that get_vector_db_client returns Qdrant client (legacy compatibility)."""

        # Setup
        mock_client = Mock(spec=QdrantClientWrapper)
        mock_get_qdrant.return_value = mock_client

        # Test
        client = get_vector_db_client()

        # Verify
        assert client == mock_client
        mock_get_qdrant.assert_called_once()

    @patch("utils.create_embedding")
    @patch("utils.QdrantClientWrapper")
    def test_search_documents_integration(
        self, mock_wrapper_class, mock_create_embedding
    ):
        """Test document search integration with Qdrant."""

        # Setup mocks
        mock_client = Mock()
        mock_client.search_documents.return_value = [
            {"id": "doc1", "similarity": 0.9, "content": "test content"}
        ]
        mock_wrapper_class.return_value = mock_client

        mock_create_embedding.return_value = [0.1] * get_embedding_dimensions()

        # Test
        results = search_documents(mock_client, "test query", match_count=5)

        # Verify
        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        mock_create_embedding.assert_called_once_with("test query")
        mock_client.search_documents.assert_called_once()

    @patch("utils.create_embedding")
    @patch("utils.QdrantClientWrapper")
    def test_search_code_examples_integration(
        self, mock_wrapper_class, mock_create_embedding
    ):
        """Test code examples search integration with Qdrant."""

        # Setup mocks
        mock_client = Mock()
        mock_client.search_code_examples.return_value = [
            {"id": "code1", "similarity": 0.85, "content": "def test(): pass"}
        ]
        mock_wrapper_class.return_value = mock_client

        mock_create_embedding.return_value = [0.1] * get_embedding_dimensions()

        # Test
        results = search_code_examples(
            mock_client, "function definition", match_count=3
        )

        # Verify
        assert len(results) == 1
        assert results[0]["id"] == "code1"
        mock_create_embedding.assert_called_once()
        # Enhanced query should be used
        call_args = mock_create_embedding.call_args[0][0]
        assert "Code example for" in call_args

    @patch("utils.create_embeddings_batch")
    @patch("utils.QdrantClientWrapper")
    def test_add_documents_integration(
        self, mock_wrapper_class, mock_create_embeddings
    ):
        """Test document addition integration with Qdrant."""

        # Setup mocks
        mock_client = Mock()
        mock_client.add_documents_to_qdrant.return_value = [
            [{"id": "doc1", "payload": {"content": "test"}, "content": "test"}]
        ]
        mock_client.upsert_points.return_value = None
        mock_wrapper_class.return_value = mock_client

        mock_create_embeddings.return_value = [[0.1] * get_embedding_dimensions()]

        # Test data
        urls = ["https://example.com"]
        chunk_numbers = [1]
        contents = ["test content"]
        metadatas = [{"category": "test"}]
        url_to_full_document = {"https://example.com": "full document"}

        # Test
        add_documents_to_vector_db(
            mock_client, urls, chunk_numbers, contents, metadatas, url_to_full_document
        )

        # Verify
        mock_client.add_documents_to_qdrant.assert_called_once()
        mock_create_embeddings.assert_called_once()
        mock_client.upsert_points.assert_called_once()

    @patch("utils.create_embeddings_batch")
    @patch("utils.QdrantClientWrapper")
    def test_add_code_examples_integration(
        self, mock_wrapper_class, mock_create_embeddings
    ):
        """Test code examples addition integration with Qdrant."""

        # Setup mocks
        mock_client = Mock()
        mock_client.add_code_examples_to_qdrant.return_value = [
            [
                {
                    "id": "code1",
                    "payload": {"content": "code"},
                    "combined_text": "code\n\nSummary: test",
                }
            ]
        ]
        mock_client.upsert_points.return_value = None
        mock_wrapper_class.return_value = mock_client

        mock_create_embeddings.return_value = [[0.1] * get_embedding_dimensions()]

        # Test data
        urls = ["https://example.com"]
        chunk_numbers = [1]
        code_examples = ["def test(): pass"]
        summaries = ["Test function"]
        metadatas = [{"language": "python"}]

        # Test
        add_code_examples_to_vector_db(
            mock_client, urls, chunk_numbers, code_examples, summaries, metadatas
        )

        # Verify
        mock_client.add_code_examples_to_qdrant.assert_called_once()
        mock_create_embeddings.assert_called_once()
        mock_client.upsert_points.assert_called_once()

    @patch("utils.QdrantClientWrapper")
    def test_update_source_info_integration(self, mock_wrapper_class):
        """Test source info update integration."""

        # Setup mock
        mock_client = Mock()
        mock_client.update_source_info.return_value = None
        mock_wrapper_class.return_value = mock_client

        # Test
        update_source_info(mock_client, "example.com", "Test site", 1000)

        # Verify
        mock_client.update_source_info.assert_called_once_with(
            "example.com", "Test site", 1000
        )


class TestEmbeddingFunctions:
    """Test embedding creation functions."""

    @patch("utils.openai.embeddings.create")
    def test_create_embeddings_batch_success(self, mock_openai_create):
        """Test successful batch embedding creation."""

        # Setup mock
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * get_embedding_dimensions()),
            Mock(embedding=[0.2] * get_embedding_dimensions()),
        ]
        mock_openai_create.return_value = mock_response

        # Test
        texts = ["text1", "text2"]
        embeddings = create_embeddings_batch(texts)

        # Verify
        assert len(embeddings) == 2
        assert len(embeddings[0]) == get_embedding_dimensions()
        assert len(embeddings[1]) == get_embedding_dimensions()
        mock_openai_create.assert_called_once_with(
            model="text-embedding-3-small", input=texts
        )

    @patch("utils.time.sleep")  # Mock sleep to speed up test
    @patch("utils.openai.embeddings.create")
    def test_create_embeddings_batch_failure_with_fallback(
        self, mock_openai_create, mock_sleep
    ):
        """Test batch embedding creation with fallback to individual."""

        # Setup mock to fail on all batch attempts (3 retries), then succeed on individual calls
        mock_openai_create.side_effect = [
            Exception("Batch failed"),  # 1st batch attempt fails
            Exception("Batch failed"),  # 2nd batch attempt fails
            Exception("Batch failed"),  # 3rd batch attempt fails
            Mock(
                data=[Mock(embedding=[0.1] * get_embedding_dimensions())]
            ),  # 1st individual call succeeds
            Mock(
                data=[Mock(embedding=[0.2] * get_embedding_dimensions())]
            ),  # 2nd individual call succeeds
        ]

        # Test
        texts = ["text1", "text2"]
        embeddings = create_embeddings_batch(texts)

        # Verify
        assert len(embeddings) == 2
        assert len(embeddings[0]) == get_embedding_dimensions()
        assert len(embeddings[1]) == get_embedding_dimensions()
        assert mock_openai_create.call_count == 5  # 3 batch attempts + 2 individual
        assert mock_sleep.call_count == 2  # Called for retry delays

    @patch("utils.create_embeddings_batch")
    def test_create_embedding_single(self, mock_batch):
        """Test single embedding creation."""

        # Setup mock
        mock_batch.return_value = [[0.1] * get_embedding_dimensions()]

        # Test
        embedding = create_embedding("test text")

        # Verify
        assert len(embedding) == get_embedding_dimensions()
        mock_batch.assert_called_once_with(["test text"])

    @patch("utils.create_embeddings_batch")
    def test_create_embedding_failure(self, mock_batch):
        """Test single embedding creation failure."""

        # Setup mock to fail
        mock_batch.side_effect = Exception("Failed")

        # Test
        embedding = create_embedding("test text")

        # Verify fallback to zero embedding
        assert len(embedding) == get_embedding_dimensions()
        assert all(v == 0.0 for v in embedding)


class TestCodeExtraction:
    """Test code block extraction."""

    def test_extract_code_blocks_basic(self):
        """Test basic code block extraction."""

        markdown = """

        Some text before
        
        ```python
        def hello():
            print("world")
        ```
        
        Some text after
        """

        blocks = extract_code_blocks(markdown, min_length=10)

        assert len(blocks) == 1
        block = blocks[0]
        assert block["language"] == "python"
        assert "def hello():" in block["code"]
        assert "Some text before" in block["context_before"]
        assert "Some text after" in block["context_after"]

    def test_extract_code_blocks_no_language(self):
        """Test code block extraction without language specifier."""

        markdown = """

        ```
        function test() {
            console.log("test");
        }
        ```
        """

        blocks = extract_code_blocks(markdown, min_length=10)

        assert len(blocks) == 1
        assert blocks[0]["language"] == ""
        assert "function test()" in blocks[0]["code"]

    def test_extract_code_blocks_min_length_filter(self):
        """Test that short code blocks are filtered out."""

        markdown = """

        ```python
        x = 1
        ```
        """

        blocks = extract_code_blocks(markdown, min_length=1000)

        assert len(blocks) == 0  # Block too short


if __name__ == "__main__":
    pytest.main([__file__])
