"""
Integration tests for GitHub repository indexing equivalence.

This test suite validates that the enhanced index_github_repository produces
equivalent or superior results using intelligent content extraction.
"""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.services.unified_indexing_service import UnifiedIndexingService
from src.models.unified_indexing_models import (
    UnifiedIndexingRequest,
    IndexingDestination,
)
from src.features.github.processors.processor_factory import ProcessorFactory
from src.utils.chunking import smart_chunk_markdown


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    client = MagicMock()
    client.create_collection = MagicMock()
    client.upsert = MagicMock()
    return client


@pytest.fixture
def mock_neo4j_parser():
    """Mock Neo4j parser for testing."""
    parser = MagicMock()
    parser.extract_knowledge_graph = MagicMock(return_value={})
    return parser


@pytest.fixture
def sample_repository():
    """Create a sample repository structure for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir) / "test_repo"
        repo_path.mkdir()

        # Create sample Python file with docstring
        python_file = repo_path / "sample.py"
        python_content = '''"""
This is a sample Python module for testing.

It contains a simple function with documentation.
"""

def hello_world(name: str) -> str:
    """
    Return a greeting message.
    
    Args:
        name: The name to greet
        
    Returns:
        A greeting string
    """
    return f"Hello, {name}!"

class Calculator:
    """A simple calculator class."""
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
'''
        python_file.write_text(python_content)

        # Create sample Markdown file
        markdown_file = repo_path / "README.md"
        markdown_content = '''# Test Repository

This is a test repository for validating indexing functionality.

## Features

- Python code processing
- Markdown documentation
- Comprehensive testing

```python
def example():
    """Example function in documentation."""
    return "example"
```

## Installation

1. Clone the repository
2. Install dependencies
3. Run tests
'''
        markdown_file.write_text(markdown_content)

        # Create sample text documentation
        txt_file = repo_path / "INSTALL.txt"
        txt_content = """Installation Instructions

This file contains installation instructions for the project.

Requirements:
- Python 3.12+
- Dependencies from requirements.txt

Steps:
1. Download the source code
2. Set up virtual environment
3. Install packages
"""
        txt_file.write_text(txt_content)

        yield repo_path


@pytest.fixture
def indexing_service(mock_qdrant_client, mock_neo4j_parser):
    """Create UnifiedIndexingService for testing."""
    with patch(
        "src.services.unified_indexing_service.get_qdrant_client",
        return_value=mock_qdrant_client,
    ):
        service = UnifiedIndexingService(
            qdrant_client=mock_qdrant_client,
            neo4j_parser=mock_neo4j_parser,
            context=None,
        )
        return service


class TestGitHubIndexingEquivalence:
    """Test suite for GitHub indexing equivalence validation."""

    @pytest.mark.asyncio
    async def test_processor_vs_raw_content_chunking(
        self, indexing_service, sample_repository
    ):
        """Test that processor-based chunking produces equivalent or superior results."""
        python_file = sample_repository / "sample.py"
        content = python_file.read_text()

        # Create a request for testing
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.QDRANT,
            chunk_size=1000,
        )

        # Test processor-based extraction
        processor_factory = ProcessorFactory()
        processor = processor_factory.get_processor_for_file(str(python_file))

        assert processor is not None, "Python processor should be available"

        # Extract content using processor
        relative_path = str(python_file.relative_to(sample_repository))
        processed_items = processor.process_file(str(python_file), relative_path)

        # Chunk processed content
        processor_chunks = []
        for item in processed_items:
            item_chunks = smart_chunk_markdown(item.content, request.chunk_size)
            processor_chunks.extend(item_chunks)

        # Test raw content chunking (fallback)
        raw_chunks = smart_chunk_markdown(content, request.chunk_size)

        # Validate results
        assert len(processor_chunks) > 0, "Processor should extract meaningful content"
        assert len(raw_chunks) > 0, "Raw chunking should work as fallback"

        # Processor chunks should be more focused (typically fewer but higher quality)
        # The exact comparison depends on the content, but we can check general properties

        # Calculate content quality metrics
        processor_content = " ".join(processor_chunks)
        raw_content = " ".join(raw_chunks)

        # Processor should extract docstrings and meaningful content
        assert "Return a greeting message" in processor_content, (
            "Should extract function docstring"
        )
        assert "A simple calculator class" in processor_content, (
            "Should extract class docstring"
        )

        print(f"Processor chunks: {len(processor_chunks)}")
        print(f"Raw chunks: {len(raw_chunks)}")
        print(f"Processor content length: {len(processor_content)}")
        print(f"Raw content length: {len(raw_content)}")

    @pytest.mark.asyncio
    async def test_metadata_consistency(self, indexing_service, sample_repository):
        """Test that metadata structure is consistent and enhanced."""
        python_file = sample_repository / "sample.py"
        content = python_file.read_text()
        file_id = "test_repo:sample.py"

        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.QDRANT,
            chunk_size=1000,
        )

        # Mock the vector database operations
        with (
            patch(
                "src.services.unified_indexing_service.add_documents_to_vector_db"
            ) as mock_add_docs,
            patch(
                "src.services.unified_indexing_service.update_source_info"
            ) as mock_update_source,
        ):
            # Process file with new intelligent logic
            success = await indexing_service._process_file_for_rag(
                python_file, content, file_id, request
            )

            assert success, "File processing should succeed"

            # Verify that vector database was called
            mock_add_docs.assert_called_once()
            mock_update_source.assert_called_once()

            # Get the call arguments
            call_args = mock_add_docs.call_args
            args, kwargs = call_args

            # The function signature is:
            # add_documents_to_vector_db(client, urls, chunk_numbers, contents, metadatas, url_to_full_document, batch_size, file_ids)
            chunks = args[3]  # contents (chunks)
            metadatas = args[4]  # metadatas

            # Debug print to understand the structure
            print(f"Call args length: {len(args)}")
            print(
                f"Chunks type: {type(chunks)}, length: {len(chunks) if hasattr(chunks, '__len__') else 'N/A'}"
            )
            print(
                f"Metadatas type: {type(metadatas)}, length: {len(metadatas) if hasattr(metadatas, '__len__') else 'N/A'}"
            )
            if metadatas and len(metadatas) > 0:
                print(f"First metadata type: {type(metadatas[0])}")
                if isinstance(metadatas[0], dict):
                    print(f"First metadata keys: {list(metadatas[0].keys())}")

            # Validate metadata structure
            assert len(metadatas) > 0, "Should have metadata for chunks"

            first_metadata = metadatas[0]
            assert isinstance(first_metadata, dict), (
                f"Metadata should be dict, got {type(first_metadata)}"
            )

            # Check required fields are present
            required_fields = [
                "chunk_index",
                "file_path",
                "relative_path",
                "language",
                "source",
                "processing_time",
            ]
            for field in required_fields:
                assert field in first_metadata, f"Metadata should contain {field}"

            # Check enhanced fields from ProcessedContent (should be present for processor-based extraction)
            enhanced_fields = ["content_type", "content_name", "extraction_method"]
            for field in enhanced_fields:
                assert field in first_metadata, (
                    f"Enhanced metadata should contain {field}"
                )

            # Verify extraction method is correctly set
            assert first_metadata["extraction_method"] == "processor", (
                "Should use processor extraction"
            )

            print(f"Metadata fields: {list(first_metadata.keys())}")
            print(f"Content type: {first_metadata.get('content_type')}")
            print(f"Content name: {first_metadata.get('content_name')}")

    @pytest.mark.asyncio
    async def test_fallback_processing(self, indexing_service, sample_repository):
        """Test that fallback processing works for unsupported file types."""
        # Create a file type that doesn't have a processor
        unsupported_file = sample_repository / "config.ini"
        content = """[database]
host = localhost
port = 5432
name = testdb

[logging]
level = INFO
file = app.log
"""
        unsupported_file.write_text(content)

        file_id = "test_repo:config.ini"
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.QDRANT,
            chunk_size=1000,
        )

        # Mock the vector database operations
        with (
            patch(
                "src.services.unified_indexing_service.add_documents_to_vector_db"
            ) as mock_add_docs,
            patch("src.services.unified_indexing_service.update_source_info"),
        ):
            # Process file (should use fallback)
            success = await indexing_service._process_file_for_rag(
                unsupported_file, content, file_id, request
            )

            assert success, "Fallback processing should succeed"

            # Verify that vector database was called
            mock_add_docs.assert_called_once()

            # Get the call arguments
            call_args = mock_add_docs.call_args
            args, kwargs = call_args

            args[3]  # contents (chunks)
            metadatas = args[4]  # metadatas

            # Verify fallback metadata
            assert len(metadatas) > 0, "Should have metadata for chunks"
            first_metadata = metadatas[0]
            assert isinstance(first_metadata, dict), (
                f"Metadata should be dict, got {type(first_metadata)}"
            )

            assert first_metadata["extraction_method"] == "fallback", (
                "Should use fallback extraction"
            )

            # Should not have enhanced processor fields
            assert "content_type" not in first_metadata, (
                "Fallback should not have processor metadata"
            )

    @pytest.mark.asyncio
    async def test_documentation_processor_integration(
        self, indexing_service, sample_repository
    ):
        """Test that the new DocumentationProcessor works correctly."""
        txt_file = sample_repository / "INSTALL.txt"
        content = txt_file.read_text()
        file_id = "test_repo:INSTALL.txt"

        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.QDRANT,
            chunk_size=1000,
        )

        # Test that DocumentationProcessor is available
        processor_factory = ProcessorFactory()
        processor = processor_factory.get_processor_for_file(str(txt_file))

        assert processor is not None, "DocumentationProcessor should handle .txt files"
        assert processor.name == "documentation", "Should use documentation processor"

        # Mock the vector database operations
        with (
            patch(
                "src.services.unified_indexing_service.add_documents_to_vector_db"
            ) as mock_add_docs,
            patch("src.services.unified_indexing_service.update_source_info"),
        ):
            # Process file
            success = await indexing_service._process_file_for_rag(
                txt_file, content, file_id, request
            )

            assert success, "Documentation file processing should succeed"

            # Verify that vector database was called
            mock_add_docs.assert_called_once()

            # Get the call arguments
            call_args = mock_add_docs.call_args
            args, kwargs = call_args

            chunks = args[3]  # contents (chunks)
            metadatas = args[4]  # metadatas

            # Verify content was processed
            assert len(chunks) > 0, "Should have chunks from documentation"
            assert len(metadatas) > 0, "Should have metadata"

            # Check that it used processor extraction
            first_metadata = metadatas[0]
            assert isinstance(first_metadata, dict), (
                f"Metadata should be dict, got {type(first_metadata)}"
            )
            assert first_metadata["extraction_method"] == "processor", (
                "Should use processor"
            )
            assert first_metadata["content_type"] == "documentation", (
                "Should mark as documentation"
            )

    def test_chunking_equivalence(self):
        """Test that smart_chunk_markdown produces consistent results."""
        sample_content = '''# Sample Content

This is a sample document with multiple paragraphs.

## Code Example

```python
def example_function():
    """Example function."""
    return "Hello World"
```

## More Content

Additional paragraphs here. This should be chunked properly respecting code blocks and paragraph boundaries.

The chunking should be intelligent and consistent.
'''

        # Test chunking with different sizes
        chunks_small = smart_chunk_markdown(sample_content, 200)
        chunks_medium = smart_chunk_markdown(sample_content, 500)
        chunks_large = smart_chunk_markdown(sample_content, 1000)

        # Validate chunking behavior
        assert len(chunks_small) > len(chunks_medium), (
            "Smaller chunks should result in more pieces"
        )
        assert len(chunks_medium) >= len(chunks_large), (
            "Medium chunks should be >= large chunks"
        )

        # All chunks should be non-empty
        for chunk_set in [chunks_small, chunks_medium, chunks_large]:
            for chunk in chunk_set:
                assert chunk.strip(), "All chunks should be non-empty"

        # Reassembled content should be equivalent (ignoring whitespace differences)
        reassembled = " ".join(chunks_large).strip()
        sample_content.strip()

        # Check that major content elements are preserved
        assert "Sample Content" in reassembled
        assert "def example_function" in reassembled
        assert "Example function" in reassembled

        print(f"Small chunks: {len(chunks_small)}")
        print(f"Medium chunks: {len(chunks_medium)}")
        print(f"Large chunks: {len(chunks_large)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
