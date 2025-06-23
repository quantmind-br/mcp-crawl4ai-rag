"""
Unit tests for content processing utilities.
"""

import pytest
from unittest.mock import patch

from src.utils.content_utils import (
    smart_chunk_markdown,
    extract_section_info,
    extract_code_blocks
)


class TestContentUtils:
    """Test suite for content utility functions."""
    
    def test_smart_chunk_markdown_basic(self):
        """Test basic text chunking functionality."""
        text = "This is a test. " * 1000  # Create long text
        chunks = smart_chunk_markdown(text, chunk_size=100)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 150  # Some flexibility for sentence boundaries
    
    def test_smart_chunk_markdown_respects_paragraphs(self):
        """Test that chunking respects paragraph boundaries."""
        text = "Paragraph 1 content. " * 10 + "\n\n" + "Paragraph 2 content. " * 10
        chunks = smart_chunk_markdown(text, chunk_size=100)
        
        # Should break at paragraph boundary
        assert len(chunks) >= 2
    
    def test_smart_chunk_markdown_respects_code_blocks(self):
        """Test that chunking respects code block boundaries."""
        text = "Some text before.\n\n```python\n" + "code line\n" * 20 + "```\n\nSome text after."
        chunks = smart_chunk_markdown(text, chunk_size=100)
        
        # Code block should be preserved
        code_found = any("```" in chunk for chunk in chunks)
        assert code_found
    
    def test_smart_chunk_markdown_sentence_boundaries(self):
        """Test that chunking falls back to sentence boundaries."""
        text = "Sentence one. Sentence two. Sentence three. " * 50
        chunks = smart_chunk_markdown(text, chunk_size=100)
        
        # Should break at sentence boundaries
        assert len(chunks) > 1
        # Most chunks should end with period (except possibly the last)
        sentence_endings = sum(1 for chunk in chunks[:-1] if chunk.strip().endswith('.'))
        assert sentence_endings > 0
    
    def test_smart_chunk_markdown_empty_input(self):
        """Test chunking with empty input."""
        chunks = smart_chunk_markdown("")
        assert chunks == []
        
        chunks = smart_chunk_markdown("   ")
        assert chunks == []
    
    def test_smart_chunk_markdown_small_text(self):
        """Test chunking with text smaller than chunk size."""
        text = "Small text content."
        chunks = smart_chunk_markdown(text, chunk_size=1000)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_extract_section_info(self):
        """Test section information extraction."""
        chunk = """# Main Header
        
        Some content here with multiple words.
        
        ## Sub Header
        
        More content."""
        
        info = extract_section_info(chunk)
        
        assert "headers" in info
        assert "char_count" in info
        assert "word_count" in info
        
        assert "# Main Header" in info["headers"]
        assert "## Sub Header" in info["headers"]
        assert info["char_count"] == len(chunk)
        assert info["word_count"] > 0
    
    def test_extract_section_info_no_headers(self):
        """Test section info extraction with no headers."""
        chunk = "Just some plain text without headers."
        
        info = extract_section_info(chunk)
        
        assert info["headers"] == ""
        assert info["char_count"] == len(chunk)
        assert info["word_count"] == 6
    
    @patch.dict('os.environ', {'MIN_CODE_BLOCK_LENGTH': '50'})
    def test_extract_code_blocks_basic(self):
        """Test basic code block extraction."""
        markdown = """
        Some text before.
        
        ```python
        def hello():
            print("Hello, world!")
            return "Hello"
        
        # This is a longer code block to meet minimum length
        for i in range(100):
            print(f"Number: {i}")
        ```
        
        Some text after.
        """
        
        code_blocks = extract_code_blocks(markdown)
        
        assert len(code_blocks) >= 1
        block = code_blocks[0]
        assert "language" in block
        assert "code" in block
        assert "context_before" in block
        assert "context_after" in block
        assert block["language"] == "python"
        assert "def hello():" in block["code"]
    
    def test_extract_code_blocks_multiple(self):
        """Test extraction of multiple code blocks."""
        markdown = """
        ```python
        """ + "x = 1\n" * 50 + """
        ```
        
        Some text in between.
        
        ```javascript
        """ + "console.log('test');\n" * 50 + """
        ```
        """
        
        code_blocks = extract_code_blocks(markdown)
        
        assert len(code_blocks) == 2
        assert code_blocks[0]["language"] == "python"
        assert code_blocks[1]["language"] == "javascript"
    
    def test_extract_code_blocks_no_language(self):
        """Test code block extraction without language specifier."""
        markdown = """
        ```
        """ + "some code without language\n" * 50 + """
        ```
        """
        
        code_blocks = extract_code_blocks(markdown)
        
        assert len(code_blocks) >= 1
        assert code_blocks[0]["language"] == ""
    
    @patch.dict('os.environ', {'MIN_CODE_BLOCK_LENGTH': '1000'})
    def test_extract_code_blocks_too_short(self):
        """Test that short code blocks are filtered out."""
        markdown = """
        ```python
        print("short")
        ```
        """
        
        code_blocks = extract_code_blocks(markdown)
        
        assert len(code_blocks) == 0
    
    def test_extract_code_blocks_context(self):
        """Test that context is correctly extracted around code blocks."""
        markdown = """
        This is context before the code block.
        It has multiple sentences and paragraphs.
        
        ```python
        """ + "print('hello')\n" * 50 + """
        ```
        
        This is context after the code block.
        It also has multiple sentences.
        """
        
        code_blocks = extract_code_blocks(markdown)
        
        assert len(code_blocks) >= 1
        block = code_blocks[0]
        
        assert "context before" in block["context_before"].lower()
        assert "context after" in block["context_after"].lower()
        assert len(block["context_before"]) <= 1000
        assert len(block["context_after"]) <= 1000
    
    def test_extract_code_blocks_wrapped_content(self):
        """Test handling of content wrapped in backticks."""
        markdown = """```
        This entire content is wrapped in backticks.
        
        ```python
        """ + "actual_code = True\n" * 50 + """
        ```
        
        More wrapped content.
        ```"""
        
        code_blocks = extract_code_blocks(markdown)
        
        # Should find the python block despite the wrapping
        python_blocks = [b for b in code_blocks if b["language"] == "python"]
        assert len(python_blocks) >= 1