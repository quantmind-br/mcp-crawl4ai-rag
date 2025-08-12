"""
Tests for Web tools.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from src.tools.web_tools import (
    extract_source_summary,
    extract_code_blocks,
    generate_code_example_summary,
    is_sitemap,
    is_txt,
    parse_sitemap,
    smart_chunk_markdown,
    extract_section_info,
    process_code_example,
    crawl_markdown_file,
    crawl_batch,
    crawl_recursive_internal_links,
    crawl_single_page,
    create_base_prefix,
    smart_crawl_url,
)


class TestWebTools:
    """Test cases for Web tools."""

    def test_extract_source_summary_empty_content(self):
        """Test extract_source_summary with empty content."""
        result = extract_source_summary("test-source", "")
        assert result == "Empty source: test-source"

    def test_extract_source_summary_short_content(self):
        """Test extract_source_summary with short content."""
        content = "This is a short summary."
        result = extract_source_summary("test-source", content)
        assert result == "This is a short summary."

    def test_extract_source_summary_long_content(self):
        """Test extract_source_summary with long content."""
        content = "A" * 600  # 600 characters
        result = extract_source_summary("test-source", content, max_length=500)
        assert len(result) <= 503  # 500 + "..."
        assert result.endswith("...")

    def test_extract_code_blocks_no_blocks(self):
        """Test extract_code_blocks with no code blocks."""
        content = "This is plain text without code blocks."
        result = extract_code_blocks(content)
        assert len(result) == 0

    def test_extract_code_blocks_with_blocks(self):
        """Test extract_code_blocks with code blocks."""
        content = """Some text before.

```python
def example():
    return "code"
```

Some text after."""

        result = extract_code_blocks(
            content, min_length=0
        )  # Lower threshold for testing
        assert len(result) == 1
        assert result[0]["language"] == "python"
        assert "def example():" in result[0]["code"]

    def test_extract_code_blocks_with_context(self):
        """Test extract_code_blocks with context extraction."""
        content = """# Title
Some introductory text.

```python
def example():
    return "code"
```

More text after."""

        result = extract_code_blocks(content, min_length=0)
        assert len(result) == 1
        assert "# Title" in result[0]["context_before"]
        assert "More text after" in result[0]["context_after"]

    def test_generate_code_example_summary_with_function(self):
        """Test generate_code_example_summary with function definition."""
        code = "def example_function():\n    return 'test'"
        result = generate_code_example_summary(code)
        assert "Python code: def example_function()" in result

    def test_generate_code_example_summary_with_context(self):
        """Test generate_code_example_summary with context."""
        code = "console.log('hello');"
        context_before = "JavaScript example:"
        result = generate_code_example_summary(code, context_before=context_before)
        assert "JavaScript code example:" in result

    def test_is_sitemap_true(self):
        """Test is_sitemap with sitemap URL."""
        assert is_sitemap("https://example.com/sitemap.xml") is True

    def test_is_sitemap_false(self):
        """Test is_sitemap with non-sitemap URL."""
        assert is_sitemap("https://example.com/page.html") is False

    def test_is_txt_true(self):
        """Test is_txt with text file URL."""
        assert is_txt("https://example.com/file.txt") is True

    def test_is_txt_false(self):
        """Test is_txt with non-text file URL."""
        assert is_txt("https://example.com/file.html") is False

    @patch("src.tools.web_tools.requests.get")
    def test_parse_sitemap_success(self, mock_get):
        """Test parse_sitemap with successful response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
   <url>
      <loc>https://example.com/page1</loc>
   </url>
   <url>
      <loc>https://example.com/page2</loc>
   </url>
</urlset>"""
        mock_get.return_value = mock_response

        result = parse_sitemap("https://example.com/sitemap.xml")
        assert len(result) == 2
        assert "https://example.com/page1" in result
        assert "https://example.com/page2" in result

    @patch("src.tools.web_tools.requests.get")
    def test_parse_sitemap_failure(self, mock_get):
        """Test parse_sitemap with failed response."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = parse_sitemap("https://example.com/sitemap.xml")
        assert len(result) == 0

    def test_smart_chunk_markdown_small_text(self):
        """Test smart_chunk_markdown with small text."""
        text = "This is a small text that should fit in one chunk."
        result = smart_chunk_markdown(text, chunk_size=100)
        assert len(result) == 1
        assert result[0] == text

    def test_smart_chunk_markdown_large_text(self):
        """Test smart_chunk_markdown with large text."""
        text = "A" * 250  # 250 characters
        result = smart_chunk_markdown(text, chunk_size=100)
        assert len(result) == 3
        assert all(len(chunk) <= 100 for chunk in result)

    def test_smart_chunk_markdown_with_code_blocks(self):
        """Test smart_chunk_markdown with code blocks."""
        text = """This is some text.

```python
def example():
    return "code block"
```

More text after code block."""

        result = smart_chunk_markdown(text, chunk_size=50)
        assert len(result) >= 1

    def test_extract_section_info_with_headers(self):
        """Test extract_section_info with headers."""
        chunk = """# Main Title
## Subtitle
This is content with headers."""

        result = extract_section_info(chunk)
        assert "Main Title" in result["headers"]
        assert "Subtitle" in result["headers"]
        assert result["char_count"] == len(chunk)
        assert result["word_count"] > 0

    def test_process_code_example(self):
        """Test process_code_example function."""
        args = ("def example():\n    return 'test'", "Python example:", "")
        result = process_code_example(args)
        assert "Python code: def example()" in result

    @pytest.mark.asyncio
    async def test_crawl_markdown_file_success(self):
        """Test crawl_markdown_file with successful crawl."""
        mock_crawler = AsyncMock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.markdown = "Test markdown content"
        mock_crawler.arun.return_value = mock_result

        result = await crawl_markdown_file(mock_crawler, "https://example.com/test.txt")
        assert len(result) == 1
        assert result[0]["url"] == "https://example.com/test.txt"
        assert result[0]["markdown"] == "Test markdown content"

    @pytest.mark.asyncio
    async def test_crawl_markdown_file_failure(self):
        """Test crawl_markdown_file with failed crawl."""
        mock_crawler = AsyncMock()
        mock_result = Mock()
        mock_result.success = False
        mock_result.error_message = "Crawl failed"
        mock_crawler.arun.return_value = mock_result

        result = await crawl_markdown_file(mock_crawler, "https://example.com/test.txt")
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_crawl_batch_success(self):
        """Test crawl_batch with successful crawls."""
        mock_crawler = AsyncMock()
        mock_result1 = Mock()
        mock_result1.success = True
        mock_result1.markdown = "Content 1"
        mock_result1.url = "https://example.com/page1"

        mock_result2 = Mock()
        mock_result2.success = True
        mock_result2.markdown = "Content 2"
        mock_result2.url = "https://example.com/page2"

        mock_crawler.arun_many.return_value = [mock_result1, mock_result2]

        result = await crawl_batch(
            mock_crawler, ["https://example.com/page1", "https://example.com/page2"]
        )
        assert len(result) == 2
        assert result[0]["url"] == "https://example.com/page1"
        assert result[1]["url"] == "https://example.com/page2"

    @pytest.mark.asyncio
    async def test_crawl_recursive_internal_links(self):
        """Test crawl_recursive_internal_links."""
        mock_crawler = AsyncMock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.markdown = "Page content"
        mock_result.url = "https://example.com/page1"
        mock_result.links = {
            "internal": [
                {"href": "https://example.com/page2"},
                {"href": "https://example.com/page3"},
            ]
        }

        mock_crawler.arun_many.return_value = [mock_result]

        result = await crawl_recursive_internal_links(
            mock_crawler,
            ["https://example.com/page1"],
            "https://example.com/",
            max_depth=1,
        )
        assert len(result) == 1
        assert result[0]["url"] == "https://example.com/page1"

    @pytest.mark.asyncio
    async def test_crawl_single_page_success(self):
        """Test crawl_single_page with successful execution."""
        # Mock context and its components
        mock_context = Mock()
        mock_context.request_context = Mock()
        mock_context.request_context.lifespan_context = Mock()

        mock_crawler = AsyncMock()
        mock_qdrant_client = Mock()

        mock_context.request_context.lifespan_context.crawler = mock_crawler
        mock_context.request_context.lifespan_context.qdrant_client = mock_qdrant_client

        # Mock crawler result
        mock_result = Mock()
        mock_result.success = True
        mock_result.markdown = "# Test Page\n\nThis is test content."
        mock_result.error_message = None
        mock_result.links = {"internal": [], "external": []}
        # Mock the media attribute that might be accessed
        mock_result.media = {"images": [], "videos": [], "audios": []}
        mock_crawler.arun.return_value = mock_result

        # Mock the add_documents_to_vector_db function
        with patch("src.tools.web_tools.add_documents_to_vector_db"):
            with patch("src.tools.web_tools.update_source_info"):
                result = await crawl_single_page(
                    mock_context, "https://example.com/test"
                )
                result_data = json.loads(result)

                assert result_data["success"] is True
                assert result_data["url"] == "https://example.com/test"
                assert result_data["chunks_stored"] > 0

    @pytest.mark.asyncio
    async def test_crawl_single_page_failure(self):
        """Test crawl_single_page with failed execution."""
        # Mock context and its components
        mock_context = Mock()
        mock_context.request_context = Mock()
        mock_context.request_context.lifespan_context = Mock()

        mock_crawler = AsyncMock()
        mock_context.request_context.lifespan_context.crawler = mock_crawler

        # Mock crawler result with failure
        mock_result = Mock()
        mock_result.success = False
        mock_result.error_message = "Crawl failed"
        mock_crawler.arun.return_value = mock_result

        result = await crawl_single_page(mock_context, "https://example.com/test")
        result_data = json.loads(result)

        assert result_data["success"] is False
        assert "Crawl failed" in result_data["error"]

    def test_create_base_prefix_root(self):
        """Test create_base_prefix with root URL."""
        result = create_base_prefix("https://example.com")
        assert result == "https://example.com/"

    def test_create_base_prefix_with_path(self):
        """Test create_base_prefix with path."""
        result = create_base_prefix("https://example.com/docs")
        assert result == "https://example.com/docs/"

    def test_create_base_prefix_with_trailing_slash(self):
        """Test create_base_prefix with trailing slash."""
        result = create_base_prefix("https://example.com/docs/")
        assert result == "https://example.com/docs/"

    @pytest.mark.asyncio
    async def test_smart_crawl_url_txt_file(self):
        """Test smart_crawl_url with text file."""
        # Mock context and its components
        mock_context = Mock()
        mock_context.request_context = Mock()
        mock_context.request_context.lifespan_context = Mock()

        mock_crawler = AsyncMock()
        mock_qdrant_client = Mock()

        mock_context.request_context.lifespan_context.crawler = mock_crawler
        mock_context.request_context.lifespan_context.qdrant_client = mock_qdrant_client

        # Mock crawl_markdown_file result
        with patch("src.tools.web_tools.crawl_markdown_file") as mock_crawl_file:
            with patch("src.tools.web_tools.add_documents_to_vector_db"):
                with patch("src.tools.web_tools.update_source_info"):
                    mock_crawl_file.return_value = [
                        {
                            "url": "https://example.com/test.txt",
                            "markdown": "Test content",
                        }
                    ]

                    result = await smart_crawl_url(
                        mock_context, "https://example.com/test.txt"
                    )
                    result_data = json.loads(result)

                    assert result_data["success"] is True
                    assert result_data["url"] == "https://example.com/test.txt"
                    assert result_data["crawl_type"] == "text_file"

    @pytest.mark.asyncio
    async def test_smart_crawl_url_sitemap(self):
        """Test smart_crawl_url with sitemap."""
        # Mock context and its components
        mock_context = Mock()
        mock_context.request_context = Mock()
        mock_context.request_context.lifespan_context = Mock()

        mock_crawler = AsyncMock()
        mock_qdrant_client = Mock()

        mock_context.request_context.lifespan_context.crawler = mock_crawler
        mock_context.request_context.lifespan_context.qdrant_client = mock_qdrant_client

        # Mock sitemap parsing and batch crawl
        with patch("src.tools.web_tools.parse_sitemap") as mock_parse_sitemap:
            with patch("src.tools.web_tools.crawl_batch") as mock_crawl_batch:
                with patch("src.tools.web_tools.add_documents_to_vector_db"):
                    with patch("src.tools.web_tools.update_source_info"):
                        mock_parse_sitemap.return_value = [
                            "https://example.com/page1",
                            "https://example.com/page2",
                        ]
                        mock_crawl_batch.return_value = [
                            {
                                "url": "https://example.com/page1",
                                "markdown": "Content 1",
                            },
                            {
                                "url": "https://example.com/page2",
                                "markdown": "Content 2",
                            },
                        ]

                        result = await smart_crawl_url(
                            mock_context, "https://example.com/sitemap.xml"
                        )
                        result_data = json.loads(result)

                        assert result_data["success"] is True
                        assert result_data["url"] == "https://example.com/sitemap.xml"
                        assert result_data["crawl_type"] == "sitemap"

    @pytest.mark.asyncio
    async def test_smart_crawl_url_regular_page(self):
        """Test smart_crawl_url with regular webpage."""
        # Mock context and its components
        mock_context = Mock()
        mock_context.request_context = Mock()
        mock_context.request_context.lifespan_context = Mock()

        mock_crawler = AsyncMock()
        mock_qdrant_client = Mock()

        mock_context.request_context.lifespan_context.crawler = mock_crawler
        mock_context.request_context.lifespan_context.qdrant_client = mock_qdrant_client

        # Mock recursive crawl
        with patch(
            "src.tools.web_tools.crawl_recursive_internal_links"
        ) as mock_crawl_recursive:
            with patch("src.tools.web_tools.add_documents_to_vector_db"):
                with patch("src.tools.web_tools.update_source_info"):
                    mock_crawl_recursive.return_value = [
                        {"url": "https://example.com/page1", "markdown": "Content 1"}
                    ]

                    result = await smart_crawl_url(
                        mock_context, "https://example.com/page1"
                    )
                    result_data = json.loads(result)

                    assert result_data["success"] is True
                    assert result_data["url"] == "https://example.com/page1"
                    assert result_data["crawl_type"] == "webpage"
