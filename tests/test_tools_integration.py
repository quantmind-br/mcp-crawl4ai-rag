"""
Integration tests for all MCP tools to ensure they work together properly.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestWebToolsIntegration:
    """Test web tools integration and functionality."""

    @pytest.mark.asyncio
    @patch('src.tools.web_tools.AsyncWebCrawler')
    @patch('src.tools.web_tools.get_qdrant_client')
    async def test_crawl_single_page_integration(self, mock_get_client, mock_crawler_class):
        """Test crawl_single_page tool integration."""
        from src.tools.web_tools import crawl_single_page
        
        # Setup mocks
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
        
        mock_result = Mock()
        mock_result.success = True
        mock_result.markdown = "# Test Content"
        mock_result.extracted_content = "Test content"
        mock_crawler.arun.return_value = mock_result
        
        mock_client = Mock()
        mock_client.add_documents_to_qdrant.return_value = ["doc1", "doc2"]
        mock_get_client.return_value = mock_client
        
        # Test the tool
        result = await crawl_single_page("https://example.com")
        
        assert "success" in result
        assert "document_ids" in result
        assert result["success"] is True
        assert len(result["document_ids"]) == 2

    @pytest.mark.asyncio
    @patch('src.tools.web_tools.AsyncWebCrawler')
    @patch('src.tools.web_tools.get_qdrant_client')
    async def test_smart_crawl_url_integration(self, mock_get_client, mock_crawler_class):
        """Test smart_crawl_url tool integration."""
        from src.tools.web_tools import smart_crawl_url
        
        # Setup mocks for regular webpage crawling
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
        
        mock_result = Mock()
        mock_result.success = True
        mock_result.markdown = "# Page Content"
        mock_result.extracted_content = "Page content"
        mock_crawler.arun.return_value = mock_result
        
        mock_client = Mock()
        mock_client.add_documents_to_qdrant.return_value = ["doc1"]
        mock_get_client.return_value = mock_client
        
        # Test regular webpage
        result = await smart_crawl_url("https://example.com", max_depth=1)
        
        assert "success" in result
        assert "total_documents" in result
        assert result["success"] is True

    def test_extract_code_blocks(self):
        """Test code block extraction utility."""
        from src.tools.web_tools import extract_code_blocks
        
        markdown_content = """
        # Sample Content
        
        Here's some Python code:
        
        ```python
        def hello():
            return "Hello, World!"
        ```
        
        And some JavaScript:
        
        ```javascript
        function greet(name) {
            return `Hello, ${name}!`;
        }
        ```
        
        Regular text here.
        """
        
        code_blocks = extract_code_blocks(markdown_content)
        
        assert len(code_blocks) == 2
        assert any("def hello()" in block["content"] for block in code_blocks)
        assert any("function greet" in block["content"] for block in code_blocks)
        assert any(block["language"] == "python" for block in code_blocks)
        assert any(block["language"] == "javascript" for block in code_blocks)


class TestRagToolsIntegration:
    """Test RAG tools integration and functionality."""

    @patch('src.tools.rag_tools.get_qdrant_client')
    def test_get_available_sources(self, mock_get_client):
        """Test get_available_sources tool."""
        from src.tools.rag_tools import get_available_sources
        
        mock_client = Mock()
        mock_sources = [
            {
                "source_id": "example.com",
                "summary": "Example website",
                "total_word_count": 1000,
                "updated_at": "2024-01-01T00:00:00Z"
            }
        ]
        mock_client.get_available_sources.return_value = mock_sources
        mock_get_client.return_value = mock_client
        
        result = get_available_sources()
        
        assert "sources" in result
        assert len(result["sources"]) == 1
        assert result["sources"][0]["source_id"] == "example.com"

    @patch('src.tools.rag_tools.get_qdrant_client')
    def test_perform_rag_query(self, mock_get_client):
        """Test perform_rag_query tool."""
        from src.tools.rag_tools import perform_rag_query
        
        mock_client = Mock()
        mock_results = [
            {
                "id": "doc1",
                "similarity": 0.95,
                "content": "Python programming content",
                "url": "https://python.org",
                "source_id": "python.org"
            }
        ]
        mock_client.search_documents.return_value = mock_results
        mock_get_client.return_value = mock_client
        
        result = perform_rag_query("python programming", match_count=5)
        
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["content"] == "Python programming content"

    @pytest.mark.skipif(
        os.getenv("USE_AGENTIC_RAG", "false") != "true",
        reason="Agentic RAG not enabled"
    )
    @patch('src.tools.rag_tools.get_qdrant_client')
    def test_search_code_examples(self, mock_get_client):
        """Test search_code_examples tool (if agentic RAG is enabled)."""
        from src.tools.rag_tools import search_code_examples
        
        mock_client = Mock()
        mock_results = [
            {
                "id": "code1",
                "similarity": 0.92,
                "content": "def example(): pass",
                "summary": "Example function",
                "url": "https://example.com/code",
                "source_id": "example.com"
            }
        ]
        mock_client.search_code_examples.return_value = mock_results
        mock_get_client.return_value = mock_client
        
        result = search_code_examples("python function example")
        
        assert "code_examples" in result
        assert len(result["code_examples"]) == 1
        assert "def example()" in result["code_examples"][0]["content"]


class TestGithubToolsIntegration:
    """Test GitHub tools integration."""

    @pytest.mark.asyncio
    @patch('src.tools.github_tools.AsyncWebCrawler')
    @patch('src.tools.github_tools.get_qdrant_client')
    async def test_smart_crawl_github_integration(self, mock_get_client, mock_crawler_class):
        """Test smart_crawl_github tool integration."""
        from src.tools.github_tools import smart_crawl_github
        
        # Setup mocks
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
        
        # Mock file processing
        mock_client = Mock()
        mock_client.add_documents_to_qdrant.return_value = ["doc1", "doc2"]
        mock_client.add_code_examples_to_qdrant.return_value = ["code1"]
        mock_get_client.return_value = mock_client
        
        # Mock file system operations
        with patch('src.tools.github_tools.GitHubRepoManager') as mock_manager:
            mock_repo_manager = Mock()
            mock_manager.return_value = mock_repo_manager
            mock_repo_manager.clone_repository.return_value = "/tmp/repo"
            mock_repo_manager.get_file_list.return_value = [
                "/tmp/repo/README.md",
                "/tmp/repo/src/main.py"
            ]
            mock_repo_manager.cleanup.return_value = None
            
            with patch('builtins.open', mock_open_multiple_files({
                "/tmp/repo/README.md": "# Test Repository\n\nThis is a test.",
                "/tmp/repo/src/main.py": "def main():\n    print('Hello')\n"
            })):
                result = await smart_crawl_github(
                    "https://github.com/user/repo.git",
                    max_files=10,
                    file_types_to_index=[".md", ".py"]
                )
        
        assert "success" in result
        assert "total_documents" in result
        assert "total_code_examples" in result


class TestKnowledgeGraphToolsIntegration:
    """Test knowledge graph tools integration (if enabled)."""

    @pytest.mark.skipif(
        os.getenv("USE_KNOWLEDGE_GRAPH", "false") != "true",
        reason="Knowledge graph not enabled"
    )
    @patch('src.tools.kg_tools.DirectNeo4jExtractor')
    @pytest.mark.asyncio
    async def test_parse_github_repository(self, mock_extractor):
        """Test parse_github_repository tool."""
        from src.tools.kg_tools import parse_github_repository
        
        mock_extractor_instance = AsyncMock()
        mock_extractor.return_value = mock_extractor_instance
        mock_extractor_instance.extract_repository_to_neo4j.return_value = {
            "repositories": 1,
            "files": 5,
            "classes": 3,
            "methods": 10
        }
        
        result = await parse_github_repository("https://github.com/user/repo.git")
        
        assert "success" in result
        assert "statistics" in result
        assert result["statistics"]["repositories"] == 1

    @pytest.mark.skipif(
        os.getenv("USE_KNOWLEDGE_GRAPH", "false") != "true",
        reason="Knowledge graph not enabled"
    )
    @patch('src.tools.kg_tools.KnowledgeGraphValidator')
    @pytest.mark.asyncio
    async def test_check_ai_script_hallucinations(self, mock_validator):
        """Test check_ai_script_hallucinations tool."""
        from src.tools.kg_tools import check_ai_script_hallucinations
        
        mock_validator_instance = AsyncMock()
        mock_validator.return_value = mock_validator_instance
        mock_validator_instance.validate_script.return_value = {
            "confidence": 0.95,
            "hallucinations": [],
            "verified_imports": ["os", "sys"],
            "verified_methods": []
        }
        
        # Create a temporary test script
        test_script = "import os\nimport sys\nprint('Hello, World!')"
        
        with patch('builtins.open', mock_open(read_data=test_script)):
            result = await check_ai_script_hallucinations("/tmp/test_script.py")
        
        assert "confidence" in result
        assert "hallucinations" in result
        assert result["confidence"] == 0.95


def mock_open_multiple_files(files_dict):
    """Helper to mock opening multiple files with different content."""
    from unittest.mock import mock_open
    
    def side_effect(filename, *args, **kwargs):
        for file_path, content in files_dict.items():
            if filename.endswith(Path(file_path).name) or filename == file_path:
                return mock_open(read_data=content).return_value
        return mock_open(read_data="").return_value
    
    return Mock(side_effect=side_effect)


class TestToolsHealthCheck:
    """Test tools health check and error handling."""

    @patch('src.tools.rag_tools.get_qdrant_client')
    def test_health_check_reranking(self, mock_get_client):
        """Test reranking health check tool."""
        from src.tools.rag_tools import health_check_reranking
        
        mock_client = Mock()
        mock_client.health_check.return_value = {"status": "healthy"}
        mock_get_client.return_value = mock_client
        
        with patch.dict(os.environ, {"USE_RERANKING": "true"}):
            result = health_check_reranking()
        
        assert "reranking_enabled" in result
        assert "qdrant_status" in result
        assert result["reranking_enabled"] is True

    def test_tools_error_handling(self):
        """Test that tools handle errors gracefully."""
        from src.tools.rag_tools import get_available_sources
        
        with patch('src.tools.rag_tools.get_qdrant_client', side_effect=Exception("Connection failed")):
            result = get_available_sources()
            
            assert "error" in result
            assert "sources" not in result
            assert "Connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_tools_input_validation(self):
        """Test that tools validate inputs properly."""
        from src.tools.web_tools import crawl_single_page
        
        # Test invalid URL
        result = await crawl_single_page("not-a-valid-url")
        
        assert "error" in result
        assert result.get("success") is False