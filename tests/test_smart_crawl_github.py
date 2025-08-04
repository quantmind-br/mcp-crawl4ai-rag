"""
# ruff: noqa: E402
Unit tests for smart_crawl_github MCP tool.

Tests the complete GitHub repository crawling functionality.
"""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from crawl4ai_mcp import smart_crawl_github


class TestSmartCrawlGitHub:
    """Test cases for smart_crawl_github MCP tool."""

    def setup_method(self):
        """Setup test method with common fixtures."""
        self.mock_context = Mock()
        self.mock_qdrant_client = Mock()
        self.mock_context.request_context.lifespan_context.qdrant_client = (
            self.mock_qdrant_client
        )

    @pytest.mark.asyncio
    async def test_invalid_github_url(self):
        """Test with invalid GitHub URL."""
        result = await smart_crawl_github(
            ctx=self.mock_context, repo_url="https://gitlab.com/user/repo"
        )

        response = json.loads(result)
        assert response["success"] is False
        assert "Invalid GitHub repository URL" in response["error"]
        assert response["repo_url"] == "https://gitlab.com/user/repo"

    @pytest.mark.asyncio
    async def test_empty_url(self):
        """Test with empty URL."""
        result = await smart_crawl_github(ctx=self.mock_context, repo_url="")

        response = json.loads(result)
        assert response["success"] is False
        assert "Invalid GitHub repository URL" in response["error"]

    @pytest.mark.asyncio
    @patch("crawl4ai_mcp.GitHubRepoManager")
    @patch("crawl4ai_mcp.MarkdownDiscovery")
    @patch("crawl4ai_mcp.GitHubMetadataExtractor")
    async def test_no_markdown_files_found(
        self, mock_extractor_cls, mock_discovery_cls, mock_manager_cls
    ):
        """Test when no markdown files are found in repository."""
        # Setup mocks
        mock_manager = Mock()
        mock_discovery = Mock()
        mock_extractor = Mock()

        mock_manager_cls.return_value = mock_manager
        mock_discovery_cls.return_value = mock_discovery
        mock_extractor_cls.return_value = mock_extractor

        mock_manager.clone_repository.return_value = "/tmp/test_repo"
        mock_extractor.extract_repo_metadata.return_value = {
            "owner": "user",
            "repo_name": "repo",
        }
        mock_discovery.discover_markdown_files.return_value = []

        result = await smart_crawl_github(
            ctx=self.mock_context, repo_url="https://github.com/user/repo"
        )

        response = json.loads(result)
        assert response["success"] is False
        assert "No markdown files found" in response["error"]
        mock_manager.cleanup.assert_called_once()

    @pytest.mark.asyncio
    @patch("crawl4ai_mcp.GitHubRepoManager")
    @patch("crawl4ai_mcp.MarkdownDiscovery")
    @patch("crawl4ai_mcp.GitHubMetadataExtractor")
    @patch("crawl4ai_mcp.smart_chunk_markdown")
    @patch("crawl4ai_mcp.extract_source_summary")
    @patch("crawl4ai_mcp.update_source_info")
    @patch("crawl4ai_mcp.add_documents_to_supabase")
    async def test_successful_crawl(
        self,
        mock_add_docs,
        mock_update_source,
        mock_extract_summary,
        mock_chunk,
        mock_extractor_cls,
        mock_discovery_cls,
        mock_manager_cls,
    ):
        """Test successful GitHub repository crawling."""
        # Setup mocks
        mock_manager = Mock()
        mock_discovery = Mock()
        mock_extractor = Mock()

        mock_manager_cls.return_value = mock_manager
        mock_discovery_cls.return_value = mock_discovery
        mock_extractor_cls.return_value = mock_extractor

        # Mock repository data
        mock_manager.clone_repository.return_value = "/tmp/test_repo"
        mock_manager._get_directory_size_mb.return_value = 25.5

        mock_repo_metadata = {
            "owner": "user",
            "repo_name": "test-repo",
            "full_name": "user/test-repo",
            "language": "python",
            "description": "A test repository",
        }
        mock_extractor.extract_repo_metadata.return_value = mock_repo_metadata

        # Mock markdown files
        mock_markdown_files = [
            {
                "filename": "README.md",
                "relative_path": "README.md",
                "content": "# Test Repository\n\nThis is a test repository for testing purposes.",
                "size_bytes": 1024,
                "word_count": 50,
                "is_readme": True,
            },
            {
                "filename": "docs.md",
                "relative_path": "docs/guide.md",
                "content": "# User Guide\n\nThis is the user guide.",
                "size_bytes": 512,
                "word_count": 25,
                "is_readme": False,
            },
        ]
        mock_discovery.discover_markdown_files.return_value = mock_markdown_files

        # Mock chunking
        mock_chunk.side_effect = lambda content, chunk_size: [content[:chunk_size]]

        # Mock source summary
        mock_extract_summary.return_value = "Test repository for demonstration"

        result = await smart_crawl_github(
            ctx=self.mock_context, repo_url="https://github.com/user/test-repo"
        )

        response = json.loads(result)

        # Verify success response
        assert response["success"] is True
        assert response["repo_url"] == "https://github.com/user/test-repo"
        assert response["owner"] == "user"
        assert response["repo_name"] == "test-repo"
        assert response["markdown_files_processed"] == 2
        assert response["chunks_stored"] == 2
        assert response["total_word_count"] == 75
        assert response["repository_size_mb"] == 25.5
        assert "github.com/user/test-repo" in response["source_id"]
        assert len(response["files_processed"]) == 2

        # Verify method calls
        mock_manager.clone_repository.assert_called_once_with(
            "https://github.com/user/test-repo", 500
        )
        mock_extractor.extract_repo_metadata.assert_called_once()
        mock_discovery.discover_markdown_files.assert_called_once()
        mock_update_source.assert_called_once()
        mock_add_docs.assert_called_once()
        mock_manager.cleanup.assert_called_once()

    @pytest.mark.asyncio
    @patch("crawl4ai_mcp.GitHubRepoManager")
    @patch("crawl4ai_mcp.MarkdownDiscovery")
    @patch("crawl4ai_mcp.GitHubMetadataExtractor")
    @patch("crawl4ai_mcp.smart_chunk_markdown")
    @patch("crawl4ai_mcp.extract_source_summary")
    @patch("crawl4ai_mcp.update_source_info")
    @patch("crawl4ai_mcp.add_documents_to_supabase")
    @patch("crawl4ai_mcp.extract_code_blocks")
    @patch("crawl4ai_mcp.generate_code_example_summary")
    @patch("crawl4ai_mcp.add_code_examples_to_supabase")
    @patch("crawl4ai_mcp.os.getenv")
    async def test_with_code_examples_enabled(
        self,
        mock_getenv,
        mock_add_code,
        mock_gen_summary,
        mock_extract_code,
        mock_add_docs,
        mock_update_source,
        mock_extract_summary,
        mock_chunk,
        mock_extractor_cls,
        mock_discovery_cls,
        mock_manager_cls,
    ):
        """Test crawling with code examples extraction enabled."""
        # Setup environment variable
        mock_getenv.side_effect = (
            lambda key, default=None: "true" if key == "USE_AGENTIC_RAG" else default
        )

        # Setup mocks
        mock_manager = Mock()
        mock_discovery = Mock()
        mock_extractor = Mock()

        mock_manager_cls.return_value = mock_manager
        mock_discovery_cls.return_value = mock_discovery
        mock_extractor_cls.return_value = mock_extractor

        mock_manager.clone_repository.return_value = "/tmp/test_repo"
        mock_manager._get_directory_size_mb.return_value = 10.0

        mock_repo_metadata = {"owner": "user", "repo_name": "test-repo"}
        mock_extractor.extract_repo_metadata.return_value = mock_repo_metadata

        # Mock markdown files with code
        mock_markdown_files = [
            {
                "filename": "README.md",
                "relative_path": "README.md",
                "content": "# Test\n```python\ndef hello():\n    print('Hello')\n```",
                "size_bytes": 1024,
                "word_count": 20,
                "is_readme": True,
            }
        ]
        mock_discovery.discover_markdown_files.return_value = mock_markdown_files

        # Mock code extraction
        mock_code_blocks = [
            {
                "code": "def hello():\n    print('Hello')",
                "language": "python",
                "context_before": "# Test",
                "context_after": "",
            }
        ]
        mock_extract_code.return_value = mock_code_blocks
        mock_gen_summary.return_value = "Python function that prints Hello"

        mock_chunk.side_effect = lambda content, chunk_size: [content]
        mock_extract_summary.return_value = "Test repository"

        result = await smart_crawl_github(
            ctx=self.mock_context, repo_url="https://github.com/user/test-repo"
        )

        response = json.loads(result)

        # Verify code examples were processed
        assert response["success"] is True
        assert response["code_examples_stored"] == 1

        # Verify code processing methods were called
        mock_extract_code.assert_called_once()
        mock_gen_summary.assert_called_once()
        mock_add_code.assert_called_once()

    @pytest.mark.asyncio
    @patch("crawl4ai_mcp.GitHubRepoManager")
    async def test_clone_failure(self, mock_manager_cls):
        """Test when repository cloning fails."""
        mock_manager = Mock()
        mock_manager_cls.return_value = mock_manager
        mock_manager.clone_repository.side_effect = RuntimeError("Clone failed")

        result = await smart_crawl_github(
            ctx=self.mock_context, repo_url="https://github.com/user/repo"
        )

        response = json.loads(result)
        assert response["success"] is False
        assert "Clone failed" in response["error"]
        mock_manager.cleanup.assert_called_once()

    @pytest.mark.asyncio
    @patch("crawl4ai_mcp.GitHubRepoManager")
    @patch("crawl4ai_mcp.MarkdownDiscovery")
    @patch("crawl4ai_mcp.GitHubMetadataExtractor")
    async def test_metadata_extraction_failure(
        self, mock_extractor_cls, mock_discovery_cls, mock_manager_cls
    ):
        """Test when metadata extraction fails."""
        mock_manager = Mock()
        mock_discovery = Mock()
        mock_extractor = Mock()

        mock_manager_cls.return_value = mock_manager
        mock_discovery_cls.return_value = mock_discovery
        mock_extractor_cls.return_value = mock_extractor

        mock_manager.clone_repository.return_value = "/tmp/test_repo"
        mock_extractor.extract_repo_metadata.side_effect = Exception(
            "Metadata extraction failed"
        )

        result = await smart_crawl_github(
            ctx=self.mock_context, repo_url="https://github.com/user/repo"
        )

        response = json.loads(result)
        assert response["success"] is False
        assert "Metadata extraction failed" in response["error"]
        mock_manager.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_parameters(self):
        """Test with custom parameters."""
        with (
            patch("crawl4ai_mcp.GitHubRepoManager") as mock_manager_cls,
            patch("crawl4ai_mcp.MarkdownDiscovery") as mock_discovery_cls,
            patch("crawl4ai_mcp.GitHubMetadataExtractor") as mock_extractor_cls,
            patch("crawl4ai_mcp.smart_chunk_markdown") as mock_chunk,
            patch("crawl4ai_mcp.extract_source_summary") as mock_extract_summary,
            patch("crawl4ai_mcp.update_source_info"),
            patch("crawl4ai_mcp.add_documents_to_supabase"),
        ):
            # Setup mocks
            mock_manager = Mock()
            mock_discovery = Mock()
            mock_extractor = Mock()

            mock_manager_cls.return_value = mock_manager
            mock_discovery_cls.return_value = mock_discovery
            mock_extractor_cls.return_value = mock_extractor

            mock_manager.clone_repository.return_value = "/tmp/test_repo"
            mock_manager._get_directory_size_mb.return_value = 100.0

            mock_extractor.extract_repo_metadata.return_value = {
                "owner": "user",
                "repo_name": "repo",
            }

            mock_markdown_files = [
                {
                    "filename": "README.md",
                    "relative_path": "README.md",
                    "content": "Test content",
                    "size_bytes": 500,
                    "word_count": 10,
                    "is_readme": True,
                }
            ]
            mock_discovery.discover_markdown_files.return_value = mock_markdown_files
            mock_chunk.return_value = ["Test content"]
            mock_extract_summary.return_value = "Test"

            # Test with custom parameters
            result = await smart_crawl_github(
                ctx=self.mock_context,
                repo_url="https://github.com/user/repo",
                max_files=25,
                chunk_size=3000,
                max_size_mb=200,
            )

            response = json.loads(result)
            assert response["success"] is True

            # Verify custom parameters were used
            mock_manager.clone_repository.assert_called_once_with(
                "https://github.com/user/repo", 200
            )
            mock_discovery.discover_markdown_files.assert_called_once()

            # Check that max_files parameter was passed
            call_args = mock_discovery.discover_markdown_files.call_args
            assert call_args[1]["max_files"] == 25

    @pytest.mark.asyncio
    @patch("crawl4ai_mcp.GitHubRepoManager")
    @patch("crawl4ai_mcp.MarkdownDiscovery")
    @patch("crawl4ai_mcp.GitHubMetadataExtractor")
    @patch("crawl4ai_mcp.smart_chunk_markdown")
    @patch("crawl4ai_mcp.extract_source_summary")
    @patch("crawl4ai_mcp.update_source_info")
    @patch("crawl4ai_mcp.add_documents_to_supabase")
    async def test_many_files_truncation(
        self,
        mock_add_docs,
        mock_update_source,
        mock_extract_summary,
        mock_chunk,
        mock_extractor_cls,
        mock_discovery_cls,
        mock_manager_cls,
    ):
        """Test that file list is truncated in response when there are many files."""
        # Setup mocks
        mock_manager = Mock()
        mock_discovery = Mock()
        mock_extractor = Mock()

        mock_manager_cls.return_value = mock_manager
        mock_discovery_cls.return_value = mock_discovery
        mock_extractor_cls.return_value = mock_extractor

        mock_manager.clone_repository.return_value = "/tmp/test_repo"
        mock_manager._get_directory_size_mb.return_value = 50.0

        mock_extractor.extract_repo_metadata.return_value = {
            "owner": "user",
            "repo_name": "repo",
        }

        # Create 15 mock files (more than the 10 limit)
        mock_markdown_files = []
        for i in range(15):
            mock_markdown_files.append(
                {
                    "filename": f"file{i}.md",
                    "relative_path": f"docs/file{i}.md",
                    "content": f"Content of file {i}",
                    "size_bytes": 100,
                    "word_count": 5,
                    "is_readme": False,
                }
            )

        mock_discovery.discover_markdown_files.return_value = mock_markdown_files
        mock_chunk.side_effect = lambda content, chunk_size: [content]
        mock_extract_summary.return_value = "Test repository"

        result = await smart_crawl_github(
            ctx=self.mock_context, repo_url="https://github.com/user/repo"
        )

        response = json.loads(result)
        assert response["success"] is True
        assert response["markdown_files_processed"] == 15

        # Check that files_processed list is truncated and has "..." indicator
        assert len(response["files_processed"]) == 11  # 10 files + "..."
        assert response["files_processed"][-1] == "..."

    @pytest.mark.asyncio
    @patch("crawl4ai_mcp.GitHubRepoManager")
    async def test_cleanup_on_exception(self, mock_manager_cls):
        """Test that cleanup is called even when an exception occurs."""
        mock_manager = Mock()
        mock_manager_cls.return_value = mock_manager
        mock_manager.clone_repository.side_effect = Exception("Unexpected error")

        result = await smart_crawl_github(
            ctx=self.mock_context, repo_url="https://github.com/user/repo"
        )

        response = json.loads(result)
        assert response["success"] is False
        assert "Unexpected error" in response["error"]

        # Verify cleanup was called despite the exception
        mock_manager.cleanup.assert_called_once()


@pytest.mark.integration
class TestSmartCrawlGitHubIntegration:
    """Integration tests for smart_crawl_github with real components."""

    def setup_method(self):
        """Setup test method with real components."""
        self.mock_context = Mock()
        self.mock_qdrant_client = Mock()
        self.mock_context.request_context.lifespan_context.qdrant_client = (
            self.mock_qdrant_client
        )

        # Setup realistic Qdrant client behavior
        self.mock_qdrant_client.update_source_info.return_value = None
        self.mock_qdrant_client.upsert_points.return_value = None

    @pytest.mark.asyncio
    @patch("utils.github_processor.tempfile.mkdtemp")
    @patch("utils.github_processor.subprocess.run")
    @patch("utils.github_processor.os.walk")
    @patch("utils.github_processor.os.path.getsize")
    @patch("utils.github_processor.os.path.exists")
    @patch("builtins.open")
    async def test_end_to_end_workflow(
        self,
        mock_open,
        mock_exists,
        mock_getsize,
        mock_walk,
        mock_subprocess,
        mock_mkdtemp,
    ):
        """Test end-to-end workflow with realistic file system simulation."""
        # Setup file system simulation
        mock_mkdtemp.return_value = "/tmp/github_clone_test"
        mock_subprocess.return_value = Mock(returncode=0, stderr="")

        # Simulate repository structure
        mock_walk.return_value = [
            ("/tmp/github_clone_test", ["docs"], ["README.md", "setup.py"]),
            ("/tmp/github_clone_test/docs", [], ["guide.md", "api.md"]),
        ]

        mock_getsize.return_value = 2048  # 2KB files
        mock_exists.return_value = False  # No package files for simplicity

        # Setup file contents
        readme_content = """# Test Repository

This is a test repository for demonstration purposes.

## Features
- Feature 1
- Feature 2

## Installation
```bash
pip install test-package
```
"""

        guide_content = """# User Guide

This guide explains how to use the test package.

## Quick Start
```python
import test_package
test_package.hello()
```
"""

        mock_file_contents = {
            "/tmp/github_clone_test/README.md": readme_content,
            "/tmp/github_clone_test/docs/guide.md": guide_content,
        }

        def mock_open_func(filename, mode="r", encoding=None, errors=None):
            mock_file = Mock()
            mock_file.read.return_value = mock_file_contents.get(filename, "")
            mock_file.__enter__.return_value = mock_file
            mock_file.__exit__.return_value = None
            return mock_file

        mock_open.side_effect = mock_open_func

        # Mock additional utilities
        with (
            patch("crawl4ai_mcp.smart_chunk_markdown") as mock_chunk,
            patch("crawl4ai_mcp.extract_source_summary") as mock_extract_summary,
            patch("crawl4ai_mcp.update_source_info") as mock_update_source,
            patch("crawl4ai_mcp.add_documents_to_supabase") as mock_add_docs,
            patch("utils.github_processor.shutil.rmtree") as mock_rmtree,
        ):
            # Setup utility mocks
            mock_chunk.side_effect = (
                lambda content, chunk_size: [content[:chunk_size], content[chunk_size:]]
                if len(content) > chunk_size
                else [content]
            )
            mock_extract_summary.return_value = (
                "Test repository for demonstration purposes"
            )

            result = await smart_crawl_github(
                ctx=self.mock_context, repo_url="https://github.com/testuser/test-repo"
            )

            response = json.loads(result)

            # Verify successful execution
            assert response["success"] is True
            assert response["owner"] == "testuser"
            assert response["repo_name"] == "test-repo"
            assert response["markdown_files_processed"] >= 1
            assert response["chunks_stored"] >= 1

            # Verify all components were used
            mock_subprocess.assert_called_once()  # Git clone
            mock_update_source.assert_called_once()  # Source info updated
            mock_add_docs.assert_called_once()  # Documents added
            mock_rmtree.assert_called_once()  # Cleanup performed
