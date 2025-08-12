"""
Tests for GitHub processor.
"""

import pytest
import os
import tempfile
from unittest.mock import patch

# Updated imports for modular architecture
from src.features.github import (
    GitRepository as GitHubRepoManager,
    MarkdownDiscovery,
    MultiFileDiscovery,
    MetadataExtractor as GitHubMetadataExtractor,
    GitHubService,  # Use the new service instead of legacy adapter
)
from src.features.github.processors import (
    BaseFileProcessor as FileTypeProcessor,  # Updated base class name
    MarkdownProcessor,
    PythonProcessor,
    ConfigProcessor,
)


class TestGitHubRepoManager:
    """Test cases for GitHubRepoManager."""

    def test_init(self):
        """Test GitHubRepoManager initialization."""
        manager = GitHubRepoManager()
        assert manager.temp_dirs == []
        assert manager.logger is not None

    def test_is_valid_github_url_valid(self):
        """Test _is_valid_github_url with valid URLs."""
        manager = GitHubRepoManager()
        assert manager._is_valid_github_url("https://github.com/user/repo") is True
        assert manager._is_valid_github_url("https://www.github.com/user/repo") is True
        assert manager._is_valid_github_url("https://github.com/user/repo.git") is True

    def test_is_valid_github_url_invalid(self):
        """Test _is_valid_github_url with invalid URLs."""
        manager = GitHubRepoManager()
        assert manager._is_valid_github_url("https://gitlab.com/user/repo") is False
        assert manager._is_valid_github_url("https://github.com/user") is False
        assert manager._is_valid_github_url("invalid-url") is False

    def test_normalize_clone_url(self):
        """Test _normalize_clone_url."""
        manager = GitHubRepoManager()
        assert (
            manager._normalize_clone_url("https://github.com/user/repo")
            == "https://github.com/user/repo.git"
        )
        assert (
            manager._normalize_clone_url("https://github.com/user/repo.git")
            == "https://github.com/user/repo.git"
        )

    def test_get_directory_size_mb(self):
        """Test _get_directory_size_mb."""
        manager = GitHubRepoManager()
        # Create a temporary directory with a small file
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            size_mb = manager._get_directory_size_mb(temp_dir)
            assert size_mb > 0

    def test_cleanup_directory(self):
        """Test _cleanup_directory."""
        manager = GitHubRepoManager()
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        assert os.path.exists(temp_dir)

        # Clean it up
        manager._cleanup_directory(temp_dir)
        # Note: This might not actually remove the directory on Windows due to permissions,
        # but it should not raise an exception


class TestMarkdownDiscovery:
    """Test cases for MarkdownDiscovery."""

    def test_init(self):
        """Test MarkdownDiscovery initialization."""
        discovery = MarkdownDiscovery()
        assert discovery.logger is not None

    def test_is_markdown_file(self):
        """Test _is_markdown_file."""
        discovery = MarkdownDiscovery()
        assert discovery._is_markdown_file("README.md") is True
        assert discovery._is_markdown_file("readme.markdown") is True
        assert discovery._is_markdown_file("test.txt") is False

    def test_should_exclude_dir(self):
        """Test _should_exclude_dir."""
        discovery = MarkdownDiscovery()
        assert discovery._should_exclude_dir(".git") is True
        assert discovery._should_exclude_dir("src") is False

    def test_is_readme_file(self):
        """Test _is_readme_file."""
        discovery = MarkdownDiscovery()
        assert discovery._is_readme_file("README.md") is True
        assert discovery._is_readme_file("readme.txt") is True
        assert discovery._is_readme_file("test.md") is False

    def test_file_priority_key(self):
        """Test _file_priority_key."""
        discovery = MarkdownDiscovery()
        readme_file = {"is_readme": True, "word_count": 1000}
        regular_file = {"is_readme": False, "word_count": 1000}

        readme_priority = discovery._file_priority_key(readme_file)
        regular_priority = discovery._file_priority_key(regular_file)

        assert readme_priority > regular_priority


class TestFileTypeProcessors:
    """Test cases for file type processors."""

    def test_file_type_processor_base_class(self):
        """Test FileTypeProcessor base class."""
        # BaseFileProcessor is now abstract, so test that it can't be instantiated
        with pytest.raises(TypeError):
            processor = FileTypeProcessor()

    def test_markdown_processor(self):
        """Test MarkdownProcessor."""
        processor = MarkdownProcessor()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            # Use longer content to meet minimum length requirement
            test_content = "# Test\n\nThis is a comprehensive test markdown file with sufficient content to meet the minimum length requirements."
            f.write(test_content)
            temp_path = f.name

        try:
            result = processor.process_file(temp_path, "test.md")
            assert len(result) == 1
            # Updated to use object attributes instead of dictionary access
            assert result[0].content_type == "markdown"
            assert result[0].language == "markdown"
        finally:
            os.unlink(temp_path)

    def test_python_processor(self):
        """Test PythonProcessor."""
        processor = PythonProcessor()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '"""Module docstring."""\ndef test():\n    """Function docstring."""\n    pass'
            )
            temp_path = f.name

        try:
            result = processor.process_file(temp_path, "test.py")
            assert len(result) >= 1
            # Should have at least the module docstring
            assert any(item.content_type == "module" for item in result)
        finally:
            os.unlink(temp_path)

    def test_config_processor(self):
        """Test ConfigProcessor."""
        processor = ConfigProcessor()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"key": "value"}')
            temp_path = f.name

        try:
            result = processor.process_file(temp_path, "test.json")
            assert len(result) == 1
            assert result[0].content_type == "configuration"
            assert result[0].language == "json"
        finally:
            os.unlink(temp_path)


class TestMultiFileDiscovery:
    """Test cases for MultiFileDiscovery."""

    def test_init(self):
        """Test MultiFileDiscovery initialization."""
        discovery = MultiFileDiscovery()
        assert discovery.logger is not None
        assert len(discovery.get_supported_extensions()) > 0

    def test_is_supported_file(self):
        """Test _is_supported_file."""
        discovery = MultiFileDiscovery()
        assert discovery._is_supported_file("test.md", [".md"]) is True
        assert discovery._is_supported_file("test.py", [".md"]) is False
        assert discovery._is_supported_file("test.md", [".py"]) is False

    def test_get_supported_file_types(self):
        """Test get_supported_extensions."""
        discovery = MultiFileDiscovery()
        supported_types = discovery.get_supported_extensions()
        assert isinstance(supported_types, list)
        assert ".md" in supported_types


class TestGitHubMetadataExtractor:
    """Test cases for GitHubMetadataExtractor."""

    def test_init(self):
        """Test GitHubMetadataExtractor initialization."""
        extractor = GitHubMetadataExtractor()
        assert extractor.logger is not None

    def test_parse_repo_info(self):
        """Test _parse_repo_info."""
        extractor = GitHubMetadataExtractor()
        owner, repo = extractor._parse_repo_info("https://github.com/user/repo")
        assert owner == "user"
        assert repo == "repo"

        owner, repo = extractor._parse_repo_info("https://github.com/user/repo.git")
        assert owner == "user"
        assert repo == "repo"

    def test_extract_repo_metadata(self):
        """Test extract_repo_metadata."""
        extractor = GitHubMetadataExtractor()
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple git repo for testing
            os.system(f"git init {temp_dir}")

            result = extractor.extract_repo_metadata(
                "https://github.com/user/test", temp_dir
            )
            # Updated to use object attributes instead of dictionary access
            assert result.repo_url == "https://github.com/user/test"
            assert result.owner == "user"
            assert result.repo_name == "test"


class TestGitHubService:
    """Test cases for GitHubService."""

    def test_init(self):
        """Test GitHubService initialization."""
        service = GitHubService()
        assert isinstance(service.git_repo, GitHubRepoManager)
        assert isinstance(service.metadata_extractor, GitHubMetadataExtractor)
        assert isinstance(service.file_discovery, MultiFileDiscovery)
        assert service.logger is not None

    def test_cleanup(self):
        """Test cleanup method."""
        service = GitHubService()
        with patch.object(service.git_repo, "cleanup") as mock_cleanup:
            service.cleanup()
            mock_cleanup.assert_called_once()

    def test_get_supported_file_types(self):
        """Test get_supported_file_types."""
        service = GitHubService()
        supported_types = service.get_supported_file_types()
        assert isinstance(supported_types, list)
        assert ".md" in supported_types

    def test_get_processors(self):
        """Test get_processors."""
        service = GitHubService()
        processors = service.get_processors()
        assert isinstance(processors, list)
        assert len(processors) > 0

    def test_get_service_statistics(self):
        """Test get_service_statistics."""
        service = GitHubService()
        stats = service.get_service_statistics()
        # Check that the stats object has the expected attributes
        assert hasattr(stats, "supported_file_types")
        assert hasattr(stats, "file_extensions")
        assert hasattr(stats, "processors_registered")
        assert hasattr(stats, "total_files_processed")
