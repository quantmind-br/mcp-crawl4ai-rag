"""
Unit tests for GitHub processor utilities.

Tests the GitHubRepoManager, MarkdownDiscovery, and GitHubMetadataExtractor classes.
"""
import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import List, Dict, Any

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from utils.github_processor import GitHubRepoManager, MarkdownDiscovery, GitHubMetadataExtractor
from utils.validation import validate_github_url, normalize_github_url


class TestGitHubRepoManager:
    """Test cases for GitHubRepoManager class."""
    
    def test_init(self):
        """Test initialization of GitHubRepoManager."""
        manager = GitHubRepoManager()
        assert manager.temp_dirs == []
        assert manager.logger is not None
    
    def test_is_valid_github_url(self):
        """Test GitHub URL validation."""
        manager = GitHubRepoManager()
        
        # Valid URLs
        assert manager._is_valid_github_url("https://github.com/user/repo") is True
        assert manager._is_valid_github_url("https://github.com/user/repo.git") is True
        assert manager._is_valid_github_url("https://www.github.com/user/repo") is True
        
        # Invalid URLs
        assert manager._is_valid_github_url("https://gitlab.com/user/repo") is False
        assert manager._is_valid_github_url("https://github.com/user") is False
        assert manager._is_valid_github_url("https://github.com/") is False
        assert manager._is_valid_github_url("invalid-url") is False
    
    def test_normalize_clone_url(self):
        """Test URL normalization for git clone."""
        manager = GitHubRepoManager()
        
        assert manager._normalize_clone_url("https://github.com/user/repo") == "https://github.com/user/repo.git"
        assert manager._normalize_clone_url("https://github.com/user/repo.git") == "https://github.com/user/repo.git"
        assert manager._normalize_clone_url("https://github.com/user/repo/") == "https://github.com/user/repo.git"
    
    @patch('utils.github_processor.tempfile.mkdtemp')
    @patch('utils.github_processor.subprocess.run')
    @patch('utils.github_processor.os.walk')
    @patch('utils.github_processor.os.path.getsize')
    def test_clone_repository_success(self, mock_getsize, mock_walk, mock_subprocess, mock_mkdtemp):
        """Test successful repository cloning."""
        # Setup mocks
        mock_mkdtemp.return_value = "/tmp/test_clone"
        mock_subprocess.return_value = Mock(returncode=0, stderr="")
        mock_walk.return_value = [("/tmp/test_clone", [], ["file1.md", "file2.txt"])]
        mock_getsize.side_effect = [1024, 2048]  # File sizes
        
        manager = GitHubRepoManager()
        
        # Test
        result = manager.clone_repository("https://github.com/user/repo")
        
        # Verify
        assert result == "/tmp/test_clone"
        assert "/tmp/test_clone" in manager.temp_dirs
        mock_subprocess.assert_called_once_with(
            ["git", "clone", "--depth", "1", "https://github.com/user/repo.git", "/tmp/test_clone"],
            capture_output=True,
            text=True,
            timeout=300
        )
    
    @patch('utils.github_processor.tempfile.mkdtemp')
    @patch('utils.github_processor.subprocess.run')
    def test_clone_repository_invalid_url(self, mock_subprocess, mock_mkdtemp):
        """Test cloning with invalid URL."""
        mock_mkdtemp.return_value = "/tmp/test_clone"
        
        manager = GitHubRepoManager()
        
        with pytest.raises(ValueError, match="Invalid GitHub repository URL"):
            manager.clone_repository("https://gitlab.com/user/repo")
    
    @patch('utils.github_processor.tempfile.mkdtemp')
    @patch('utils.github_processor.subprocess.run')
    def test_clone_repository_git_failure(self, mock_subprocess, mock_mkdtemp):
        """Test cloning when git command fails."""
        mock_mkdtemp.return_value = "/tmp/test_clone"
        mock_subprocess.return_value = Mock(returncode=1, stderr="Repository not found")
        
        manager = GitHubRepoManager()
        
        with pytest.raises(RuntimeError, match="Git clone failed"):
            manager.clone_repository("https://github.com/user/nonexistent")
    
    @patch('utils.github_processor.tempfile.mkdtemp')
    @patch('utils.github_processor.subprocess.run')
    @patch('utils.github_processor.os.walk')
    @patch('utils.github_processor.os.path.getsize')
    def test_clone_repository_too_large(self, mock_getsize, mock_walk, mock_subprocess, mock_mkdtemp):
        """Test cloning when repository is too large."""
        # Setup mocks
        mock_mkdtemp.return_value = "/tmp/test_clone"
        mock_subprocess.return_value = Mock(returncode=0, stderr="")
        mock_walk.return_value = [("/tmp/test_clone", [], ["large_file.bin"])]
        mock_getsize.return_value = 600 * 1024 * 1024  # 600MB file
        
        manager = GitHubRepoManager()
        
        with pytest.raises(RuntimeError, match="Repository too large"):
            manager.clone_repository("https://github.com/user/large-repo", max_size_mb=500)
    
    @patch('shutil.rmtree')
    @patch('os.path.exists')
    def test_cleanup(self, mock_exists, mock_rmtree):
        """Test cleanup of temporary directories."""
        mock_exists.return_value = True
        
        manager = GitHubRepoManager()
        manager.temp_dirs = ["/tmp/dir1", "/tmp/dir2"]
        
        manager.cleanup()
        
        assert manager.temp_dirs == []
        assert mock_rmtree.call_count == 2


class TestMarkdownDiscovery:
    """Test cases for MarkdownDiscovery class."""
    
    def test_init(self):
        """Test initialization of MarkdownDiscovery."""
        discovery = MarkdownDiscovery()
        assert discovery.logger is not None
        assert discovery.EXCLUDED_DIRS is not None
        assert discovery.EXCLUDED_PATTERNS is not None
    
    def test_is_markdown_file(self):
        """Test markdown file detection."""
        discovery = MarkdownDiscovery()
        
        # Valid markdown files
        assert discovery._is_markdown_file("README.md") is True
        assert discovery._is_markdown_file("doc.markdown") is True
        assert discovery._is_markdown_file("guide.mdown") is True
        assert discovery._is_markdown_file("notes.mkd") is True
        
        # Non-markdown files
        assert discovery._is_markdown_file("script.py") is False
        assert discovery._is_markdown_file("data.json") is False
        assert discovery._is_markdown_file("image.png") is False
    
    def test_should_exclude_dir(self):
        """Test directory exclusion logic."""
        discovery = MarkdownDiscovery()
        
        # Excluded directories
        assert discovery._should_exclude_dir(".git") is True
        assert discovery._should_exclude_dir("node_modules") is True
        assert discovery._should_exclude_dir(".hidden") is True
        
        # Included directories
        assert discovery._should_exclude_dir("docs") is False
        assert discovery._should_exclude_dir("src") is False
        assert discovery._should_exclude_dir("examples") is False
    
    def test_should_exclude_file(self):
        """Test file exclusion logic."""
        discovery = MarkdownDiscovery()
        
        # Excluded files
        assert discovery._should_exclude_file("CHANGELOG.md") is True
        assert discovery._should_exclude_file("package-lock.json") is True
        assert discovery._should_exclude_file("script.min.js") is True
        
        # Included files
        assert discovery._should_exclude_file("README.md") is False
        assert discovery._should_exclude_file("guide.md") is False
    
    def test_is_readme_file(self):
        """Test README file detection."""
        discovery = MarkdownDiscovery()
        
        assert discovery._is_readme_file("README.md") is True
        assert discovery._is_readme_file("readme.txt") is True
        assert discovery._is_readme_file("ReadMe.markdown") is True
        assert discovery._is_readme_file("guide.md") is False
    
    def test_file_priority_key(self):
        """Test file priority calculation."""
        discovery = MarkdownDiscovery()
        
        readme_file = {
            'is_readme': True,
            'word_count': 1000
        }
        
        regular_file = {
            'is_readme': False,
            'word_count': 2000
        }
        
        readme_priority = discovery._file_priority_key(readme_file)
        regular_priority = discovery._file_priority_key(regular_file)
        
        # README files should have higher priority
        assert readme_priority[0] > regular_priority[0]
    
    @patch('utils.github_processor.os.walk')
    @patch('utils.github_processor.os.stat')
    @patch('builtins.open', new_callable=mock_open)
    def test_discover_markdown_files(self, mock_file, mock_stat, mock_walk):
        """Test markdown file discovery."""
        # Setup mocks
        mock_walk.return_value = [
            ("/repo", ["docs", ".git"], ["README.md", "script.py"]),
            ("/repo/docs", [], ["guide.md", "api.md"])
        ]
        
        mock_stat.return_value = Mock(st_size=5000)
        mock_file.return_value.read.return_value = "# Test\nThis is test content with enough words to pass the minimum."
        
        discovery = MarkdownDiscovery()
        
        # Test
        result = discovery.discover_markdown_files("/repo", max_files=10)
        
        # Verify
        assert len(result) > 0
        assert all(f['filename'].endswith('.md') for f in result)
        assert all('content' in f for f in result)
        assert all('relative_path' in f for f in result)


class TestGitHubMetadataExtractor:
    """Test cases for GitHubMetadataExtractor class."""
    
    def test_init(self):
        """Test initialization of GitHubMetadataExtractor."""
        extractor = GitHubMetadataExtractor()
        assert extractor.logger is not None
    
    def test_parse_repo_info(self):
        """Test repository information parsing."""
        extractor = GitHubMetadataExtractor()
        
        # Test various URL formats
        owner, repo = extractor._parse_repo_info("https://github.com/user/repo")
        assert owner == "user"
        assert repo == "repo"
        
        owner, repo = extractor._parse_repo_info("https://github.com/org/project.git")
        assert owner == "org"
        assert repo == "project"
        
        # Test invalid URL
        with pytest.raises(ValueError):
            extractor._parse_repo_info("https://github.com/user")
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('utils.github_processor.os.path.exists')
    def test_extract_package_info_nodejs(self, mock_exists, mock_file):
        """Test package.json extraction."""
        # Setup mocks
        mock_exists.side_effect = lambda path: path.endswith('package.json')
        package_json = {
            "name": "test-package",
            "description": "A test package",
            "version": "1.0.0",
            "license": "MIT"
        }
        mock_file.return_value.read.return_value = str(package_json).replace("'", '"')
        
        extractor = GitHubMetadataExtractor()
        
        with patch('json.load', return_value=package_json):
            result = extractor._extract_package_info("/repo")
        
        assert result['language'] == 'javascript'
        assert result['package_name'] == 'test-package'
        assert result['description'] == 'A test package'
        assert result['version'] == '1.0.0'
        assert result['license'] == 'MIT'
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('utils.github_processor.os.path.exists')
    def test_extract_package_info_python(self, mock_exists, mock_file):
        """Test pyproject.toml extraction."""
        # Setup mocks
        mock_exists.side_effect = lambda path: path.endswith('pyproject.toml')
        toml_content = """
        [project]
        name = "test-package"
        description = "A Python test package"
        """
        mock_file.return_value.read.return_value = toml_content
        
        extractor = GitHubMetadataExtractor()
        result = extractor._extract_package_info("/repo")
        
        assert result['language'] == 'python'
        assert result['package_name'] == 'test-package'
        assert result['description'] == 'A Python test package'
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('utils.github_processor.os.path.exists')
    def test_extract_readme_info(self, mock_exists, mock_file):
        """Test README extraction."""
        # Setup mocks
        mock_exists.side_effect = lambda path: path.endswith('README.md')
        readme_content = """# Test Project
        
        This is a test project for demonstration.
        
        ## Features
        - Feature 1
        - Feature 2
        """
        mock_file.return_value.read.return_value = readme_content
        
        extractor = GitHubMetadataExtractor()
        result = extractor._extract_readme_info("/repo")
        
        assert result['readme_title'] == 'Test Project'
    
    @patch('utils.github_processor.subprocess.run')
    def test_extract_git_info(self, mock_subprocess):
        """Test Git information extraction."""
        # Setup mock
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="abc123|Initial commit|2024-01-15 10:30:00 +0000"
        )
        
        extractor = GitHubMetadataExtractor()
        result = extractor._extract_git_info("/repo")
        
        assert result['latest_commit_hash'] == 'abc123'
        assert result['latest_commit_message'] == 'Initial commit'
        assert result['latest_commit_date'] == '2024-01-15 10:30:00 +0000'
    
    def test_extract_repo_metadata_integration(self):
        """Test complete repository metadata extraction."""
        extractor = GitHubMetadataExtractor()
        
        with patch.object(extractor, '_parse_repo_info', return_value=('user', 'repo')), \
             patch.object(extractor, '_extract_package_info', return_value={'language': 'python'}), \
             patch.object(extractor, '_extract_readme_info', return_value={'readme_title': 'Test'}), \
             patch.object(extractor, '_extract_git_info', return_value={'latest_commit_hash': 'abc123'}):
            
            result = extractor.extract_repo_metadata("https://github.com/user/repo", "/repo")
            
            assert result['repo_url'] == "https://github.com/user/repo"
            assert result['owner'] == 'user'
            assert result['repo_name'] == 'repo'
            assert result['full_name'] == 'user/repo'
            assert result['source_type'] == 'github_repository'
            assert result['language'] == 'python'
            assert result['readme_title'] == 'Test'
            assert result['latest_commit_hash'] == 'abc123'


class TestValidationFunctions:
    """Test cases for validation functions."""
    
    def test_validate_github_url_valid(self):
        """Test validation of valid GitHub URLs."""
        valid_urls = [
            "https://github.com/user/repo",
            "https://github.com/user/repo.git",
            "https://www.github.com/user/repo",
            "http://github.com/org/project",
            "https://github.com/user/repo/tree/main",
            "https://github.com/user/repo/blob/main/README.md"
        ]
        
        for url in valid_urls:
            is_valid, error = validate_github_url(url)
            assert is_valid, f"URL should be valid: {url}, Error: {error}"
            assert error == ""
    
    def test_validate_github_url_invalid(self):
        """Test validation of invalid GitHub URLs."""
        invalid_urls = [
            ("", "URL must be a non-empty string"),
            ("not-a-url", "URL must use http or https scheme"),
            ("https://gitlab.com/user/repo", "URL must be from github.com"),
            ("https://github.com/user", "URL must include both owner and repository name"),
            ("https://github.com/", "URL must include repository path"),
            ("ftp://github.com/user/repo", "URL must use http or https scheme"),
            ("https://github.com/user-/repo", "Invalid GitHub owner/organization name format"),
            ("https://github.com/user/repo-", "Invalid GitHub repository name format")
        ]
        
        for url, expected_error in invalid_urls:
            is_valid, error = validate_github_url(url)
            assert not is_valid, f"URL should be invalid: {url}"
            assert expected_error in error, f"Expected error '{expected_error}' in '{error}'"
    
    def test_normalize_github_url(self):
        """Test GitHub URL normalization."""
        test_cases = [
            ("https://github.com/user/repo", "https://github.com/user/repo.git"),
            ("https://github.com/user/repo.git", "https://github.com/user/repo.git"),
            ("https://github.com/user/repo/tree/main", "https://github.com/user/repo.git"),
            ("https://www.github.com/user/repo", "https://github.com/user/repo.git")
        ]
        
        for input_url, expected in test_cases:
            result = normalize_github_url(input_url)
            assert result == expected, f"Expected {expected}, got {result}"
    
    def test_normalize_github_url_invalid(self):
        """Test normalization with invalid URLs."""
        with pytest.raises(ValueError, match="Invalid GitHub URL"):
            normalize_github_url("https://gitlab.com/user/repo")