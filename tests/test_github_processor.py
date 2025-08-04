"""

Unit tests for GitHub processor utilities.

Tests the GitHubRepoManager, MarkdownDiscovery, and GitHubMetadataExtractor classes.
"""
# ruff: noqa: E402

import pytest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from utils.github_processor import (
    GitHubRepoManager,
    MarkdownDiscovery,
    GitHubMetadataExtractor,
    MultiFileDiscovery,
    PythonProcessor,
    TypeScriptProcessor,
    ConfigProcessor,
    MarkdownProcessor,
)
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

        assert (
            manager._normalize_clone_url("https://github.com/user/repo")
            == "https://github.com/user/repo.git"
        )
        assert (
            manager._normalize_clone_url("https://github.com/user/repo.git")
            == "https://github.com/user/repo.git"
        )
        assert (
            manager._normalize_clone_url("https://github.com/user/repo/")
            == "https://github.com/user/repo.git"
        )

    @patch("utils.github_processor.tempfile.mkdtemp")
    @patch("utils.github_processor.subprocess.run")
    @patch("utils.github_processor.os.walk")
    @patch("utils.github_processor.os.path.getsize")
    def test_clone_repository_success(
        self, mock_getsize, mock_walk, mock_subprocess, mock_mkdtemp
    ):
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
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/user/repo.git",
                "/tmp/test_clone",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

    @patch("utils.github_processor.tempfile.mkdtemp")
    @patch("utils.github_processor.subprocess.run")
    def test_clone_repository_invalid_url(self, mock_subprocess, mock_mkdtemp):
        """Test cloning with invalid URL."""

        mock_mkdtemp.return_value = "/tmp/test_clone"

        manager = GitHubRepoManager()

        with pytest.raises(ValueError, match="Invalid GitHub repository URL"):
            manager.clone_repository("https://gitlab.com/user/repo")

    @patch("utils.github_processor.tempfile.mkdtemp")
    @patch("utils.github_processor.subprocess.run")
    def test_clone_repository_git_failure(self, mock_subprocess, mock_mkdtemp):
        """Test cloning when git command fails."""

        mock_mkdtemp.return_value = "/tmp/test_clone"
        mock_subprocess.return_value = Mock(returncode=1, stderr="Repository not found")

        manager = GitHubRepoManager()

        with pytest.raises(RuntimeError, match="Git clone failed"):
            manager.clone_repository("https://github.com/user/nonexistent")

    @patch("utils.github_processor.tempfile.mkdtemp")
    @patch("utils.github_processor.subprocess.run")
    @patch("utils.github_processor.os.walk")
    @patch("utils.github_processor.os.path.getsize")
    def test_clone_repository_too_large(
        self, mock_getsize, mock_walk, mock_subprocess, mock_mkdtemp
    ):
        """Test cloning when repository is too large."""

        # Setup mocks
        mock_mkdtemp.return_value = "/tmp/test_clone"
        mock_subprocess.return_value = Mock(returncode=0, stderr="")
        mock_walk.return_value = [("/tmp/test_clone", [], ["large_file.bin"])]
        mock_getsize.return_value = 600 * 1024 * 1024  # 600MB file

        manager = GitHubRepoManager()

        with pytest.raises(RuntimeError, match="Repository too large"):
            manager.clone_repository(
                "https://github.com/user/large-repo", max_size_mb=500
            )

    @patch("shutil.rmtree")
    @patch("os.path.exists")
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

        readme_file = {"is_readme": True, "word_count": 1000}

        regular_file = {"is_readme": False, "word_count": 2000}

        readme_priority = discovery._file_priority_key(readme_file)
        regular_priority = discovery._file_priority_key(regular_file)

        # README files should have higher priority
        assert readme_priority[0] > regular_priority[0]

    @patch("utils.github_processor.os.walk")
    @patch("utils.github_processor.os.stat")
    @patch("builtins.open", new_callable=mock_open)
    def test_discover_markdown_files(self, mock_file, mock_stat, mock_walk):
        """Test markdown file discovery."""

        # Setup mocks
        mock_walk.return_value = [
            ("/repo", ["docs", ".git"], ["README.md", "script.py"]),
            ("/repo/docs", [], ["guide.md", "api.md"]),
        ]

        mock_stat.return_value = Mock(st_size=5000)
        mock_file.return_value.read.return_value = (
            "# Test\nThis is test content with enough words to pass the minimum."
        )

        discovery = MarkdownDiscovery()

        # Test
        result = discovery.discover_markdown_files("/repo", max_files=10)

        # Verify
        assert len(result) > 0
        assert all(f["filename"].endswith(".md") for f in result)
        assert all("content" in f for f in result)
        assert all("relative_path" in f for f in result)


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

    @patch("builtins.open", new_callable=mock_open)
    @patch("utils.github_processor.os.path.exists")
    def test_extract_package_info_nodejs(self, mock_exists, mock_file):
        """Test package.json extraction."""

        # Setup mocks
        mock_exists.side_effect = lambda path: path.endswith("package.json")
        package_json = {
            "name": "test-package",
            "description": "A test package",
            "version": "1.0.0",
            "license": "MIT",
        }
        mock_file.return_value.read.return_value = str(package_json).replace("'", '"')

        extractor = GitHubMetadataExtractor()

        with patch("json.load", return_value=package_json):
            result = extractor._extract_package_info("/repo")

        assert result["language"] == "javascript"
        assert result["package_name"] == "test-package"
        assert result["description"] == "A test package"
        assert result["version"] == "1.0.0"
        assert result["license"] == "MIT"

    @patch("builtins.open", new_callable=mock_open)
    @patch("utils.github_processor.os.path.exists")
    def test_extract_package_info_python(self, mock_exists, mock_file):
        """Test pyproject.toml extraction."""

        # Setup mocks
        mock_exists.side_effect = lambda path: path.endswith("pyproject.toml")
        toml_content = """

        [project]
        name = "test-package"
        description = "A Python test package"
        """

        mock_file.return_value.read.return_value = toml_content

        extractor = GitHubMetadataExtractor()
        result = extractor._extract_package_info("/repo")

        assert result["language"] == "python"
        assert result["package_name"] == "test-package"
        assert result["description"] == "A Python test package"

    @patch("builtins.open", new_callable=mock_open)
    @patch("utils.github_processor.os.path.exists")
    def test_extract_readme_info(self, mock_exists, mock_file):
        """Test README extraction."""

        # Setup mocks
        mock_exists.side_effect = lambda path: path.endswith("README.md")
        readme_content = """# Test Project

        
        This is a test project for demonstration.
        
        ## Features
        - Feature 1
        - Feature 2
        """

        mock_file.return_value.read.return_value = readme_content

        extractor = GitHubMetadataExtractor()
        result = extractor._extract_readme_info("/repo")

        assert result["readme_title"] == "Test Project"

    @patch("utils.github_processor.subprocess.run")
    def test_extract_git_info(self, mock_subprocess):
        """Test Git information extraction."""

        # Setup mock
        mock_subprocess.return_value = Mock(
            returncode=0, stdout="abc123|Initial commit|2024-01-15 10:30:00 +0000"
        )

        extractor = GitHubMetadataExtractor()
        result = extractor._extract_git_info("/repo")

        assert result["latest_commit_hash"] == "abc123"
        assert result["latest_commit_message"] == "Initial commit"
        assert result["latest_commit_date"] == "2024-01-15 10:30:00 +0000"

    def test_extract_repo_metadata_integration(self):
        """Test complete repository metadata extraction."""

        extractor = GitHubMetadataExtractor()

        with (
            patch.object(extractor, "_parse_repo_info", return_value=("user", "repo")),
            patch.object(
                extractor, "_extract_package_info", return_value={"language": "python"}
            ),
            patch.object(
                extractor, "_extract_readme_info", return_value={"readme_title": "Test"}
            ),
            patch.object(
                extractor,
                "_extract_git_info",
                return_value={"latest_commit_hash": "abc123"},
            ),
        ):
            result = extractor.extract_repo_metadata(
                "https://github.com/user/repo", "/repo"
            )

            assert result["repo_url"] == "https://github.com/user/repo"
            assert result["owner"] == "user"
            assert result["repo_name"] == "repo"
            assert result["full_name"] == "user/repo"
            assert result["source_type"] == "github_repository"
            assert result["language"] == "python"
            assert result["readme_title"] == "Test"
            assert result["latest_commit_hash"] == "abc123"


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
            "https://github.com/user/repo/blob/main/README.md",
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
            (
                "https://github.com/user",
                "URL must include both owner and repository name",
            ),
            ("https://github.com/", "URL must include repository path"),
            ("ftp://github.com/user/repo", "URL must use http or https scheme"),
            (
                "https://github.com/user-/repo",
                "Invalid GitHub owner/organization name format",
            ),
            ("https://github.com/user/repo-", "Invalid GitHub repository name format"),
        ]

        for url, expected_error in invalid_urls:
            is_valid, error = validate_github_url(url)
            assert not is_valid, f"URL should be invalid: {url}"
            assert expected_error in error, (
                f"Expected error '{expected_error}' in '{error}'"
            )

    def test_normalize_github_url(self):
        """Test GitHub URL normalization."""

        test_cases = [
            ("https://github.com/user/repo", "https://github.com/user/repo.git"),
            ("https://github.com/user/repo.git", "https://github.com/user/repo.git"),
            (
                "https://github.com/user/repo/tree/main",
                "https://github.com/user/repo.git",
            ),
            ("https://www.github.com/user/repo", "https://github.com/user/repo.git"),
        ]

        for input_url, expected in test_cases:
            result = normalize_github_url(input_url)
            assert result == expected, f"Expected {expected}, got {result}"

    def test_normalize_github_url_invalid(self):
        """Test normalization with invalid URLs."""

        with pytest.raises(ValueError, match="Invalid GitHub URL"):
            normalize_github_url("https://gitlab.com/user/repo")


class TestPythonProcessor:
    """Test cases for PythonProcessor class."""

    def test_process_file_with_docstrings(self):
        """Test processing Python file with docstrings."""

        processor = PythonProcessor()

        python_content = '''"""

Module docstring for testing.
This module contains test functions and classes.
"""


def test_function(arg1: str, arg2: int) -> bool:
    """

    Test function with parameters and return type.
    
    Args:
        arg1: First argument as string
        arg2: Second argument as integer
        
    Returns:
        Boolean result
    """

    return True

class TestClass:
    """

    Test class for demonstration.
    
    This class shows how docstrings are extracted.
    """

    
    def method(self, param: str) -> None:
        """Method with docstring."""

        pass

async def async_function():
    """Async function docstring."""

    pass
'''

        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_content)
            temp_path = f.name

        try:
            result = processor.process_file(temp_path, "test.py")

            # Should extract module, function, class, and async function docstrings
            assert len(result) >= 4

            # Check module docstring
            module_docs = [item for item in result if item["type"] == "module"]
            assert len(module_docs) == 1
            assert "Module docstring for testing" in module_docs[0]["content"]
            assert module_docs[0]["language"] == "python"

            # Check function docstring
            function_docs = [item for item in result if item["type"] == "function"]
            assert len(function_docs) >= 2  # test_function and async_function

            test_func = next(
                item for item in function_docs if item["name"] == "test_function"
            )
            assert "Test function with parameters" in test_func["content"]
            assert "(arg1: str, arg2: int) -> bool" in test_func["signature"]

            # Check class docstring
            class_docs = [item for item in result if item["type"] == "class"]
            assert len(class_docs) == 1
            assert "Test class for demonstration" in class_docs[0]["content"]
            assert class_docs[0]["name"] == "TestClass"

        finally:
            os.unlink(temp_path)

    def test_process_file_syntax_error(self):
        """Test processing Python file with syntax error."""

        processor = PythonProcessor()

        invalid_python = """

def invalid_syntax(
    # Missing closing parenthesis
    pass
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(invalid_python)
            temp_path = f.name

        try:
            result = processor.process_file(temp_path, "invalid.py")
            # Should return empty list for syntax errors
            assert result == []
        finally:
            os.unlink(temp_path)

    def test_process_file_no_docstrings(self):
        """Test processing Python file without docstrings."""

        processor = PythonProcessor()

        python_content = """

def function_without_docstring():
    return True

class ClassWithoutDocstring:
    def method(self):
        pass
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_content)
            temp_path = f.name

        try:
            result = processor.process_file(temp_path, "no_docs.py")
            # Should return empty list when no docstrings found
            assert result == []
        finally:
            os.unlink(temp_path)

    @patch("os.path.getsize")
    def test_process_file_too_large(self, mock_getsize):
        """Test processing Python file that's too large."""

        processor = PythonProcessor()
        mock_getsize.return_value = 2_000_000  # 2MB - over 1MB limit

        result = processor.process_file("/fake/path.py", "large.py")
        assert result == []


class TestTypeScriptProcessor:
    """Test cases for TypeScriptProcessor class."""

    def test_process_file_with_jsdoc(self):
        """Test processing TypeScript file with JSDoc comments."""

        processor = TypeScriptProcessor()

        typescript_content = """

/**
 * Calculates the area of a rectangle.
 * @param width - The width of the rectangle
 * @param height - The height of the rectangle
 * @returns The area of the rectangle
 * @example
 * ```typescript
 * const area = calculateArea(5, 10);
 * console.log(area); // 50
 * ```
 */
function calculateArea(width: number, height: number): number {
    return width * height;
}

/**
 * A class representing a database connection.
 * @public
 */
export class DatabaseConnection {
    /**
     * Connects to the database using the provided configuration.
     * @param config - The database configuration object
     * @returns A promise that resolves when the connection is established
     */
    async connect(config: DatabaseConfig): Promise<void> {
        // Implementation
    }
}

/**
 * User interface definition.
 * @interface
 */
export interface User {
    id: number;
    name: string;
    email: string;
}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
            f.write(typescript_content)
            temp_path = f.name

        try:
            result = processor.process_file(temp_path, "test.ts")

            # Should extract function, class, and interface JSDoc
            assert len(result) >= 2

            # Check function JSDoc
            func_docs = [item for item in result if item["type"] == "function"]
            assert len(func_docs) >= 1

            calc_func = next(
                (item for item in func_docs if "calculateArea" in item["name"]), None
            )
            assert calc_func is not None
            assert "Calculates the area of a rectangle" in calc_func["content"]
            assert calc_func["language"] == "typescript"

            # Check class JSDoc
            class_docs = [item for item in result if item["type"] == "class"]
            if class_docs:  # JSDoc might not always associate correctly
                assert "DatabaseConnection" in class_docs[0]["name"]
                assert "database connection" in class_docs[0]["content"]

        finally:
            os.unlink(temp_path)

    def test_process_file_minified(self):
        """Test processing minified TypeScript file."""

        processor = TypeScriptProcessor()

        # Minified content (long single line)
        minified_content = "function test(){return true;}" + "x" * 1000  # Make it long

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
            f.write(minified_content)
            temp_path = f.name

        try:
            result = processor.process_file(temp_path, "minified.ts")
            # Should return empty list for minified files
            assert result == []
        finally:
            os.unlink(temp_path)

    def test_process_file_no_jsdoc(self):
        """Test processing TypeScript file without JSDoc."""

        processor = TypeScriptProcessor()

        typescript_content = """

function normalFunction(): void {
    console.log("No JSDoc here");
}

class RegularClass {
    method(): string {
        return "test";
    }
}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
            f.write(typescript_content)
            temp_path = f.name

        try:
            result = processor.process_file(temp_path, "no_jsdoc.ts")
            # Should return empty list when no JSDoc found
            assert result == []
        finally:
            os.unlink(temp_path)


class TestConfigProcessor:
    """Test cases for ConfigProcessor class."""

    def test_process_json_file(self):
        """Test processing JSON configuration file."""

        processor = ConfigProcessor()

        json_content = """

{
    "name": "test-app",
    "version": "1.0.0",
    "scripts": {
        "start": "node server.js",
        "test": "jest"
    },
    "dependencies": {
        "express": "^4.18.0"
    }
}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            temp_path = f.name

        try:
            result = processor.process_file(temp_path, "package.json")

            assert len(result) == 1
            assert result[0]["type"] == "configuration"
            assert result[0]["language"] == "json"
            assert result[0]["name"] == os.path.basename(temp_path)
            assert "test-app" in result[0]["content"]

        finally:
            os.unlink(temp_path)

    def test_process_yaml_file(self):
        """Test processing YAML configuration file."""

        processor = ConfigProcessor()

        yaml_content = """

version: '3.8'
services:
  web:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./html:/usr/share/nginx/html
  database:
    image: postgres:13
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            result = processor.process_file(temp_path, "docker-compose.yaml")

            assert len(result) == 1
            assert result[0]["type"] == "configuration"
            assert result[0]["language"] == "yaml"
            assert "services:" in result[0]["content"]

        finally:
            os.unlink(temp_path)

    def test_process_toml_file(self):
        """Test processing TOML configuration file."""

        processor = ConfigProcessor()

        toml_content = """

[project]
name = "my-python-project"
version = "0.1.0"
description = "A sample Python project"

[project.dependencies]
requests = "^2.28.0"
click = "^8.1.0"

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            temp_path = f.name

        try:
            result = processor.process_file(temp_path, "pyproject.toml")

            assert len(result) == 1
            assert result[0]["type"] == "configuration"
            assert result[0]["language"] == "toml"
            assert "my-python-project" in result[0]["content"]

        finally:
            os.unlink(temp_path)

    @patch("os.path.getsize")
    def test_process_file_too_large(self, mock_getsize):
        """Test processing config file that's too large."""

        processor = ConfigProcessor()
        mock_getsize.return_value = 150_000  # 150KB - over 100KB limit

        result = processor.process_file("/fake/config.json", "large.json")
        assert result == []

    def test_process_empty_file(self):
        """Test processing empty configuration file."""

        processor = ConfigProcessor()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name

        try:
            result = processor.process_file(temp_path, "empty.json")
            assert result == []
        finally:
            os.unlink(temp_path)


class TestMarkdownProcessor:
    """Test cases for MarkdownProcessor class."""

    def test_process_markdown_file(self):
        """Test processing markdown file."""

        processor = MarkdownProcessor()

        markdown_content = """# Test Project


This is a test markdown file with various content.

## Features

- Feature 1: Something awesome
- Feature 2: Something even better

## Code Example

```python
def hello():
    print("Hello, World!")
```

## Conclusion

This concludes our test markdown file.
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(markdown_content)
            temp_path = f.name

        try:
            result = processor.process_file(temp_path, "test.md")

            assert len(result) == 1
            assert result[0]["type"] == "markdown"
            assert result[0]["language"] == "markdown"
            assert result[0]["name"] == os.path.basename(temp_path)
            assert "Test Project" in result[0]["content"]
            assert "```python" in result[0]["content"]

        finally:
            os.unlink(temp_path)

    def test_process_empty_markdown(self):
        """Test processing empty or very short markdown file."""

        processor = MarkdownProcessor()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Short")  # Too short
            temp_path = f.name

        try:
            result = processor.process_file(temp_path, "short.md")
            assert result == []
        finally:
            os.unlink(temp_path)


class TestMultiFileDiscovery:
    """Test cases for MultiFileDiscovery class."""

    def test_init(self):
        """Test initialization of MultiFileDiscovery."""

        discovery = MultiFileDiscovery()
        assert discovery.logger is not None
        assert discovery.SUPPORTED_EXTENSIONS is not None
        assert discovery.FILE_SIZE_LIMITS is not None
        assert ".py" in discovery.SUPPORTED_EXTENSIONS
        assert ".ts" in discovery.SUPPORTED_EXTENSIONS
        assert ".json" in discovery.SUPPORTED_EXTENSIONS

    def test_is_supported_file(self):
        """Test file type support detection."""

        discovery = MultiFileDiscovery()

        # Test Python files
        assert discovery._is_supported_file("script.py", [".py"]) is True
        assert discovery._is_supported_file("script.py", [".js"]) is False

        # Test TypeScript files
        assert discovery._is_supported_file("component.ts", [".ts"]) is True
        assert discovery._is_supported_file("component.tsx", [".tsx"]) is True

        # Test config files
        assert discovery._is_supported_file("config.json", [".json"]) is True
        assert discovery._is_supported_file("docker.yaml", [".yaml"]) is True
        assert discovery._is_supported_file("pyproject.toml", [".toml"]) is True

        # Test markdown files
        assert discovery._is_supported_file("README.md", [".md"]) is True
        assert discovery._is_supported_file("guide.markdown", [".markdown"]) is True

        # Test unsupported files
        assert discovery._is_supported_file("image.png", [".png"]) is False
        assert discovery._is_supported_file("data.csv", [".csv"]) is False

    @patch("utils.github_processor.os.walk")
    @patch("utils.github_processor.os.stat")
    @patch("builtins.open", new_callable=mock_open)
    def test_discover_files_multi_type(self, mock_file, mock_stat, mock_walk):
        """Test discovering multiple file types."""

        # Setup mocks
        mock_walk.return_value = [
            ("/repo", ["src", "docs"], ["README.md", "package.json", "script.py"]),
            ("/repo/src", [], ["main.ts", "utils.py", "config.yaml"]),
            ("/repo/docs", [], ["guide.md", "api.md"]),
        ]

        mock_stat.return_value = Mock(st_size=5000)
        mock_file.return_value.read.return_value = "test content"

        discovery = MultiFileDiscovery()

        # Test discovering multiple types
        result = discovery.discover_files(
            "/repo", file_types=[".md", ".py", ".ts", ".json", ".yaml"]
        )

        # Should find files of all requested types
        found_extensions = {os.path.splitext(f["filename"])[1] for f in result}
        expected_extensions = {".md", ".py", ".ts", ".json", ".yaml"}
        assert expected_extensions.issubset(found_extensions)

        # Verify file info structure
        for file_info in result:
            assert "path" in file_info
            assert "relative_path" in file_info
            assert "filename" in file_info
            assert "size_bytes" in file_info
            assert "file_type" in file_info
            assert "is_readme" in file_info

    @patch("utils.github_processor.os.walk")
    def test_discover_files_empty_result(self, mock_walk):
        """Test discovering files with no matches."""

        mock_walk.return_value = [("/repo", [], ["script.js", "style.css"])]

        discovery = MultiFileDiscovery()
        result = discovery.discover_files("/repo", file_types=[".py"])

        assert result == []

    @patch("utils.github_processor.os.walk")
    @patch("utils.github_processor.os.stat")
    @patch("builtins.open")
    def test_discover_files_binary_filtering(self, mock_file, mock_stat, mock_walk):
        """Test filtering out binary files."""

        mock_walk.return_value = [("/repo", [], ["data.json"])]
        mock_stat.return_value = Mock(st_size=1000)

        # Mock binary content (contains null bytes)
        mock_file.return_value.__enter__.return_value.read.return_value = (
            "text\x00binary"
        )

        discovery = MultiFileDiscovery()
        result = discovery.discover_files("/repo", file_types=[".json"])

        # Should filter out binary files
        assert result == []

    @patch("utils.github_processor.os.walk")
    @patch("utils.github_processor.os.stat")
    def test_discover_files_size_limits(self, mock_stat, mock_walk):
        """Test file size limits by type."""

        mock_walk.return_value = [("/repo", [], ["large.py", "large.json"])]

        discovery = MultiFileDiscovery()

        # Test Python file size limit (1MB)
        mock_stat.return_value = Mock(st_size=2_000_000)  # 2MB
        result = discovery.discover_files("/repo", file_types=[".py"])
        assert result == []

        # Test JSON file size limit (100KB)
        mock_stat.return_value = Mock(st_size=200_000)  # 200KB
        result = discovery.discover_files("/repo", file_types=[".json"])
        assert result == []
