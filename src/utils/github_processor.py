"""
GitHub repository processing utilities for the Crawl4AI MCP server.

This module provides functionality to clone GitHub repositories, discover markdown files,
and extract metadata for storing in the vector database.
"""

import os
import re
import shutil
import stat
import subprocess
import tempfile
import logging
import ast
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse


class GitHubRepoManager:
    """Manages GitHub repository cloning and cleanup operations."""

    def __init__(self):
        self.temp_dirs: List[str] = []
        self.logger = logging.getLogger(__name__)

    def clone_repository(self, repo_url: str, max_size_mb: int = 500) -> str:
        """
        Clone a GitHub repository to a temporary directory with size checks.

        Args:
            repo_url: GitHub repository URL
            max_size_mb: Maximum repository size in MB (default: 500MB)

        Returns:
            Path to the cloned repository directory

        Raises:
            ValueError: If repository URL is invalid
            RuntimeError: If cloning fails or repository is too large
        """
        # Validate GitHub URL
        if not self._is_valid_github_url(repo_url):
            raise ValueError(f"Invalid GitHub repository URL: {repo_url}")

        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="github_clone_")
        self.temp_dirs.append(temp_dir)

        try:
            # Normalize URL for cloning
            clone_url = self._normalize_clone_url(repo_url)

            # Clone with depth=1 for efficiency
            self.logger.info(f"Cloning repository: {clone_url}")
            result = subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, temp_dir],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"Git clone failed: {result.stderr}")

            # Check repository size
            repo_size_mb = self._get_directory_size_mb(temp_dir)
            if repo_size_mb > max_size_mb:
                raise RuntimeError(
                    f"Repository too large: {repo_size_mb:.1f}MB exceeds limit of {max_size_mb}MB"
                )

            self.logger.info(f"Successfully cloned repository ({repo_size_mb:.1f}MB)")
            return temp_dir

        except Exception as e:
            # Clean up on failure
            self._cleanup_directory(temp_dir)
            if temp_dir in self.temp_dirs:
                self.temp_dirs.remove(temp_dir)
            raise e

    def cleanup(self):
        """Clean up all temporary directories."""
        for temp_dir in self.temp_dirs:
            self._cleanup_directory(temp_dir)
        self.temp_dirs.clear()

    def _is_valid_github_url(self, url: str) -> bool:
        """Check if URL is a valid GitHub repository URL."""
        try:
            parsed = urlparse(url)
            if parsed.netloc not in ["github.com", "www.github.com"]:
                return False

            # Check path format: /owner/repo or /owner/repo.git
            path_parts = parsed.path.strip("/").split("/")
            if len(path_parts) < 2:
                return False

            # Basic validation - owner and repo should be non-empty
            owner, repo = path_parts[0], path_parts[1]
            if not owner or not repo:
                return False

            return True
        except Exception:
            return False

    def _normalize_clone_url(self, url: str) -> str:
        """Normalize GitHub URL for git clone."""
        url = url.rstrip("/")
        if not url.endswith(".git"):
            url += ".git"
        return url

    def _get_directory_size_mb(self, directory: str) -> float:
        """Calculate directory size in MB."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, IOError):
                    # Skip files that can't be accessed
                    continue
        return total_size / (1024 * 1024)

    def _cleanup_directory(self, directory: str):
        """Safely remove a directory and its contents."""
        try:
            if os.path.exists(directory):
                # Handle Windows read-only files (common with Git repositories)
                def handle_remove_readonly(func, path, exc):
                    """Handle read-only files on Windows."""
                    if os.path.exists(path):
                        # Clear the readonly bit and try again
                        os.chmod(path, stat.S_IWRITE)
                        func(path)

                shutil.rmtree(directory, onerror=handle_remove_readonly)
                self.logger.debug(f"Cleaned up directory: {directory}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup directory {directory}: {e}")


class MarkdownDiscovery:
    """Discovers and filters markdown files in a repository."""

    # Common directories to exclude from markdown discovery
    EXCLUDED_DIRS = {
        ".git",
        ".github",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "build",
        "dist",
        ".next",
        ".nuxt",
        "target",
        "vendor",
        ".cache",
        "coverage",
        ".coverage",
        "htmlcov",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        ".eggs",
        "*.egg-info",
        ".DS_Store",
        "Thumbs.db",
    }

    # File patterns to exclude
    EXCLUDED_PATTERNS = {
        "CHANGELOG*",
        "HISTORY*",
        "NEWS*",
        "RELEASES*",
        "*.lock",
        "package-lock.json",
        "yarn.lock",
        "Gemfile.lock",
        "*.min.*",
        "*.bundle.*",
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def discover_markdown_files(
        self,
        repo_path: str,
        max_files: int = 100,
        min_size_bytes: int = 100,
        max_size_bytes: int = 1_000_000,  # 1MB
    ) -> List[Dict[str, Any]]:
        """
        Discover markdown files in the repository with filtering.

        Args:
            repo_path: Path to the cloned repository
            max_files: Maximum number of files to process
            min_size_bytes: Minimum file size in bytes
            max_size_bytes: Maximum file size in bytes

        Returns:
            List of dictionaries containing file information
        """
        markdown_files = []
        processed_count = 0

        try:
            for root, dirs, files in os.walk(repo_path):
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if not self._should_exclude_dir(d)]

                for file in files:
                    if processed_count >= max_files:
                        break

                    if self._is_markdown_file(file):
                        file_path = os.path.join(root, file)
                        file_info = self._analyze_markdown_file(
                            file_path, repo_path, min_size_bytes, max_size_bytes
                        )

                        if file_info:
                            markdown_files.append(file_info)
                            processed_count += 1

                if processed_count >= max_files:
                    break

            # Sort by priority (README files first, then by size)
            markdown_files.sort(key=self._file_priority_key, reverse=True)

            self.logger.info(f"Discovered {len(markdown_files)} markdown files")
            return markdown_files

        except Exception as e:
            self.logger.error(f"Error discovering markdown files: {e}")
            return []

    def _is_markdown_file(self, filename: str) -> bool:
        """Check if file is a markdown file."""
        return filename.lower().endswith((".md", ".markdown", ".mdown", ".mkd"))

    def _should_exclude_dir(self, dirname: str) -> bool:
        """Check if directory should be excluded."""
        return dirname in self.EXCLUDED_DIRS or dirname.startswith(".")

    def _should_exclude_file(self, filename: str) -> bool:
        """Check if file should be excluded based on patterns."""
        import fnmatch

        for pattern in self.EXCLUDED_PATTERNS:
            if fnmatch.fnmatch(filename.lower(), pattern.lower()):
                return True
        return False

    def _analyze_markdown_file(
        self, file_path: str, repo_path: str, min_size: int, max_size: int
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a markdown file and return its metadata.

        Args:
            file_path: Absolute path to the file
            repo_path: Path to the repository root
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes

        Returns:
            Dictionary with file metadata or None if file should be skipped
        """
        try:
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size

            # Size filtering
            if file_size < min_size or file_size > max_size:
                return None

            # Read file content
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Skip if content is too short or looks like binary
            if len(content.strip()) < 50:
                return None

            # Calculate relative path
            relative_path = os.path.relpath(file_path, repo_path)

            return {
                "path": file_path,
                "relative_path": relative_path,
                "filename": os.path.basename(file_path),
                "size_bytes": file_size,
                "content": content,
                "word_count": len(content.split()),
                "is_readme": self._is_readme_file(os.path.basename(file_path)),
            }

        except Exception as e:
            self.logger.warning(f"Error analyzing file {file_path}: {e}")
            return None

    def _is_readme_file(self, filename: str) -> bool:
        """Check if file is a README file."""
        return filename.lower().startswith("readme")

    def _file_priority_key(self, file_info: Dict[str, Any]) -> Tuple[int, int]:
        """Generate priority key for sorting files."""
        # README files get highest priority
        readme_priority = 1 if file_info["is_readme"] else 0

        # Size priority (moderate size preferred)
        size_priority = min(file_info["word_count"], 5000)

        return (readme_priority, size_priority)


class FileTypeProcessor:
    """Base class for file type processors."""

    def process_file(self, file_path: str, relative_path: str) -> List[Dict[str, Any]]:
        """Process file and return list of extractable content chunks."""
        raise NotImplementedError


class MarkdownProcessor(FileTypeProcessor):
    """Process markdown files using existing content."""

    def process_file(self, file_path: str, relative_path: str) -> List[Dict[str, Any]]:
        """Process markdown file and return content."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()

            if len(content) < 50:
                return []

            return [
                {
                    "content": content,
                    "type": "markdown",
                    "name": os.path.basename(file_path),
                    "signature": None,
                    "line_number": 1,
                    "language": "markdown",
                }
            ]

        except Exception:
            return []


class PythonProcessor(FileTypeProcessor):
    """Process Python files using AST for docstring extraction."""

    def process_file(self, file_path: str, relative_path: str) -> List[Dict[str, Any]]:
        """Extract docstrings from Python file using AST."""
        try:
            # Size check - skip large Python files
            file_size = os.path.getsize(file_path)
            if file_size > 1_000_000:  # 1MB limit
                return []

            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=file_path)
            extracted_items = []

            # Module docstring
            module_doc = ast.get_docstring(tree, clean=True)
            if module_doc:
                extracted_items.append(
                    {
                        "content": module_doc,
                        "type": "module",
                        "name": relative_path,
                        "signature": None,
                        "line_number": 1,
                        "language": "python",
                    }
                )

            # Walk AST for functions and classes
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    docstring = ast.get_docstring(node, clean=True)
                    if docstring:
                        extracted_items.append(
                            {
                                "content": docstring,
                                "type": "function",
                                "name": node.name,
                                "signature": self._extract_signature(node),
                                "line_number": node.lineno,
                                "language": "python",
                            }
                        )

                elif isinstance(node, ast.ClassDef):
                    docstring = ast.get_docstring(node, clean=True)
                    if docstring:
                        extracted_items.append(
                            {
                                "content": docstring,
                                "type": "class",
                                "name": node.name,
                                "signature": None,
                                "line_number": node.lineno,
                                "language": "python",
                            }
                        )

            return extracted_items

        except SyntaxError:
            # Skip files with syntax errors
            return []
        except Exception:
            return []

    def _extract_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature with type annotations."""
        try:
            args = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                args.append(arg_str)

            signature = f"({', '.join(args)})"
            if node.returns:
                signature += f" -> {ast.unparse(node.returns)}"

            return signature
        except Exception:
            return "(signature_extraction_failed)"


class TypeScriptProcessor(FileTypeProcessor):
    """Process TypeScript files for JSDoc comments."""

    def process_file(self, file_path: str, relative_path: str) -> List[Dict[str, Any]]:
        """Extract JSDoc comments from TypeScript file."""
        try:
            # Size check - skip large TypeScript files
            file_size = os.path.getsize(file_path)
            if file_size > 1_000_000:  # 1MB limit
                return []

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check if file is minified
            first_line = content.split("\n")[0] if "\n" in content else content
            if len(first_line) > 1000:
                return []  # Skip minified files

            # JSDoc comment pattern
            jsdoc_pattern = re.compile(
                r"/\*\*\s*\n((?:\s*\*[^\n]*\n)*)\s*\*/", re.MULTILINE | re.DOTALL
            )

            # Declaration patterns
            declaration_patterns = {
                "function": re.compile(
                    r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)",
                    re.MULTILINE,
                ),
                "class": re.compile(r"(?:export\s+)?class\s+(\w+)", re.MULTILINE),
                "interface": re.compile(
                    r"(?:export\s+)?interface\s+(\w+)", re.MULTILINE
                ),
            }

            extracted_items = []

            for match in jsdoc_pattern.finditer(content):
                comment_text = match.group(1)
                start_pos = match.start()

                # Clean comment text
                lines = comment_text.split("\n")
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    if line.startswith("*"):
                        line = line[1:].strip()
                    if line:
                        cleaned_lines.append(line)

                cleaned_comment = "\n".join(cleaned_lines)
                line_number = content[:start_pos].count("\n") + 1

                # Find associated declaration
                after_comment = content[match.end() :]
                declaration = self._find_next_declaration(
                    after_comment, declaration_patterns
                )

                if declaration and cleaned_comment:
                    extracted_items.append(
                        {
                            "content": cleaned_comment,
                            "type": declaration["type"],
                            "name": declaration["name"],
                            "signature": declaration.get("signature", ""),
                            "line_number": line_number,
                            "language": "typescript",
                        }
                    )

            return extracted_items

        except Exception:
            return []

    def _find_next_declaration(
        self, content: str, declaration_patterns: Dict
    ) -> Optional[Dict[str, Any]]:
        """Find the next function/class/interface declaration."""
        # Remove leading whitespace and newlines
        content = content.lstrip()

        # Try each declaration pattern
        for decl_type, pattern in declaration_patterns.items():
            match = pattern.search(content)
            if match and match.start() < 200:  # Must be close to comment
                return {
                    "type": decl_type,
                    "name": match.group(1),
                    "signature": match.group(0),
                }

        return None


class ConfigProcessor(FileTypeProcessor):
    """Process configuration files with full content."""

    def process_file(self, file_path: str, relative_path: str) -> List[Dict[str, Any]]:
        """Process configuration file and return full content."""
        try:
            # Size check - skip large config files
            file_size = os.path.getsize(file_path)
            if file_size > 100_000:  # 100KB limit
                return []

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                return []

            # Determine file type
            ext = os.path.splitext(file_path)[1].lower()

            return [
                {
                    "content": content,
                    "type": "configuration",
                    "name": os.path.basename(file_path),
                    "signature": None,
                    "line_number": 1,
                    "language": self._get_config_language(ext),
                }
            ]

        except Exception:
            return []

    def _get_config_language(self, ext: str) -> str:
        """Map file extension to language."""
        mapping = {".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml"}
        return mapping.get(ext, "text")


class MultiFileDiscovery(MarkdownDiscovery):
    """Enhanced file discovery supporting multiple file types."""

    SUPPORTED_EXTENSIONS = {
        ".md",
        ".markdown",
        ".mdown",
        ".mkd",  # Markdown
        ".py",  # Python
        ".ts",
        ".tsx",  # TypeScript
        ".json",
        ".yaml",
        ".yml",
        ".toml",  # Configuration
    }

    # File size limits by type
    FILE_SIZE_LIMITS = {
        ".py": 1_000_000,  # 1MB for Python files
        ".ts": 1_000_000,  # 1MB for TypeScript files
        ".tsx": 1_000_000,  # 1MB for TypeScript files
        ".json": 100_000,  # 100KB for JSON files
        ".yaml": 100_000,  # 100KB for YAML files
        ".yml": 100_000,  # 100KB for YAML files
        ".toml": 100_000,  # 100KB for TOML files
        ".md": 1_000_000,  # 1MB for Markdown files
        ".markdown": 1_000_000,
        ".mdown": 1_000_000,
        ".mkd": 1_000_000,
    }

    def discover_files(
        self, repo_path: str, file_types: List[str] = [".md"], max_files: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Discover files of specified types with metadata.

        Args:
            repo_path: Path to the cloned repository
            file_types: List of file extensions to process
            max_files: Maximum number of files to process

        Returns:
            List of dictionaries containing file information
        """
        discovered_files = []
        processed_count = 0

        # Normalize file types
        file_types = [ft.lower() for ft in file_types]

        try:
            for root, dirs, files in os.walk(repo_path):
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if not self._should_exclude_dir(d)]

                for file in files:
                    if processed_count >= max_files:
                        break

                    if self._is_supported_file(file, file_types):
                        file_path = os.path.join(root, file)
                        file_info = self._analyze_file(file_path, repo_path, file_types)

                        if file_info:
                            discovered_files.append(file_info)
                            processed_count += 1

                if processed_count >= max_files:
                    break

            # Sort by priority (README files first, then by size)
            discovered_files.sort(key=self._file_priority_key, reverse=True)

            self.logger.info(
                f"Discovered {len(discovered_files)} files of types {file_types}"
            )
            return discovered_files

        except Exception as e:
            self.logger.error(f"Error discovering files: {e}")
            return []

    def _is_supported_file(self, filename: str, file_types: List[str]) -> bool:
        """Check if file is of a supported type."""
        file_ext = os.path.splitext(filename)[1].lower()

        # Check if extension is in requested file types
        if file_ext not in file_types:
            return False

        # Check if extension is in supported extensions
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            return False

        # Additional filtering for excluded patterns
        if self._should_exclude_file(filename):
            return False

        return True

    def _analyze_file(
        self, file_path: str, repo_path: str, file_types: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a file and return its metadata.

        Args:
            file_path: Absolute path to the file
            repo_path: Path to the repository root
            file_types: List of requested file types

        Returns:
            Dictionary with file metadata or None if file should be skipped
        """
        try:
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size
            file_ext = os.path.splitext(file_path)[1].lower()

            # Size filtering based on file type
            max_size = self.FILE_SIZE_LIMITS.get(file_ext, 1_000_000)
            if file_size > max_size:
                return None

            # Basic content check for text files
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    # Read first 1000 chars to check if it's text
                    sample = f.read(1000)
                    if "\x00" in sample:  # Binary file indicator
                        return None
            except UnicodeDecodeError:
                return None  # Skip non-UTF-8 files

            # Calculate relative path
            relative_path = os.path.relpath(file_path, repo_path)

            return {
                "path": file_path,
                "relative_path": relative_path,
                "filename": os.path.basename(file_path),
                "size_bytes": file_size,
                "file_type": file_ext,
                "is_readme": self._is_readme_file(os.path.basename(file_path)),
                "word_count": max(
                    1, file_size // 5
                ),  # Estimate word count for priority sorting
            }

        except Exception as e:
            self.logger.warning(f"Error analyzing file {file_path}: {e}")
            return None


class GitHubMetadataExtractor:
    """Extracts metadata from GitHub repositories."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_repo_metadata(self, repo_url: str, repo_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a GitHub repository.

        Args:
            repo_url: Original GitHub repository URL
            repo_path: Path to the cloned repository

        Returns:
            Dictionary containing repository metadata
        """
        try:
            # Parse repository info from URL
            owner, repo_name = self._parse_repo_info(repo_url)

            # Extract basic metadata
            metadata = {
                "repo_url": repo_url,
                "owner": owner,
                "repo_name": repo_name,
                "full_name": f"{owner}/{repo_name}",
                "source_type": "github_repository",
                "clone_path": repo_path,
            }

            # Try to extract additional metadata from common files
            metadata.update(self._extract_package_info(repo_path))
            metadata.update(self._extract_readme_info(repo_path))
            metadata.update(self._extract_git_info(repo_path))

            return metadata

        except Exception as e:
            self.logger.error(f"Error extracting repository metadata: {e}")
            return {
                "repo_url": repo_url,
                "source_type": "github_repository",
                "error": str(e),
            }

    def _parse_repo_info(self, repo_url: str) -> Tuple[str, str]:
        """Parse owner and repository name from GitHub URL."""
        parsed = urlparse(repo_url)
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) < 2:
            raise ValueError(f"Invalid GitHub URL format: {repo_url}")

        owner = path_parts[0]
        repo_name = path_parts[1]

        # Remove .git suffix if present
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        return owner, repo_name

    def _extract_package_info(self, repo_path: str) -> Dict[str, Any]:
        """Extract information from package files (package.json, pyproject.toml, etc.)."""
        metadata = {}

        # Check for package.json (Node.js)
        package_json_path = os.path.join(repo_path, "package.json")
        if os.path.exists(package_json_path):
            try:
                import json

                with open(package_json_path, "r", encoding="utf-8") as f:
                    package_data = json.load(f)

                metadata.update(
                    {
                        "language": "javascript",
                        "package_name": package_data.get("name"),
                        "description": package_data.get("description"),
                        "version": package_data.get("version"),
                        "license": package_data.get("license"),
                    }
                )
            except Exception as e:
                self.logger.warning(f"Error parsing package.json: {e}")

        # Check for pyproject.toml (Python)
        pyproject_path = os.path.join(repo_path, "pyproject.toml")
        if os.path.exists(pyproject_path):
            try:
                # Simple TOML parsing for basic info
                with open(pyproject_path, "r", encoding="utf-8") as f:
                    content = f.read()

                metadata["language"] = "python"

                # Extract name and description using regex (simple approach)
                name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
                if name_match:
                    metadata["package_name"] = name_match.group(1)

                desc_match = re.search(
                    r'description\s*=\s*["\']([^"\']+)["\']', content
                )
                if desc_match:
                    metadata["description"] = desc_match.group(1)

            except Exception as e:
                self.logger.warning(f"Error parsing pyproject.toml: {e}")

        # Check for Cargo.toml (Rust)
        cargo_path = os.path.join(repo_path, "Cargo.toml")
        if os.path.exists(cargo_path):
            try:
                with open(cargo_path, "r", encoding="utf-8") as f:
                    content = f.read()

                metadata["language"] = "rust"

                # Extract basic info using regex
                name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
                if name_match:
                    metadata["package_name"] = name_match.group(1)

            except Exception as e:
                self.logger.warning(f"Error parsing Cargo.toml: {e}")

        return metadata

    def _extract_readme_info(self, repo_path: str) -> Dict[str, Any]:
        """Extract information from README files."""
        readme_patterns = ["README.md", "README.txt", "README.rst", "readme.md"]

        for pattern in readme_patterns:
            readme_path = os.path.join(repo_path, pattern)
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # Extract title from first heading
                    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
                    if title_match:
                        return {"readme_title": title_match.group(1).strip()}

                except Exception as e:
                    self.logger.warning(f"Error parsing README: {e}")
                break

        return {}

    def _extract_git_info(self, repo_path: str) -> Dict[str, Any]:
        """Extract Git repository information."""
        git_info = {}

        try:
            # Get latest commit info
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H|%s|%ai"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                parts = result.stdout.strip().split("|")
                if len(parts) >= 3:
                    git_info.update(
                        {
                            "latest_commit_hash": parts[0],
                            "latest_commit_message": parts[1],
                            "latest_commit_date": parts[2],
                        }
                    )
        except Exception as e:
            self.logger.warning(f"Error extracting git info: {e}")

        return git_info
