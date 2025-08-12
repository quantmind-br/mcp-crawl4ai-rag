"""
Markdown file discovery and filtering.

This module provides specialized discovery capabilities for markdown files
with proper filtering, prioritization, and metadata extraction.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple

from .base_discovery import IFileDiscovery
from ..core.exceptions import DiscoveryError, FileAccessError
from ..core.models import FileInfo, DiscoveryResult
from ..config.settings import DiscoverySettings, get_default_config


class MarkdownDiscovery(IFileDiscovery):
    """Discovers and filters markdown files in a repository."""

    def __init__(self, config: DiscoverySettings = None):
        """
        Initialize markdown discovery.

        Args:
            config: Discovery configuration settings
        """
        self.config = config or get_default_config().discovery
        self.logger = logging.getLogger(__name__)

    def discover_files(
        self,
        repo_path: str,
        file_types: List[str] = None,
        max_files: int = None,
        min_size_bytes: int = None,
        max_size_bytes: int = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Discover markdown files in the repository with filtering.

        Args:
            repo_path: Path to the cloned repository
            file_types: File types (defaults to markdown extensions)
            max_files: Maximum number of files to process
            min_size_bytes: Minimum file size in bytes
            max_size_bytes: Maximum file size in bytes

        Returns:
            List of dictionaries containing file information
        """
        # Use config defaults if not provided
        max_files = max_files or self.config.max_files
        min_size_bytes = min_size_bytes or self.config.min_file_size_bytes
        max_size_bytes = max_size_bytes or self.config.max_file_size_bytes

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
            raise DiscoveryError(f"Failed to discover markdown files: {e}", repo_path)

    def discover_markdown_files(
        self,
        repo_path: str,
        max_files: int = 100,
        min_size_bytes: int = 100,
        max_size_bytes: int = 1_000_000,
    ) -> List[Dict[str, Any]]:
        """
        Legacy method for backward compatibility.

        Args:
            repo_path: Path to the cloned repository
            max_files: Maximum number of files to process
            min_size_bytes: Minimum file size in bytes
            max_size_bytes: Maximum file size in bytes

        Returns:
            List of dictionaries containing file information
        """
        return self.discover_files(
            repo_path=repo_path,
            max_files=max_files,
            min_size_bytes=min_size_bytes,
            max_size_bytes=max_size_bytes,
        )

    def create_discovery_result(
        self, repo_path: str, discovered_files: List[Dict[str, Any]], **kwargs
    ) -> DiscoveryResult:
        """
        Create a DiscoveryResult instance.

        Args:
            repo_path: Repository path
            discovered_files: List of discovered files
            **kwargs: Additional result data

        Returns:
            DiscoveryResult instance
        """
        return DiscoveryResult(
            repo_path=repo_path,
            file_types=[".md"],
            max_files=self.config.max_files,
            discovered_files=[FileInfo(**file_info) for file_info in discovered_files],
            total_files_found=len(discovered_files),
            files_filtered=0,  # Calculated elsewhere
            **kwargs,
        )

    def _is_markdown_file(self, filename: str) -> bool:
        """Check if file is a markdown file."""
        return filename.lower().endswith((".md", ".markdown", ".mdown", ".mkd"))

    def _should_exclude_dir(self, dirname: str) -> bool:
        """Check if directory should be excluded."""
        return dirname in self.config.excluded_dirs or dirname.startswith(".")

    def _should_exclude_file(self, filename: str) -> bool:
        """Check if file should be excluded based on patterns."""
        import fnmatch

        for pattern in self.config.excluded_patterns:
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
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except (OSError, IOError) as e:
                raise FileAccessError(f"Cannot read file {file_path}: {e}", file_path)

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
                "file_type": os.path.splitext(file_path)[1].lower(),
            }

        except FileAccessError:
            raise  # Re-raise file access errors
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
