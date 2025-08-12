"""
Multi-file type discovery with Tree-sitter integration.

Enhanced file discovery supporting multiple file types via Tree-sitter integration.
Supports 12 programming languages through Tree-sitter parsers plus configuration
and documentation files.
"""

import os
import logging
from typing import Dict, Any, List, Optional

from .markdown_discovery import MarkdownDiscovery
from ..core.exceptions import DiscoveryError, FileAccessError
from ..core.models import FileInfo, DiscoveryResult
from ..config.settings import (
    SUPPORTED_EXTENSIONS,
    get_language_for_extension,
    get_file_size_limit,
    is_supported_extension,
)


class MultiFileDiscovery(MarkdownDiscovery):
    """
    Enhanced file discovery supporting multiple file types via Tree-sitter integration.

    Supports 12 programming languages through Tree-sitter parsers:
    - Python (.py, .pyi)
    - JavaScript/TypeScript (.js, .jsx, .mjs, .cjs, .ts, .tsx)
    - Java (.java)
    - Go (.go)
    - Rust (.rs)
    - C/C++ (.c, .h, .cpp, .cxx, .cc, .hpp, .hxx, .hh)
    - C# (.cs)
    - PHP (.php, .php3, .php4, .php5, .phtml)
    - Ruby (.rb, .rbw)
    - Kotlin (.kt, .kts)

    Plus configuration and documentation files:
    - Markdown (.md, .markdown, .mdown, .mkd)
    - Configuration (.json, .yaml, .yml, .toml)
    """

    def __init__(self, *args, **kwargs):
        """Initialize multi-file discovery with parent configuration."""
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def discover_files(
        self,
        repo_path: str,
        file_types: List[str] = None,
        max_files: int = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Discover files of specified types with metadata.

        Args:
            repo_path: Path to the cloned repository
            file_types: List of file extensions to process (defaults to [".md"])
            max_files: Maximum number of files to process

        Returns:
            List of dictionaries containing file information
        """
        # Default parameters
        file_types = file_types or [".md"]
        max_files = max_files or self.config.max_files

        discovered_files = []
        processed_count = 0

        # Normalize file types to lowercase
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
            raise DiscoveryError(
                f"Failed to discover files of types {file_types}: {e}", repo_path
            )

    def create_discovery_result(
        self,
        repo_path: str,
        discovered_files: List[Dict[str, Any]],
        file_types: List[str] = None,
        **kwargs,
    ) -> DiscoveryResult:
        """
        Create a DiscoveryResult instance for multi-file discovery.

        Args:
            repo_path: Repository path
            discovered_files: List of discovered files
            file_types: File types that were discovered
            **kwargs: Additional result data

        Returns:
            DiscoveryResult instance
        """
        return DiscoveryResult(
            repo_path=repo_path,
            file_types=file_types or [".md"],
            max_files=self.config.max_files,
            discovered_files=[FileInfo(**file_info) for file_info in discovered_files],
            total_files_found=len(discovered_files),
            files_filtered=0,  # Calculated elsewhere
            **kwargs,
        )

    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions."""
        return sorted(list(SUPPORTED_EXTENSIONS))

    def get_file_size_limit(self, file_extension: str) -> int:
        """Get size limit for a specific file extension."""
        return get_file_size_limit(file_extension)

    def get_language_for_extension(self, file_extension: str) -> str:
        """Get language name for a file extension."""
        return get_language_for_extension(file_extension)

    def _is_supported_file(self, filename: str, file_types: List[str]) -> bool:
        """Check if file is of a supported type."""
        file_ext = os.path.splitext(filename)[1].lower()

        # Check if extension is in requested file types
        if file_ext not in file_types:
            return False

        # Check if extension is in supported extensions
        if not is_supported_extension(file_ext):
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
            max_size = get_file_size_limit(file_ext)
            if file_size > max_size:
                self.logger.debug(
                    f"File {file_path} exceeds size limit: {file_size} > {max_size}"
                )
                return None

            # Basic content check for text files
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    # Read first 1000 chars to check if it's text
                    sample = f.read(1000)
                    if "\x00" in sample:  # Binary file indicator
                        return None
            except UnicodeDecodeError:
                self.logger.debug(f"Skipping non-UTF-8 file: {file_path}")
                return None  # Skip non-UTF-8 files
            except (OSError, IOError) as e:
                raise FileAccessError(f"Cannot read file {file_path}: {e}", file_path)

            # Calculate relative path
            relative_path = os.path.relpath(file_path, repo_path)

            # Determine language
            language = get_language_for_extension(file_ext)

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
                "language": language,
            }

        except FileAccessError:
            raise  # Re-raise file access errors
        except Exception as e:
            self.logger.warning(f"Error analyzing file {file_path}: {e}")
            return None
