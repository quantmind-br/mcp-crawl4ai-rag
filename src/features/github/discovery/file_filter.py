"""
File and directory filtering utilities.

This module provides filtering capabilities for files and directories
based on various criteria such as size, patterns, and exclusion rules.
"""

import os
import logging
from typing import List, Set, Dict, Any

from ..core.models import FilterCriteria
from ..config.settings import (
    EXCLUDED_DIRS,
    SUPPORTED_EXTENSIONS,
    should_exclude_directory,
    should_exclude_file_pattern,
)


class DirectoryFilter:
    """Filters directories based on exclusion rules."""

    def __init__(self, excluded_dirs: Set[str] = None):
        """
        Initialize directory filter.

        Args:
            excluded_dirs: Set of directory names to exclude
        """
        self.excluded_dirs = excluded_dirs or EXCLUDED_DIRS.copy()
        self.logger = logging.getLogger(__name__)

    def should_exclude(self, dir_name: str, full_path: str = None) -> bool:
        """
        Check if directory should be excluded.

        Args:
            dir_name: Directory name
            full_path: Full directory path (optional)

        Returns:
            True if directory should be excluded
        """
        # Use configuration-based check
        if should_exclude_directory(dir_name):
            return True

        # Additional custom exclusion rules
        if dir_name in self.excluded_dirs:
            return True

        # Hidden directories (starting with .)
        if dir_name.startswith("."):
            return True

        return False

    def filter_directories(self, directories: List[str]) -> List[str]:
        """
        Filter a list of directories.

        Args:
            directories: List of directory names

        Returns:
            Filtered list of directories
        """
        return [d for d in directories if not self.should_exclude(d)]


class FileFilter:
    """Filters files based on various criteria."""

    def __init__(self, criteria: FilterCriteria = None):
        """
        Initialize file filter.

        Args:
            criteria: File filtering criteria
        """
        self.criteria = criteria or FilterCriteria()
        self.logger = logging.getLogger(__name__)

    def should_exclude_by_pattern(self, filename: str) -> bool:
        """
        Check if file should be excluded based on patterns.

        Args:
            filename: File name to check

        Returns:
            True if file matches exclusion patterns
        """
        return should_exclude_file_pattern(filename)

    def should_exclude_by_size(self, file_size: int) -> bool:
        """
        Check if file should be excluded based on size.

        Args:
            file_size: File size in bytes

        Returns:
            True if file should be excluded
        """
        if file_size < self.criteria.min_size_bytes:
            return True
        if file_size > self.criteria.max_size_bytes:
            return True
        return False

    def should_exclude_by_extension(self, file_extension: str) -> bool:
        """
        Check if file should be excluded based on extension.

        Args:
            file_extension: File extension (with dot)

        Returns:
            True if file should be excluded
        """
        # If included_extensions is specified, only allow those
        if self.criteria.included_extensions:
            return file_extension.lower() not in self.criteria.included_extensions

        # Otherwise, use supported extensions
        return file_extension.lower() not in SUPPORTED_EXTENSIONS

    def should_exclude(self, file_path: str, file_info: Dict[str, Any] = None) -> bool:
        """
        Check if file should be excluded based on all criteria.

        Args:
            file_path: Path to the file
            file_info: Optional file information dict

        Returns:
            True if file should be excluded
        """
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()

        # Pattern-based exclusion
        if self.should_exclude_by_pattern(filename):
            return True

        # Extension-based exclusion
        if self.should_exclude_by_extension(file_ext):
            return True

        # Size-based exclusion (if file info provided)
        if file_info and "size_bytes" in file_info:
            if self.should_exclude_by_size(file_info["size_bytes"]):
                return True

        return False

    def filter_files(self, file_paths: List[str]) -> List[str]:
        """
        Filter a list of file paths.

        Args:
            file_paths: List of file paths

        Returns:
            Filtered list of file paths
        """
        filtered_files = []
        for file_path in file_paths:
            if not self.should_exclude(file_path):
                filtered_files.append(file_path)
            else:
                self.logger.debug(f"Filtered out file: {file_path}")

        return filtered_files

    def filter_file_infos(
        self, file_infos: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter a list of file information dictionaries.

        Args:
            file_infos: List of file info dictionaries

        Returns:
            Filtered list of file info dictionaries
        """
        filtered_files = []
        for file_info in file_infos:
            file_path = file_info.get("path", "")
            if not self.should_exclude(file_path, file_info):
                filtered_files.append(file_info)
            else:
                self.logger.debug(f"Filtered out file: {file_path}")

        return filtered_files


class CombinedFilter:
    """Combines directory and file filtering."""

    def __init__(
        self, file_criteria: FilterCriteria = None, excluded_dirs: Set[str] = None
    ):
        """
        Initialize combined filter.

        Args:
            file_criteria: File filtering criteria
            excluded_dirs: Set of directory names to exclude
        """
        self.file_filter = FileFilter(file_criteria)
        self.dir_filter = DirectoryFilter(excluded_dirs)
        self.logger = logging.getLogger(__name__)

    def walk_and_filter(
        self, root_path: str, max_files: int = None
    ) -> List[Dict[str, Any]]:
        """
        Walk directory tree and apply filtering.

        Args:
            root_path: Root directory to walk
            max_files: Maximum number of files to return

        Returns:
            List of filtered file information dictionaries
        """
        filtered_files = []
        files_processed = 0

        try:
            for root, dirs, files in os.walk(root_path):
                # Filter directories in-place
                dirs[:] = self.dir_filter.filter_directories(dirs)

                for filename in files:
                    if max_files and files_processed >= max_files:
                        break

                    file_path = os.path.join(root, filename)

                    # Basic file analysis
                    try:
                        file_stat = os.stat(file_path)
                        file_info = {
                            "path": file_path,
                            "relative_path": os.path.relpath(file_path, root_path),
                            "filename": filename,
                            "size_bytes": file_stat.st_size,
                            "file_type": os.path.splitext(filename)[1].lower(),
                        }

                        # Apply file filtering
                        if not self.file_filter.should_exclude(file_path, file_info):
                            filtered_files.append(file_info)
                            files_processed += 1

                    except (OSError, IOError) as e:
                        self.logger.warning(f"Cannot access file {file_path}: {e}")
                        continue

                if max_files and files_processed >= max_files:
                    break

        except Exception as e:
            self.logger.error(f"Error during directory walk: {e}")
            raise

        self.logger.info(f"Filtered {len(filtered_files)} files from {root_path}")
        return filtered_files
