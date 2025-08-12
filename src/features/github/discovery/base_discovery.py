"""
Abstract base class for file discovery operations.

This module defines the interface that all file discovery implementations
must follow to ensure consistent behavior across the GitHub processing system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class IFileDiscovery(ABC):
    """Interface for file discovery operations."""

    @abstractmethod
    def discover_files(
        self,
        repo_path: str,
        file_types: List[str] = None,
        max_files: int = 100,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Discover files of specified types in a repository.

        Args:
            repo_path: Path to the repository directory
            file_types: List of file extensions to discover
            max_files: Maximum number of files to return
            **kwargs: Additional discovery parameters

        Returns:
            List of file information dictionaries

        Raises:
            DiscoveryError: If file discovery fails
        """
        pass
