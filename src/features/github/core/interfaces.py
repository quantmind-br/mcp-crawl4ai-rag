"""
Abstract base classes and interfaces for GitHub processing components.

This module defines the contracts that all GitHub processing components must implement,
enabling loose coupling and easy testing through dependency injection.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class IGitRepository(ABC):
    """Interface for Git repository operations."""

    @abstractmethod
    def clone_repository(self, repo_url: str, max_size_mb: int = 500) -> str:
        """
        Clone a GitHub repository to a temporary directory.

        Args:
            repo_url: GitHub repository URL
            max_size_mb: Maximum repository size in MB

        Returns:
            Path to the cloned repository directory

        Raises:
            CloneError: If cloning fails or repository is too large
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up all temporary directories."""
        pass

    @abstractmethod
    def get_temp_dirs(self) -> List[str]:
        """Get list of active temporary directories."""
        pass


class IMetadataExtractor(ABC):
    """Interface for repository metadata extraction."""

    @abstractmethod
    def extract_repo_metadata(self, repo_url: str, repo_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a GitHub repository.

        Args:
            repo_url: Original GitHub repository URL
            repo_path: Path to the cloned repository

        Returns:
            Dictionary containing repository metadata
        """
        pass


class IFileDiscovery(ABC):
    """Interface for file discovery operations."""

    @abstractmethod
    def discover_files(
        self, repo_path: str, file_types: List[str] = None, max_files: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Discover files in the repository with filtering.

        Args:
            repo_path: Path to the cloned repository
            file_types: List of file extensions to process
            max_files: Maximum number of files to process

        Returns:
            List of dictionaries containing file information
        """
        pass


class IFileProcessor(ABC):
    """Interface for file processing operations."""

    @abstractmethod
    def can_process(self, file_path: str) -> bool:
        """
        Check if this processor can handle the given file.

        Args:
            file_path: Path to the file

        Returns:
            True if processor can handle the file
        """
        pass

    @abstractmethod
    def process_file(self, file_path: str, relative_path: str) -> List[Dict[str, Any]]:
        """
        Process file and return list of extractable content chunks.

        Args:
            file_path: Absolute path to the file
            relative_path: Relative path from repository root

        Returns:
            List of content chunks with metadata
        """
        pass

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of file extensions this processor supports.

        Returns:
            List of supported file extensions
        """
        pass


class IProcessorFactory(ABC):
    """Interface for processor factory."""

    @abstractmethod
    def get_processor(self, file_path: str) -> Optional[IFileProcessor]:
        """
        Get appropriate processor for the given file.

        Args:
            file_path: Path to the file

        Returns:
            Processor instance or None if no processor available
        """
        pass

    @abstractmethod
    def register_processor(self, processor: IFileProcessor) -> None:
        """
        Register a new processor.

        Args:
            processor: Processor instance to register
        """
        pass

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """
        Get all supported file extensions across all processors.

        Returns:
            List of supported file extensions
        """
        pass


class IFileFilter(ABC):
    """Interface for file filtering operations."""

    @abstractmethod
    def should_include(self, file_path: str, file_info: Dict[str, Any]) -> bool:
        """
        Determine if file should be included based on filtering criteria.

        Args:
            file_path: Path to the file
            file_info: File metadata

        Returns:
            True if file should be included
        """
        pass


class IGitHubService(ABC):
    """Interface for the main GitHub processing service."""

    @abstractmethod
    async def clone_repository_temp(
        self, repo_url: str, max_size_mb: int = 500, temp_dir_prefix: str = None
    ) -> Dict[str, Any]:
        """
        Clone repository to temporary directory.

        Args:
            repo_url: GitHub repository URL
            max_size_mb: Maximum repository size in MB
            temp_dir_prefix: Optional prefix for temporary directory name

        Returns:
            Dictionary with clone results and metadata
        """
        pass

    @abstractmethod
    def discover_repository_files(
        self, repo_path: str, file_types: List[str] = None, max_files: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Discover files in repository.

        Args:
            repo_path: Path to cloned repository
            file_types: List of file extensions to discover
            max_files: Maximum number of files to discover

        Returns:
            List of discovered files with metadata
        """
        pass

    @abstractmethod
    async def dispatch_processing_request(
        self,
        repo_url: str,
        destination: str = "both",
        file_types: List[str] = None,
        max_files: int = 50,
        chunk_size: int = 5000,
        max_size_mb: int = 500,
    ) -> Dict[str, Any]:
        """
        Dispatch processing request to appropriate system.

        Args:
            repo_url: GitHub repository URL
            destination: Processing destination ("qdrant", "neo4j", or "both")
            file_types: File extensions to process
            max_files: Maximum number of files to process
            chunk_size: Chunk size for RAG processing
            max_size_mb: Maximum repository size limit

        Returns:
            Dictionary with processing results and statistics
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up all temporary resources."""
        pass

    @abstractmethod
    def get_supported_file_types(self) -> List[str]:
        """Get list of supported file types for processing."""
        pass

    @abstractmethod
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics about the processor's capabilities and limits."""
        pass
