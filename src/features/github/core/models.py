"""
Data models and DTOs for GitHub processing operations.

This module contains data classes and Pydantic models used throughout
the GitHub processing system for type safety and data validation.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class RepositoryInfo:
    """Repository metadata information."""

    repo_url: str
    owner: str
    repo_name: str
    full_name: str
    source_type: str = "github_repository"
    clone_path: Optional[str] = None
    language: Optional[str] = None
    package_name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    license: Optional[str] = None
    readme_title: Optional[str] = None
    latest_commit_hash: Optional[str] = None
    latest_commit_message: Optional[str] = None
    latest_commit_date: Optional[str] = None
    error: Optional[str] = None


@dataclass
class FileInfo:
    """File metadata information."""

    path: str
    relative_path: str
    filename: str
    size_bytes: int
    file_type: Optional[str] = None
    content: Optional[str] = None
    word_count: Optional[int] = None
    is_readme: bool = False


@dataclass
class ProcessedContent:
    """Processed content chunk with metadata."""

    content: str
    content_type: str
    name: str
    signature: Optional[str] = None
    line_number: int = 1
    language: str = "text"


@dataclass
class ProcessingResult:
    """Result of file processing operation."""

    file_path: str
    relative_path: str
    success: bool
    processed_chunks: List[ProcessedContent]
    error: Optional[str] = None
    processing_time: Optional[float] = None


@dataclass
class CloneResult:
    """Result of repository cloning operation."""

    success: bool
    repo_url: str
    temp_directory: Optional[str] = None
    metadata: Optional[RepositoryInfo] = None
    size_mb: Optional[float] = None
    error: Optional[str] = None
    clone_time: Optional[float] = None


@dataclass
class DiscoveryResult:
    """Result of file discovery operation."""

    repo_path: str
    file_types: List[str]
    max_files: int
    discovered_files: List[FileInfo]
    total_files_found: int
    files_filtered: int
    discovery_time: Optional[float] = None


@dataclass
class ProcessorCapabilities:
    """Processor capabilities and configuration."""

    name: str
    supported_extensions: List[str]
    max_file_size: int
    description: str
    version: str = "1.0.0"


@dataclass
class FilterCriteria:
    """File filtering criteria."""

    min_size_bytes: int = 100
    max_size_bytes: int = 1_000_000
    excluded_dirs: Optional[List[str]] = None
    excluded_patterns: Optional[List[str]] = None
    included_extensions: Optional[List[str]] = None


@dataclass
class ProcessingConfig:
    """Configuration for processing operations."""

    max_files: int = 100
    chunk_size: int = 5000
    max_size_mb: int = 500
    file_types: List[str] = None
    filter_criteria: Optional[FilterCriteria] = None

    def __post_init__(self):
        if self.file_types is None:
            self.file_types = [".md"]


@dataclass
class ServiceStatistics:
    """Statistics about service capabilities and current state."""

    supported_file_types: int
    file_extensions: List[str]
    size_limits: Dict[str, int]
    processing_modes: List[str]
    destinations: List[str]
    temp_directories_active: int
    processors_registered: int
    total_files_processed: int = 0
    total_repositories_processed: int = 0
    average_processing_time: Optional[float] = None


@dataclass
class ValidationResult:
    """Result of data validation operation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    data: Optional[Dict[str, Any]] = None


# Type aliases for better readability
FileMetadata = Dict[str, Any]
ProcessingMetrics = Dict[str, float]
ConfigDict = Dict[str, Any]
