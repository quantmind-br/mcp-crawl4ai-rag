"""
GitHub processing module.

This module provides modular GitHub repository processing capabilities
including cloning, file discovery, content processing, and metadata extraction.
"""

# Main service layer (recommended for new code)
from .services.github_service import GitHubService

# Core components
from .repository.git_operations import GitRepository
from .repository.metadata_extractor import MetadataExtractor
from .discovery import MarkdownDiscovery, MultiFileDiscovery
from .processors import ProcessorFactory

# Configuration and models
from .config.settings import GitHubProcessorConfig, get_default_config
from .core.models import (
    RepositoryInfo,
    FileInfo,
    ProcessingResult,
    ProcessingConfig,
)
from .core.exceptions import GitHubProcessorError

__all__ = [
    # Main service (recommended)
    "GitHubService",
    # Core components
    "GitRepository",
    "MetadataExtractor",
    "MarkdownDiscovery",
    "MultiFileDiscovery",
    "ProcessorFactory",
    # Configuration and models
    "GitHubProcessorConfig",
    "get_default_config",
    "RepositoryInfo",
    "FileInfo",
    "ProcessingResult",
    "ProcessingConfig",
    "GitHubProcessorError",
]
