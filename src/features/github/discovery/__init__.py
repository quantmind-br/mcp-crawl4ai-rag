"""
File discovery system for GitHub repositories.

This package provides file discovery capabilities for different file types
and filtering strategies used throughout the GitHub processing pipeline.
"""

from .base_discovery import IFileDiscovery
from .markdown_discovery import MarkdownDiscovery
from .multi_file_discovery import MultiFileDiscovery
from .file_filter import FileFilter, DirectoryFilter

__all__ = [
    "IFileDiscovery",
    "MarkdownDiscovery",
    "MultiFileDiscovery",
    "FileFilter",
    "DirectoryFilter",
]
