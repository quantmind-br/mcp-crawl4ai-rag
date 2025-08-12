"""
Service layer for GitHub repository processing.

This package provides high-level services and orchestration logic
for managing the complete GitHub repository processing pipeline.
"""

from .github_service import GitHubService

__all__ = [
    "GitHubService",
]
