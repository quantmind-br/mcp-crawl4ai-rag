"""
Git repository operations and management.

This module provides Git repository cloning, validation, and cleanup operations
with comprehensive error handling and configuration support.
"""

import os
import shutil
import stat
import subprocess
import tempfile
import logging
from typing import List
from urllib.parse import urlparse

from ..core.interfaces import IGitRepository
from ..core.exceptions import (
    CloneError,
    InvalidRepositoryUrlError,
    RepositoryTooLargeError,
    GitCommandError,
    TimeoutError as ProcessorTimeoutError,
)
from ..core.models import CloneResult
from ..config.settings import GitSettings, get_default_config


class GitRepository(IGitRepository):
    """Git repository operations with dependency injection and improved error handling."""

    def __init__(self, config: GitSettings = None):
        """
        Initialize Git repository manager.

        Args:
            config: Git operation settings
        """
        self.config = config or get_default_config().git
        self.temp_dirs: List[str] = []
        self.logger = logging.getLogger(__name__)

    def clone_repository(self, repo_url: str, max_size_mb: int = None) -> str:
        """
        Clone a GitHub repository to a temporary directory with size checks.

        Args:
            repo_url: GitHub repository URL
            max_size_mb: Maximum repository size in MB (overrides config)

        Returns:
            Path to the cloned repository directory

        Raises:
            InvalidRepositoryUrlError: If repository URL is invalid
            RepositoryTooLargeError: If repository exceeds size limits
            GitCommandError: If git clone command fails
            ProcessorTimeoutError: If operation times out
        """
        # Use provided limit or config default
        max_size = max_size_mb or self.config.max_repo_size_mb

        # Validate GitHub URL
        if not self._is_valid_github_url(repo_url):
            raise InvalidRepositoryUrlError(
                f"Invalid GitHub repository URL: {repo_url}", repo_url
            )

        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=self.config.temp_dir_prefix)
        self.temp_dirs.append(temp_dir)

        try:
            # Normalize URL for cloning
            clone_url = self._normalize_clone_url(repo_url)

            # Clone with depth=1 for efficiency
            self.logger.info(f"Cloning repository: {clone_url}")
            result = subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    str(self.config.clone_depth),
                    clone_url,
                    temp_dir,
                ],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )

            if result.returncode != 0:
                raise GitCommandError(
                    f"Git clone failed: {result.stderr}",
                    command=f"git clone --depth {self.config.clone_depth} {clone_url}",
                    returncode=result.returncode,
                    stderr=result.stderr,
                )

            # Check repository size
            repo_size_mb = self._get_directory_size_mb(temp_dir)
            if repo_size_mb > max_size:
                raise RepositoryTooLargeError(
                    f"Repository too large: {repo_size_mb:.1f}MB exceeds limit of {max_size}MB",
                    repo_url=repo_url,
                    size_mb=repo_size_mb,
                    limit_mb=max_size,
                )

            self.logger.info(f"Successfully cloned repository ({repo_size_mb:.1f}MB)")
            return temp_dir

        except subprocess.TimeoutExpired:
            # Clean up on timeout
            self._cleanup_directory(temp_dir)
            if temp_dir in self.temp_dirs:
                self.temp_dirs.remove(temp_dir)
            raise ProcessorTimeoutError(
                f"Git clone operation timed out after {self.config.timeout_seconds} seconds",
                operation="git_clone",
                timeout_seconds=self.config.timeout_seconds,
            )
        except Exception as e:
            # Clean up on any failure
            self._cleanup_directory(temp_dir)
            if temp_dir in self.temp_dirs:
                self.temp_dirs.remove(temp_dir)
            # Re-raise our custom exceptions
            if isinstance(
                e,
                (
                    InvalidRepositoryUrlError,
                    RepositoryTooLargeError,
                    GitCommandError,
                    ProcessorTimeoutError,
                ),
            ):
                raise e
            # Wrap unexpected exceptions
            raise CloneError(
                f"Unexpected error during repository cloning: {str(e)}", repo_url
            )

    def cleanup(self) -> None:
        """Clean up all temporary directories."""
        cleanup_errors = []
        for temp_dir in self.temp_dirs:
            try:
                self._cleanup_directory(temp_dir)
            except Exception as e:
                cleanup_errors.append(f"Failed to cleanup {temp_dir}: {e}")

        self.temp_dirs.clear()

        if cleanup_errors:
            self.logger.warning(
                f"Cleanup completed with errors: {'; '.join(cleanup_errors)}"
            )
        else:
            self.logger.debug("All temporary directories cleaned up successfully")

    def get_temp_dirs(self) -> List[str]:
        """Get list of active temporary directories."""
        return self.temp_dirs.copy()

    def _is_valid_github_url(self, url: str) -> bool:
        """
        Check if URL is a valid GitHub repository URL.

        Args:
            url: URL to validate

        Returns:
            True if URL is valid GitHub repository URL
        """
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

            # Check for valid characters (basic validation)
            valid_chars = set(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
            )
            if not all(c in valid_chars for c in owner + repo.replace(".git", "")):
                return False

            return True
        except Exception:
            return False

    def _normalize_clone_url(self, url: str) -> str:
        """
        Normalize GitHub URL for git clone.

        Args:
            url: GitHub repository URL

        Returns:
            Normalized URL with .git suffix
        """
        url = url.rstrip("/")
        if not url.endswith(".git"):
            url += ".git"
        return url

    def _get_directory_size_mb(self, directory: str) -> float:
        """
        Calculate directory size in MB.

        Args:
            directory: Directory path

        Returns:
            Size in MB
        """
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, IOError):
                        # Skip files that can't be accessed
                        continue
        except Exception as e:
            self.logger.warning(f"Error calculating directory size: {e}")
            return 0.0

        return total_size / (1024 * 1024)

    def _cleanup_directory(self, directory: str) -> None:
        """
        Safely remove a directory and its contents.

        Args:
            directory: Directory path to remove
        """
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
            # Don't re-raise cleanup errors to avoid masking original errors

    def create_clone_result(
        self, success: bool, repo_url: str, **kwargs
    ) -> CloneResult:
        """
        Create a CloneResult instance with consistent structure.

        Args:
            success: Whether clone was successful
            repo_url: Repository URL
            **kwargs: Additional result data

        Returns:
            CloneResult instance
        """
        return CloneResult(success=success, repo_url=repo_url, **kwargs)
