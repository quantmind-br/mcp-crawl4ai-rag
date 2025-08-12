"""
GitHub repository metadata extraction.

This module provides comprehensive metadata extraction from GitHub repositories,
including package information, README content, and Git commit details.
"""

import os
import re
import subprocess
import logging
from typing import Dict, Any, Tuple
from urllib.parse import urlparse

from ..core.interfaces import IMetadataExtractor
from ..core.models import RepositoryInfo
from ..core.exceptions import MetadataExtractionError


class MetadataExtractor(IMetadataExtractor):
    """Repository metadata extractor with improved error handling and configuration."""

    def __init__(self, timeout_seconds: int = 10):
        """
        Initialize metadata extractor.

        Args:
            timeout_seconds: Timeout for Git operations
        """
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(__name__)

    def extract_repo_metadata(self, repo_url: str, repo_path: str) -> RepositoryInfo:
        """
        Extract metadata from a GitHub repository.

        Args:
            repo_url: Original GitHub repository URL
            repo_path: Path to the cloned repository

        Returns:
            RepositoryInfo instance with extracted metadata
        """
        try:
            # Parse repository info from URL
            owner, repo_name = self._parse_repo_info(repo_url)

            # Initialize base metadata
            metadata = RepositoryInfo(
                repo_url=repo_url,
                owner=owner,
                repo_name=repo_name,
                full_name=f"{owner}/{repo_name}",
                clone_path=repo_path,
            )

            # Extract additional metadata from repository files
            try:
                package_info = self._extract_package_info(repo_path)
                for key, value in package_info.items():
                    setattr(metadata, key, value)
            except Exception as e:
                self.logger.warning(f"Failed to extract package info: {e}")

            try:
                readme_info = self._extract_readme_info(repo_path)
                for key, value in readme_info.items():
                    setattr(metadata, key, value)
            except Exception as e:
                self.logger.warning(f"Failed to extract README info: {e}")

            try:
                git_info = self._extract_git_info(repo_path)
                for key, value in git_info.items():
                    setattr(metadata, key, value)
            except Exception as e:
                self.logger.warning(f"Failed to extract Git info: {e}")

            return metadata

        except Exception as e:
            self.logger.error(f"Error extracting repository metadata: {e}")
            return RepositoryInfo(
                repo_url=repo_url, owner="", repo_name="", full_name="", error=str(e)
            )

    def _parse_repo_info(self, repo_url: str) -> Tuple[str, str]:
        """
        Parse owner and repository name from GitHub URL.

        Args:
            repo_url: GitHub repository URL

        Returns:
            Tuple of (owner, repo_name)

        Raises:
            MetadataExtractionError: If URL format is invalid
        """
        try:
            parsed = urlparse(repo_url)
            path_parts = parsed.path.strip("/").split("/")

            if len(path_parts) < 2:
                raise MetadataExtractionError(
                    f"Invalid GitHub URL format: {repo_url}",
                    metadata_type="url_parsing",
                )

            owner = path_parts[0]
            repo_name = path_parts[1]

            # Remove .git suffix if present
            if repo_name.endswith(".git"):
                repo_name = repo_name[:-4]

            if not owner or not repo_name:
                raise MetadataExtractionError(
                    f"Empty owner or repository name in URL: {repo_url}",
                    metadata_type="url_parsing",
                )

            return owner, repo_name

        except (IndexError, AttributeError) as e:
            raise MetadataExtractionError(
                f"Failed to parse repository URL {repo_url}: {e}",
                metadata_type="url_parsing",
            )

    def _extract_package_info(self, repo_path: str) -> Dict[str, Any]:
        """
        Extract information from package files (package.json, pyproject.toml, etc.).

        Args:
            repo_path: Path to repository

        Returns:
            Dictionary with package metadata
        """
        metadata = {}

        # Check for package.json (Node.js)
        package_json_path = os.path.join(repo_path, "package.json")
        if os.path.exists(package_json_path):
            try:
                import json

                with open(package_json_path, "r", encoding="utf-8") as f:
                    package_data = json.load(f)

                metadata.update(
                    {
                        "language": "javascript",
                        "package_name": package_data.get("name"),
                        "description": package_data.get("description"),
                        "version": package_data.get("version"),
                        "license": package_data.get("license"),
                    }
                )
            except Exception as e:
                self.logger.warning(f"Error parsing package.json: {e}")

        # Check for pyproject.toml (Python)
        pyproject_path = os.path.join(repo_path, "pyproject.toml")
        if os.path.exists(pyproject_path):
            try:
                with open(pyproject_path, "r", encoding="utf-8") as f:
                    content = f.read()

                metadata["language"] = "python"

                # Extract basic info using regex (simple approach)
                name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
                if name_match:
                    metadata["package_name"] = name_match.group(1)

                desc_match = re.search(
                    r'description\s*=\s*["\']([^"\']+)["\']', content
                )
                if desc_match:
                    metadata["description"] = desc_match.group(1)

                version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                if version_match:
                    metadata["version"] = version_match.group(1)

            except Exception as e:
                self.logger.warning(f"Error parsing pyproject.toml: {e}")

        # Check for Cargo.toml (Rust)
        cargo_path = os.path.join(repo_path, "Cargo.toml")
        if os.path.exists(cargo_path):
            try:
                with open(cargo_path, "r", encoding="utf-8") as f:
                    content = f.read()

                metadata["language"] = "rust"

                # Extract basic info using regex
                name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
                if name_match:
                    metadata["package_name"] = name_match.group(1)

                version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                if version_match:
                    metadata["version"] = version_match.group(1)

            except Exception as e:
                self.logger.warning(f"Error parsing Cargo.toml: {e}")

        # Check for go.mod (Go)
        go_mod_path = os.path.join(repo_path, "go.mod")
        if os.path.exists(go_mod_path):
            try:
                with open(go_mod_path, "r", encoding="utf-8") as f:
                    content = f.read()

                metadata["language"] = "go"

                # Extract module name
                module_match = re.search(r"module\s+([^\s]+)", content)
                if module_match:
                    metadata["package_name"] = module_match.group(1)

            except Exception as e:
                self.logger.warning(f"Error parsing go.mod: {e}")

        # Check for pom.xml (Java/Maven)
        pom_path = os.path.join(repo_path, "pom.xml")
        if os.path.exists(pom_path):
            try:
                with open(pom_path, "r", encoding="utf-8") as f:
                    content = f.read()

                metadata["language"] = "java"

                # Extract basic Maven info
                artifact_match = re.search(r"<artifactId>([^<]+)</artifactId>", content)
                if artifact_match:
                    metadata["package_name"] = artifact_match.group(1)

                version_match = re.search(r"<version>([^<]+)</version>", content)
                if version_match:
                    metadata["version"] = version_match.group(1)

            except Exception as e:
                self.logger.warning(f"Error parsing pom.xml: {e}")

        return metadata

    def _extract_readme_info(self, repo_path: str) -> Dict[str, Any]:
        """
        Extract information from README files.

        Args:
            repo_path: Path to repository

        Returns:
            Dictionary with README metadata
        """
        readme_patterns = [
            "README.md",
            "README.MD",
            "readme.md",
            "README.txt",
            "README.TXT",
            "readme.txt",
            "README.rst",
            "README.RST",
            "readme.rst",
            "README",
            "readme",
        ]

        for pattern in readme_patterns:
            readme_path = os.path.join(repo_path, pattern)
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # Extract title from first heading
                    title_patterns = [
                        r"^#\s+(.+)$",  # Markdown H1
                        r"^(.+)\n[=]+\s*$",  # RST/Markdown underlined
                        r"^(.+)\n[-]+\s*$",  # RST/Markdown underlined
                        r"^\*\*(.+)\*\*",  # Bold text
                        r"^(.+)$",  # First line fallback
                    ]

                    for pattern in title_patterns:
                        title_match = re.search(pattern, content, re.MULTILINE)
                        if title_match:
                            title = title_match.group(1).strip()
                            # Skip if title is too long or looks like a URL
                            if len(title) < 100 and not title.startswith(
                                ("http", "www")
                            ):
                                return {"readme_title": title}
                            break

                except Exception as e:
                    self.logger.warning(f"Error parsing README {pattern}: {e}")
                    continue

                break  # Stop after finding first README

        return {}

    def _extract_git_info(self, repo_path: str) -> Dict[str, Any]:
        """
        Extract Git repository information.

        Args:
            repo_path: Path to repository

        Returns:
            Dictionary with Git metadata
        """
        git_info = {}

        try:
            # Get latest commit info with proper encoding handling
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H|%s|%ai"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",  # Replace invalid characters
                timeout=self.timeout_seconds,
            )

            if result.returncode == 0 and result.stdout:
                parts = result.stdout.strip().split("|")
                if len(parts) >= 3:
                    git_info.update(
                        {
                            "latest_commit_hash": parts[0],
                            "latest_commit_message": parts[1],
                            "latest_commit_date": parts[2],
                        }
                    )

        except subprocess.TimeoutExpired:
            self.logger.warning(
                f"Git command timed out after {self.timeout_seconds} seconds"
            )
        except Exception as e:
            self.logger.warning(f"Error extracting git info: {e}")

        return git_info

    def create_metadata_dict(self, metadata: RepositoryInfo) -> Dict[str, Any]:
        """
        Convert RepositoryInfo to dictionary for backward compatibility.

        Args:
            metadata: RepositoryInfo instance

        Returns:
            Dictionary representation
        """
        result = {}
        for field_name, field_value in metadata.__dict__.items():
            if field_value is not None:
                result[field_name] = field_value
        return result
