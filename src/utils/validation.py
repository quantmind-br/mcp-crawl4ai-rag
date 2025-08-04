"""
URL validation utilities for the Crawl4AI MCP server.
"""

import re
from urllib.parse import urlparse
from typing import Tuple


def validate_github_url(url: str) -> Tuple[bool, str]:
    """
    Validate a GitHub repository URL and return validation result.

    Args:
        url: GitHub repository URL to validate

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if URL is valid, False otherwise
        - error_message: Error description if invalid, empty string if valid
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"

    url = url.strip()
    if not url:
        return False, "URL cannot be empty"

    try:
        # Parse the URL
        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme not in ["http", "https"]:
            return False, "URL must use http or https scheme"

        # Check if it's GitHub
        if parsed.netloc not in ["github.com", "www.github.com"]:
            return False, "URL must be from github.com"

        # Check path format
        path = parsed.path.strip("/")
        if not path:
            return False, "URL must include repository path"

        # Split path into components
        path_parts = path.split("/")
        if len(path_parts) < 2:
            return False, "URL must include both owner and repository name"

        owner = path_parts[0]
        repo = path_parts[1]

        # Remove .git suffix if present for validation
        if repo.endswith(".git"):
            repo = repo[:-4]

        # Validate owner and repository name
        if not owner or not repo:
            return False, "Owner and repository name cannot be empty"

        # Check for valid GitHub username/organization format
        # GitHub usernames can contain alphanumeric characters and hyphens
        # but cannot start or end with hyphens
        github_name_pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$"

        if not re.match(github_name_pattern, owner):
            return False, "Invalid GitHub owner/organization name format"

        if not re.match(github_name_pattern, repo):
            return False, "Invalid GitHub repository name format"

        # Check for common invalid patterns
        if ".." in path or "//" in path:
            return False, "URL contains invalid path patterns"

        # Additional path validation - should not have more than 2 main components
        # (owner/repo), but can have additional paths for specific files/branches
        if len(path_parts) > 2:
            # Allow common GitHub URL patterns like /owner/repo/tree/branch
            # or /owner/repo/blob/branch/file.md
            valid_subpaths = ["tree", "blob", "releases", "issues", "pull", "wiki"]
            if path_parts[2] not in valid_subpaths:
                return False, "URL contains unsupported path format"

        return True, ""

    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"


def normalize_github_url(url: str) -> str:
    """
    Normalize a GitHub URL to a standard format for cloning.

    Args:
        url: GitHub repository URL

    Returns:
        Normalized GitHub URL suitable for git clone

    Raises:
        ValueError: If URL is invalid
    """
    is_valid, error = validate_github_url(url)
    if not is_valid:
        raise ValueError(f"Invalid GitHub URL: {error}")

    # Parse the URL
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    path_parts = path.split("/")

    # Extract owner and repo
    owner = path_parts[0]
    repo = path_parts[1]

    # Remove .git suffix for consistency
    if repo.endswith(".git"):
        repo = repo[:-4]

    # Return normalized HTTPS URL with .git suffix for cloning
    return f"https://github.com/{owner}/{repo}.git"
