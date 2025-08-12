"""
File ID generation utilities for unified repository processing.

This module provides consistent file_id generation for cross-system linking
between Qdrant (vector database) and Neo4j (knowledge graph). The file_id
format follows the pattern: "repo_name:relative_path" for deterministic
identification across both storage systems.
"""

import re
from urllib.parse import urlparse
from typing import Optional
from pathlib import Path


def generate_file_id(repo_url: str, relative_path: str) -> str:
    """
    Generate consistent file_id for cross-system linking between vector and graph databases.

    The file_id format is: "repo_name:relative_path" where:
    - repo_name is extracted from the repository URL and sanitized
    - relative_path is normalized for Windows/Unix compatibility

    Args:
        repo_url: GitHub repository URL (e.g., 'https://github.com/owner/repo.git')
        relative_path: File path relative to repository root

    Returns:
        Consistent file_id string for cross-system identification

    Examples:
        >>> generate_file_id('https://github.com/owner/repo', 'src/main.py')
        'owner__repo:src/main.py'

        >>> generate_file_id('https://github.com/user/project.git', 'docs\\README.md')
        'user__project:docs/README.md'
    """
    # Extract and sanitize repo name from URL
    repo_name = extract_repo_name(repo_url)

    # Normalize path separators for Windows compatibility
    normalized_path = normalize_path_separators(relative_path)

    # Construct file_id with consistent format
    return f"{repo_name}:{normalized_path}"


def extract_repo_name(repo_url: str) -> str:
    """
    Extract repository name from URL with owner prefix.

    Args:
        repo_url: Repository URL in various formats

    Returns:
        Sanitized repository name with owner prefix

    Examples:
        >>> extract_repo_name('https://github.com/owner/repo')
        'owner__repo'

        >>> extract_repo_name('https://github.com/user/project.git')
        'user__project'

        >>> extract_repo_name('git@github.com:org/name.git')
        'org__name'
    """
    # Normalize URL for consistent parsing
    normalized_url = normalize_repo_url(repo_url)

    try:
        # Parse URL components
        parsed = urlparse(normalized_url)

        # Extract path components (remove leading slash)
        path_parts = parsed.path.lstrip("/").split("/")

        # GitHub URLs should have at least owner/repo format
        if len(path_parts) >= 2:
            owner = path_parts[0]
            repo = path_parts[1]

            # Remove .git suffix if present
            repo = repo.replace(".git", "")

            # Sanitize names for file system compatibility
            owner = sanitize_name_component(owner)
            repo = sanitize_name_component(repo)

            # Use hyphen to create readable repo names
            return f"{owner}-{repo}"

    except Exception:
        # Fallback: use last component of URL as repo name
        pass

    # Fallback extraction for malformed URLs
    return extract_fallback_repo_name(repo_url)


def normalize_repo_url(repo_url: str) -> str:
    """
    Normalize repository URL to standard HTTPS format for consistent parsing.

    Args:
        repo_url: Repository URL in various formats

    Returns:
        Normalized HTTPS URL
    """
    repo_url = repo_url.strip()

    # Convert SSH URLs to HTTPS
    if repo_url.startswith("git@github.com:"):
        # git@github.com:owner/repo.git -> https://github.com/owner/repo.git
        path = repo_url[len("git@github.com:") :]
        return f"https://github.com/{path}"

    # Ensure HTTPS prefix
    if not repo_url.startswith(("http://", "https://")):
        if repo_url.startswith("github.com/"):
            return f"https://{repo_url}"
        elif "/" in repo_url and "." not in repo_url.split("/")[0]:
            # Assume it's a GitHub owner/repo format
            return f"https://github.com/{repo_url}"

    return repo_url


def extract_fallback_repo_name(repo_url: str) -> str:
    """
    Fallback repo name extraction for edge cases and malformed URLs.

    Args:
        repo_url: Repository URL that couldn't be parsed normally

    Returns:
        Best-effort repository name extraction
    """
    # Remove common prefixes and suffixes
    cleaned = repo_url.rstrip("/")

    # Remove protocol
    for prefix in ["https://", "http://", "git@", "ssh://"]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]

    # Extract last meaningful component
    components = [c for c in cleaned.split("/") if c and c != "git"]
    if components:
        repo_component = components[-1]
        # Remove .git suffix
        repo_component = repo_component.replace(".git", "")
        return sanitize_name_component(repo_component)

    # Ultimate fallback: use hash of original URL
    import hashlib

    url_hash = hashlib.sha256(repo_url.encode()).hexdigest()[:8]
    return f"repo_{url_hash}"


def sanitize_name_component(name: str) -> str:
    """
    Sanitize name component for file system and database compatibility.

    Args:
        name: Raw name component to sanitize

    Returns:
        Sanitized name safe for use in file systems and databases
    """
    if not name:
        return "unknown"

    # Replace problematic characters with underscores
    sanitized = re.sub(r"[^\w\-.]", "_", name)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    # Ensure non-empty result
    if not sanitized:
        return "unknown"

    return sanitized


def normalize_path_separators(file_path: str) -> str:
    """
    Normalize path separators for cross-platform compatibility.

    Converts Windows backslashes to forward slashes for consistent
    file_id format across different operating systems.

    Args:
        file_path: File path with potentially mixed separators

    Returns:
        Normalized path with forward slashes

    Examples:
        >>> normalize_path_separators('src\\main.py')
        'src/main.py'

        >>> normalize_path_separators('docs/README.md')
        'docs/README.md'
    """
    # Convert to Path object for normalization
    normalized = str(Path(file_path).as_posix())

    # Ensure we don't have double slashes
    normalized = re.sub(r"/+", "/", normalized)

    # Remove leading slash if present (relative paths only)
    return normalized.lstrip("/")


def validate_file_id(file_id: str) -> bool:
    """
    Validate that a file_id follows the expected format.

    Args:
        file_id: File ID to validate

    Returns:
        True if file_id is properly formatted, False otherwise

    Examples:
        >>> validate_file_id('owner__repo:src/main.py')
        True

        >>> validate_file_id('invalid-format')
        False
    """
    if not file_id or ":" not in file_id:
        return False

    try:
        repo_part, path_part = file_id.split(":", 1)

        # Validate repo part (should contain -)
        if "-" not in repo_part or not repo_part:
            return False

        # Validate path part (should not be empty and use forward slashes)
        if not path_part or "\\" in path_part:
            return False

        return True

    except ValueError:
        return False


def parse_file_id(file_id: str) -> Optional[tuple[str, str]]:
    """
    Parse file_id back into repo_name and relative_path components.

    Args:
        file_id: File ID in format "repo_name:relative_path"

    Returns:
        Tuple of (repo_name, relative_path) or None if invalid

    Examples:
        >>> parse_file_id('owner__repo:src/main.py')
        ('owner__repo', 'src/main.py')
    """
    if not validate_file_id(file_id):
        return None

    repo_name, relative_path = file_id.split(":", 1)
    return repo_name, relative_path


def get_repo_name_from_file_id(file_id: str) -> Optional[str]:
    """
    Extract just the repository name from a file_id.

    Args:
        file_id: File ID in format "repo_name:relative_path"

    Returns:
        Repository name or None if invalid

    Examples:
        >>> get_repo_name_from_file_id('owner__repo:src/main.py')
        'owner__repo'
    """
    parsed = parse_file_id(file_id)
    return parsed[0] if parsed else None


def get_relative_path_from_file_id(file_id: str) -> Optional[str]:
    """
    Extract just the relative path from a file_id.

    Args:
        file_id: File ID in format "repo_name:relative_path"

    Returns:
        Relative path or None if invalid

    Examples:
        >>> get_relative_path_from_file_id('owner__repo:src/main.py')
        'src/main.py'
    """
    parsed = parse_file_id(file_id)
    return parsed[1] if parsed else None
