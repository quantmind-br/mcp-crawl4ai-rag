"""
Configuration settings and constants for GitHub processing.

This module centralizes all configuration values, file patterns, size limits,
and other constants used throughout the GitHub processing system.
"""

from dataclasses import dataclass
from typing import Dict, Set


# Common directories to exclude from file discovery
EXCLUDED_DIRS: Set[str] = {
    ".git",
    ".github",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    ".next",
    ".nuxt",
    "target",
    "vendor",
    ".cache",
    "coverage",
    ".coverage",
    "htmlcov",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".eggs",
    "*.egg-info",
    ".DS_Store",
    "Thumbs.db",
}

# File patterns to exclude from processing
EXCLUDED_PATTERNS: Set[str] = {
    "CHANGELOG*",
    "HISTORY*",
    "NEWS*",
    "RELEASES*",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "Gemfile.lock",
    "*.min.*",
    "*.bundle.*",
}

# Supported file extensions for multi-language processing
SUPPORTED_EXTENSIONS: Set[str] = {
    # Markdown files
    ".md",
    ".markdown",
    ".mdown",
    ".mkd",
    # Python files
    ".py",
    ".pyi",
    # JavaScript/TypeScript files
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    # Java files
    ".java",
    # Go files
    ".go",
    # Rust files
    ".rs",
    # C/C++ files
    ".c",
    ".h",
    ".cpp",
    ".cxx",
    ".cc",
    ".hpp",
    ".hxx",
    ".hh",
    # C# files
    ".cs",
    # PHP files
    ".php",
    ".php3",
    ".php4",
    ".php5",
    ".phtml",
    # Ruby files
    ".rb",
    ".rbw",
    # Kotlin files
    ".kt",
    ".kts",
    # Configuration files
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}

# File size limits by extension (in bytes)
FILE_SIZE_LIMITS: Dict[str, int] = {
    # Python files
    ".py": 1_000_000,  # 1MB
    ".pyi": 500_000,  # 500KB
    # JavaScript/TypeScript files
    ".js": 1_000_000,
    ".jsx": 1_000_000,
    ".mjs": 1_000_000,
    ".cjs": 1_000_000,
    ".ts": 1_000_000,
    ".tsx": 1_000_000,
    # Java files
    ".java": 1_000_000,
    # Go files
    ".go": 1_000_000,
    # Rust files
    ".rs": 1_000_000,
    # C/C++ files
    ".c": 1_000_000,
    ".h": 500_000,
    ".cpp": 1_000_000,
    ".cxx": 1_000_000,
    ".cc": 1_000_000,
    ".hpp": 500_000,
    ".hxx": 500_000,
    ".hh": 500_000,
    # C# files
    ".cs": 1_000_000,
    # PHP files
    ".php": 1_000_000,
    ".php3": 1_000_000,
    ".php4": 1_000_000,
    ".php5": 1_000_000,
    ".phtml": 1_000_000,
    # Ruby files
    ".rb": 1_000_000,
    ".rbw": 1_000_000,
    # Kotlin files
    ".kt": 1_000_000,
    ".kts": 500_000,
    # Configuration files (smaller limits)
    ".json": 100_000,  # 100KB
    ".yaml": 100_000,
    ".yml": 100_000,
    ".toml": 100_000,
    # Markdown files
    ".md": 1_000_000,
    ".markdown": 1_000_000,
    ".mdown": 1_000_000,
    ".mkd": 1_000_000,
}

# Default processing limits
DEFAULT_LIMITS = {
    "max_files": 100,
    "min_size_bytes": 100,
    "max_size_bytes": 1_000_000,
    "chunk_size": 5000,
    "max_repo_size_mb": 500,
    "git_timeout_seconds": 300,  # 5 minutes
    "processing_timeout_seconds": 60,  # 1 minute per file
}

# Language mappings for file extensions
LANGUAGE_MAPPINGS: Dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".php3": "php",
    ".php4": "php",
    ".php5": "php",
    ".phtml": "php",
    ".rb": "ruby",
    ".rbw": "ruby",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdown": "markdown",
    ".mkd": "markdown",
}

# Processor priority (higher number = higher priority)
PROCESSOR_PRIORITY: Dict[str, int] = {
    "markdown": 10,
    "python": 8,
    "typescript": 7,
    "javascript": 7,
    "java": 6,
    "go": 6,
    "rust": 6,
    "cpp": 5,
    "c": 5,
    "csharp": 5,
    "php": 4,
    "ruby": 4,
    "kotlin": 4,
    "config": 3,
    "text": 1,
}


@dataclass
class GitSettings:
    """Git operation settings."""

    timeout_seconds: int = 300
    max_repo_size_mb: int = 500
    clone_depth: int = 1
    temp_dir_prefix: str = "github_clone_"


@dataclass
class DiscoverySettings:
    """File discovery settings."""

    max_files: int = 100
    min_file_size_bytes: int = 100
    max_file_size_bytes: int = 1_000_000
    excluded_dirs: Set[str] = None
    excluded_patterns: Set[str] = None

    def __post_init__(self):
        if self.excluded_dirs is None:
            self.excluded_dirs = EXCLUDED_DIRS.copy()
        if self.excluded_patterns is None:
            self.excluded_patterns = EXCLUDED_PATTERNS.copy()


@dataclass
class ProcessingSettings:
    """File processing settings."""

    chunk_size: int = 5000
    timeout_seconds: int = 60
    supported_extensions: Set[str] = None
    file_size_limits: Dict[str, int] = None

    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = SUPPORTED_EXTENSIONS.copy()
        if self.file_size_limits is None:
            self.file_size_limits = FILE_SIZE_LIMITS.copy()


@dataclass
class GitHubProcessorConfig:
    """Main configuration for GitHub processor."""

    git: GitSettings = None
    discovery: DiscoverySettings = None
    processing: ProcessingSettings = None
    debug: bool = False
    log_level: str = "INFO"

    def __post_init__(self):
        if self.git is None:
            self.git = GitSettings()
        if self.discovery is None:
            self.discovery = DiscoverySettings()
        if self.processing is None:
            self.processing = ProcessingSettings()


def get_default_config() -> GitHubProcessorConfig:
    """Get default configuration instance."""
    return GitHubProcessorConfig()


def get_file_size_limit(file_extension: str) -> int:
    """
    Get size limit for a specific file extension.

    Args:
        file_extension: File extension (with dot)

    Returns:
        Size limit in bytes
    """
    return FILE_SIZE_LIMITS.get(
        file_extension.lower(), DEFAULT_LIMITS["max_size_bytes"]
    )


def get_language_for_extension(file_extension: str) -> str:
    """
    Get language name for a file extension.

    Args:
        file_extension: File extension (with dot)

    Returns:
        Language name
    """
    return LANGUAGE_MAPPINGS.get(file_extension.lower(), "text")


def is_supported_extension(file_extension: str) -> bool:
    """
    Check if file extension is supported.

    Args:
        file_extension: File extension (with dot)

    Returns:
        True if extension is supported
    """
    return file_extension.lower() in SUPPORTED_EXTENSIONS


def should_exclude_directory(dir_name: str) -> bool:
    """
    Check if directory should be excluded.

    Args:
        dir_name: Directory name

    Returns:
        True if directory should be excluded
    """
    return dir_name in EXCLUDED_DIRS or dir_name.startswith(".")


def should_exclude_file_pattern(filename: str) -> bool:
    """
    Check if file should be excluded based on patterns.

    Args:
        filename: File name

    Returns:
        True if file matches exclusion patterns
    """
    import fnmatch

    for pattern in EXCLUDED_PATTERNS:
        if fnmatch.fnmatch(filename.lower(), pattern.lower()):
            return True
    return False
