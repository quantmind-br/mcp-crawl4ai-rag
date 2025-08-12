"""
Custom exceptions for GitHub processing operations.

This module defines a hierarchy of exceptions that provide specific error
handling for different types of failures in the GitHub processing pipeline.
"""


class GitHubProcessorError(Exception):
    """Base exception for all GitHub processor errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CloneError(GitHubProcessorError):
    """Raised when repository cloning operations fail."""

    def __init__(self, message: str, repo_url: str = None, details: dict = None):
        super().__init__(message, details)
        self.repo_url = repo_url


class RepositoryTooLargeError(CloneError):
    """Raised when repository exceeds size limits."""

    def __init__(
        self,
        message: str,
        repo_url: str = None,
        size_mb: float = None,
        limit_mb: float = None,
    ):
        super().__init__(message, repo_url)
        self.size_mb = size_mb
        self.limit_mb = limit_mb


class InvalidRepositoryUrlError(CloneError):
    """Raised when repository URL is invalid or malformed."""

    pass


class GitCommandError(CloneError):
    """Raised when Git command execution fails."""

    def __init__(
        self,
        message: str,
        command: str = None,
        returncode: int = None,
        stderr: str = None,
    ):
        super().__init__(message)
        self.command = command
        self.returncode = returncode
        self.stderr = stderr


class DiscoveryError(GitHubProcessorError):
    """Raised when file discovery operations fail."""

    def __init__(self, message: str, repo_path: str = None, details: dict = None):
        super().__init__(message, details)
        self.repo_path = repo_path


class FileAccessError(DiscoveryError):
    """Raised when files cannot be accessed or read."""

    def __init__(self, message: str, file_path: str = None):
        super().__init__(message)
        self.file_path = file_path


class ProcessingError(GitHubProcessorError):
    """Raised when file processing operations fail."""

    def __init__(
        self,
        message: str,
        file_path: str = None,
        processor_name: str = None,
        details: dict = None,
    ):
        super().__init__(message, details)
        self.file_path = file_path
        self.processor_name = processor_name


class UnsupportedFileTypeError(ProcessingError):
    """Raised when attempting to process unsupported file types."""

    def __init__(self, message: str, file_path: str = None, file_extension: str = None):
        super().__init__(message, file_path)
        self.file_extension = file_extension


class ProcessorNotFoundError(ProcessingError):
    """Raised when no processor is available for a file type."""

    def __init__(self, message: str, file_extension: str = None):
        super().__init__(message)
        self.file_extension = file_extension


class SyntaxError(ProcessingError):
    """Raised when file contains syntax errors (for code files)."""

    def __init__(self, message: str, file_path: str = None, line_number: int = None):
        super().__init__(message, file_path)
        self.line_number = line_number


class ValidationError(GitHubProcessorError):
    """Raised when data validation fails."""

    def __init__(
        self,
        message: str,
        field_name: str = None,
        field_value: str = None,
        details: dict = None,
    ):
        super().__init__(message, details)
        self.field_name = field_name
        self.field_value = field_value


class ConfigurationError(GitHubProcessorError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_key: str = None, config_value: str = None):
        super().__init__(message)
        self.config_key = config_key
        self.config_value = config_value


class DependencyError(GitHubProcessorError):
    """Raised when required dependencies are missing or incompatible."""

    def __init__(
        self, message: str, dependency_name: str = None, required_version: str = None
    ):
        super().__init__(message)
        self.dependency_name = dependency_name
        self.required_version = required_version


class TimeoutError(GitHubProcessorError):
    """Raised when operations exceed time limits."""

    def __init__(
        self, message: str, operation: str = None, timeout_seconds: float = None
    ):
        super().__init__(message)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class ResourceError(GitHubProcessorError):
    """Raised when system resources are insufficient."""

    def __init__(
        self,
        message: str,
        resource_type: str = None,
        required: str = None,
        available: str = None,
    ):
        super().__init__(message)
        self.resource_type = resource_type
        self.required = required
        self.available = available


class MetadataExtractionError(GitHubProcessorError):
    """Raised when metadata extraction fails."""

    def __init__(self, message: str, repo_path: str = None, metadata_type: str = None):
        super().__init__(message)
        self.repo_path = repo_path
        self.metadata_type = metadata_type


# Exception mapping for easier categorization
NETWORK_ERRORS = (CloneError, GitCommandError)
FILE_ERRORS = (FileAccessError, UnsupportedFileTypeError, ProcessorNotFoundError)
VALIDATION_ERRORS = (ValidationError, ConfigurationError)
RESOURCE_ERRORS = (RepositoryTooLargeError, TimeoutError, ResourceError)


def categorize_exception(exception: Exception) -> str:
    """
    Categorize an exception for error reporting and handling.

    Args:
        exception: The exception to categorize

    Returns:
        Category name as string
    """
    if isinstance(exception, NETWORK_ERRORS):
        return "network"
    elif isinstance(exception, FILE_ERRORS):
        return "file_system"
    elif isinstance(exception, VALIDATION_ERRORS):
        return "validation"
    elif isinstance(exception, RESOURCE_ERRORS):
        return "resource"
    elif isinstance(exception, GitHubProcessorError):
        return "github_processor"
    else:
        return "unknown"


def format_error_details(exception: GitHubProcessorError) -> dict:
    """
    Format exception details for structured error reporting.

    Args:
        exception: GitHubProcessorError instance

    Returns:
        Dictionary with formatted error details
    """
    details = {
        "error_type": exception.__class__.__name__,
        "message": exception.message,
        "category": categorize_exception(exception),
    }

    # Add specific details based on exception type
    if hasattr(exception, "repo_url") and exception.repo_url:
        details["repo_url"] = exception.repo_url
    if hasattr(exception, "file_path") and exception.file_path:
        details["file_path"] = exception.file_path
    if hasattr(exception, "processor_name") and exception.processor_name:
        details["processor_name"] = exception.processor_name
    if hasattr(exception, "details") and exception.details:
        details["additional_details"] = exception.details

    return details
