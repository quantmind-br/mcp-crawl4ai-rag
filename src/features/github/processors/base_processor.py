"""
Base processor interface and implementation.

This module defines the abstract interface for file processors and provides
a base implementation with common functionality.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List

from ..core.exceptions import ProcessingError, UnsupportedFileTypeError
from ..core.models import ProcessedContent, ProcessingResult
from ..config.settings import get_file_size_limit, get_language_for_extension


class IFileProcessor(ABC):
    """Interface for file processors."""

    @abstractmethod
    def process_file(
        self, file_path: str, relative_path: str, **kwargs
    ) -> List[ProcessedContent]:
        """
        Process a file and extract content chunks.

        Args:
            file_path: Absolute path to the file
            relative_path: Relative path from repository root
            **kwargs: Additional processing parameters

        Returns:
            List of ProcessedContent instances

        Raises:
            ProcessingError: If file processing fails
            UnsupportedFileTypeError: If file type is not supported
        """
        pass

    @abstractmethod
    def can_process(self, file_path: str) -> bool:
        """
        Check if processor can handle this file.

        Args:
            file_path: Path to the file

        Returns:
            True if processor can handle the file
        """
        pass

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.

        Returns:
            List of file extensions (with dots)
        """
        pass


class BaseFileProcessor(IFileProcessor):
    """Base implementation for file processors with common functionality."""

    def __init__(
        self, name: str, supported_extensions: List[str], max_file_size: int = None
    ):
        """
        Initialize base processor.

        Args:
            name: Processor name
            supported_extensions: List of supported file extensions
            max_file_size: Maximum file size in bytes (optional)
        """
        self.name = name
        self.supported_extensions = [ext.lower() for ext in supported_extensions]
        self.max_file_size = max_file_size
        self.logger = logging.getLogger(__name__)

    def can_process(self, file_path: str) -> bool:
        """Check if processor can handle this file."""
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in self.supported_extensions

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return self.supported_extensions.copy()

    def process_file(
        self, file_path: str, relative_path: str, **kwargs
    ) -> List[ProcessedContent]:
        """
        Process file with common validation and error handling.

        Args:
            file_path: Absolute path to the file
            relative_path: Relative path from repository root
            **kwargs: Additional processing parameters

        Returns:
            List of ProcessedContent instances
        """
        # Validate file can be processed
        if not self.can_process(file_path):
            file_ext = os.path.splitext(file_path)[1]
            raise UnsupportedFileTypeError(
                f"Processor {self.name} cannot process files with extension {file_ext}",
                file_path=file_path,
                file_extension=file_ext,
            )

        # Check file size
        try:
            file_size = os.path.getsize(file_path)
            max_size = self.max_file_size or get_file_size_limit(
                os.path.splitext(file_path)[1]
            )

            if file_size > max_size:
                self.logger.warning(
                    f"File {file_path} exceeds size limit: {file_size} > {max_size}"
                )
                return []
        except (OSError, IOError) as e:
            raise ProcessingError(
                f"Cannot access file {file_path}: {e}", file_path, self.name
            )

        # Delegate to specific processor implementation
        try:
            return self._process_file_impl(file_path, relative_path, **kwargs)
        except Exception as e:
            if isinstance(e, (ProcessingError, UnsupportedFileTypeError)):
                raise e
            # Wrap unexpected errors
            raise ProcessingError(
                f"Unexpected error processing {file_path}: {e}",
                file_path=file_path,
                processor_name=self.name,
            )

    @abstractmethod
    def _process_file_impl(
        self, file_path: str, relative_path: str, **kwargs
    ) -> List[ProcessedContent]:
        """
        Actual file processing implementation.

        Args:
            file_path: Absolute path to the file
            relative_path: Relative path from repository root
            **kwargs: Additional processing parameters

        Returns:
            List of ProcessedContent instances
        """
        pass

    def create_processed_content(
        self,
        content: str,
        content_type: str,
        name: str,
        signature: str = None,
        line_number: int = 1,
        language: str = None,
        **kwargs,
    ) -> ProcessedContent:
        """
        Create a ProcessedContent instance.

        Args:
            content: The actual content text
            content_type: Type of content (e.g., 'function', 'class', 'module')
            name: Name or identifier for the content
            signature: Function/method signature (optional)
            line_number: Line number where content starts
            language: Programming language
            **kwargs: Additional content metadata

        Returns:
            ProcessedContent instance
        """
        # Auto-detect language if not provided
        if language is None:
            file_ext = os.path.splitext(name)[1] if "." in name else ""
            language = get_language_for_extension(file_ext) if file_ext else "text"

        return ProcessedContent(
            content=content,
            content_type=content_type,
            name=name,
            signature=signature,
            line_number=line_number,
            language=language,
        )

    def create_processing_result(
        self,
        file_path: str,
        relative_path: str,
        processed_chunks: List[ProcessedContent],
        success: bool = True,
        error: str = None,
        processing_time: float = None,
        **kwargs,
    ) -> ProcessingResult:
        """
        Create a ProcessingResult instance.

        Args:
            file_path: Absolute path to the processed file
            relative_path: Relative path from repository root
            processed_chunks: List of processed content chunks
            success: Whether processing was successful
            error: Error message if processing failed
            processing_time: Time taken to process in seconds
            **kwargs: Additional result metadata

        Returns:
            ProcessingResult instance
        """
        return ProcessingResult(
            file_path=file_path,
            relative_path=relative_path,
            success=success,
            processed_chunks=processed_chunks,
            error=error,
            processing_time=processing_time,
        )

    def read_file_content(self, file_path: str, encoding: str = "utf-8") -> str:
        """
        Safely read file content with error handling.

        Args:
            file_path: Path to the file
            encoding: File encoding

        Returns:
            File content as string

        Raises:
            ProcessingError: If file cannot be read
        """
        try:
            with open(file_path, "r", encoding=encoding, errors="ignore") as f:
                return f.read()
        except (OSError, IOError, UnicodeError) as e:
            raise ProcessingError(
                f"Cannot read file {file_path}: {e}",
                file_path=file_path,
                processor_name=self.name,
            )

    def validate_content(self, content: str, min_length: int = 10) -> bool:
        """
        Validate file content.

        Args:
            content: File content to validate
            min_length: Minimum content length

        Returns:
            True if content is valid
        """
        if not content or len(content.strip()) < min_length:
            return False

        # Check for binary content
        if "\x00" in content:
            return False

        return True
