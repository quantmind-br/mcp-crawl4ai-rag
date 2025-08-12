"""
Markdown file processor.

This module provides processing capabilities for markdown files,
extracting full content for documentation and text analysis.
"""

import os
from typing import List

from .base_processor import BaseFileProcessor
from ..core.models import ProcessedContent


class MarkdownProcessor(BaseFileProcessor):
    """Process markdown files using existing content."""

    def __init__(self):
        """Initialize markdown processor."""
        super().__init__(
            name="markdown",
            supported_extensions=[".md", ".markdown", ".mdown", ".mkd"],
            max_file_size=1_000_000,  # 1MB limit for markdown files
        )

    def _process_file_impl(
        self, file_path: str, relative_path: str, **kwargs
    ) -> List[ProcessedContent]:
        """
        Process markdown file and return content.

        Args:
            file_path: Absolute path to the file
            relative_path: Relative path from repository root
            **kwargs: Additional processing parameters

        Returns:
            List of ProcessedContent instances
        """
        content = self.read_file_content(file_path)

        if not self.validate_content(content, min_length=50):
            return []

        filename = os.path.basename(file_path)

        processed_content = self.create_processed_content(
            content=content.strip(),
            content_type="markdown",
            name=filename,
            signature=None,
            line_number=1,
            language="markdown",
        )

        return [processed_content]
