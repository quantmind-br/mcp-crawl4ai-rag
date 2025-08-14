"""Documentation file processor for GitHub repositories.

This module provides processing capabilities for additional documentation formats
beyond Markdown, including ReStructuredText, plain text files, and AsciiDoc.
"""

import os
from typing import List

from ..core.models import ProcessedContent
from .base_processor import BaseFileProcessor


class DocumentationProcessor(BaseFileProcessor):
    """Process documentation files (.rst, .txt, .adoc) using existing content."""

    def __init__(self):
        """Initialize documentation processor."""
        super().__init__(
            name="documentation",
            supported_extensions=[".rst", ".txt", ".adoc"],
            max_file_size=1_000_000,  # 1MB limit for documentation files
        )

    def _process_file_impl(
        self, file_path: str, relative_path: str, **kwargs
    ) -> List[ProcessedContent]:
        """
        Process documentation file and return content.

        Args:
            file_path: Absolute path to the file
            relative_path: Relative path from repository root
            **kwargs: Additional processing parameters

        Returns:
            List of ProcessedContent instances
        """
        content = self.read_file_content(file_path)

        if not self.validate_content(content, min_length=10):
            return []

        filename = os.path.basename(file_path)

        # Determine the documentation type based on extension
        extension = os.path.splitext(file_path)[1].lower()
        doc_type_map = {".rst": "restructuredtext", ".txt": "text", ".adoc": "asciidoc"}
        doc_type = doc_type_map.get(extension, "documentation")

        processed_content = self.create_processed_content(
            content=content.strip(),
            content_type="documentation",
            name=filename,
            signature=f"{doc_type} documentation",
            line_number=1,
            language=doc_type,
        )

        return [processed_content]
