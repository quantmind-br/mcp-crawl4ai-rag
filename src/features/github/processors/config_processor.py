"""
Configuration file processor.

This module provides processing capabilities for configuration files
including JSON, YAML, TOML, and other structured configuration formats.
"""

import os
from typing import List

from .base_processor import BaseFileProcessor
from ..core.models import ProcessedContent


class ConfigProcessor(BaseFileProcessor):
    """Process configuration files with full content."""

    def __init__(self):
        """Initialize configuration processor."""
        super().__init__(
            name="config",
            supported_extensions=[".json", ".yaml", ".yml", ".toml"],
            max_file_size=100_000,  # 100KB limit for config files
        )

        # Language mapping for different config file extensions
        self.language_mapping = {
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
        }

    def _process_file_impl(
        self, file_path: str, relative_path: str, **kwargs
    ) -> List[ProcessedContent]:
        """
        Process configuration file and return full content.

        Args:
            file_path: Absolute path to the file
            relative_path: Relative path from repository root
            **kwargs: Additional processing parameters

        Returns:
            List of ProcessedContent instances
        """
        content = self.read_file_content(file_path)

        if not self.validate_content(content, min_length=1):
            return []

        # Determine language based on file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        language = self._get_config_language(file_ext)
        filename = os.path.basename(file_path)

        processed_content = self.create_processed_content(
            content=content.strip(),
            content_type="configuration",
            name=filename,
            signature=None,
            line_number=1,
            language=language,
        )

        return [processed_content]

    def _get_config_language(self, ext: str) -> str:
        """
        Map file extension to language name.

        Args:
            ext: File extension with dot

        Returns:
            Language identifier string
        """
        return self.language_mapping.get(ext.lower(), "text")
