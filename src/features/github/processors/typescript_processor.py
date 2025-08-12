"""
TypeScript file processor.

This module provides processing capabilities for TypeScript and JavaScript files,
extracting JSDoc comments and documentation from functions, classes, and interfaces.
"""

import re
from typing import List, Dict, Any, Optional

from .base_processor import BaseFileProcessor
from ..core.models import ProcessedContent
from ..core.exceptions import ProcessingError


class TypeScriptProcessor(BaseFileProcessor):
    """Process TypeScript files for JSDoc comments."""

    def __init__(self):
        """Initialize TypeScript processor."""
        super().__init__(
            name="typescript",
            supported_extensions=[".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"],
            max_file_size=1_000_000,  # 1MB limit for TypeScript files
        )

        # JSDoc comment pattern
        self.jsdoc_pattern = re.compile(
            r"/\*\*\s*\n((?:\s*\*[^\n]*\n)*)\s*\*/", re.MULTILINE | re.DOTALL
        )

        # Declaration patterns
        self.declaration_patterns = {
            "function": re.compile(
                r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)",
                re.MULTILINE,
            ),
            "class": re.compile(r"(?:export\s+)?class\s+(\w+)", re.MULTILINE),
            "interface": re.compile(r"(?:export\s+)?interface\s+(\w+)", re.MULTILINE),
        }

    def _process_file_impl(
        self, file_path: str, relative_path: str, **kwargs
    ) -> List[ProcessedContent]:
        """
        Extract JSDoc comments from TypeScript file.

        Args:
            file_path: Absolute path to the file
            relative_path: Relative path from repository root
            **kwargs: Additional processing parameters

        Returns:
            List of ProcessedContent instances
        """
        try:
            content = self.read_file_content(file_path)

            # Check if file is minified (skip if first line is too long)
            first_line = content.split("\n")[0] if "\n" in content else content
            if len(first_line) > 1000:
                self.logger.debug(f"Skipping minified file: {file_path}")
                return []  # Skip minified files

            extracted_items = []

            # Find all JSDoc comments and their associated declarations
            for match in self.jsdoc_pattern.finditer(content):
                comment_text = match.group(1)
                start_pos = match.start()

                # Clean comment text
                cleaned_comment = self._clean_jsdoc_comment(comment_text)
                if not cleaned_comment:
                    continue

                # Calculate line number
                line_number = content[:start_pos].count("\n") + 1

                # Find associated declaration
                after_comment = content[match.end() :]
                declaration = self._find_next_declaration(after_comment)

                if declaration and cleaned_comment:
                    processed_content = self.create_processed_content(
                        content=cleaned_comment,
                        content_type=declaration["type"],
                        name=declaration["name"],
                        signature=declaration.get("signature", ""),
                        line_number=line_number,
                        language="typescript",
                    )
                    extracted_items.append(processed_content)

            return extracted_items

        except Exception as e:
            raise ProcessingError(
                f"Error processing TypeScript file {file_path}: {e}",
                file_path=file_path,
                processor_name=self.name,
            )

    def _clean_jsdoc_comment(self, comment_text: str) -> str:
        """
        Clean JSDoc comment text by removing formatting characters.

        Args:
            comment_text: Raw JSDoc comment text

        Returns:
            Cleaned comment text
        """
        lines = comment_text.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if line.startswith("*"):
                line = line[1:].strip()
            if line:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _find_next_declaration(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Find the next function/class/interface declaration after a JSDoc comment.

        Args:
            content: Content to search in

        Returns:
            Dictionary with declaration info or None if not found
        """
        # Remove leading whitespace and newlines
        content = content.lstrip()

        # Try each declaration pattern
        for decl_type, pattern in self.declaration_patterns.items():
            match = pattern.search(content)
            if match and match.start() < 200:  # Must be close to comment
                return {
                    "type": decl_type,
                    "name": match.group(1),
                    "signature": match.group(0),
                }

        return None
