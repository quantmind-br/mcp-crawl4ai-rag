"""
MDX file processor.

This module provides processing capabilities for MDX files,
extracting content and JSX component metadata for documentation analysis.
"""

import os
import re
from typing import List

from .base_processor import BaseFileProcessor
from ..core.models import ProcessedContent
from ..core.exceptions import ProcessingError


class MDXProcessor(BaseFileProcessor):
    """Process MDX files with JSX component extraction."""

    def __init__(self):
        """Initialize MDX processor with regex patterns."""
        super().__init__(
            name="mdx",
            supported_extensions=[".mdx"],
            max_file_size=1_000_000,  # 1MB limit (same as markdown)
        )

        # Compile regex patterns once for performance
        self.jsx_component_pattern = re.compile(
            r"<([A-Z][a-zA-Z0-9]*)\s*([^>]*?)(?:/>|>(.*?)</\1>)",
            re.DOTALL | re.MULTILINE,
        )

        self.import_pattern = re.compile(
            r'^import\s+.*?from\s+[\'"].*?[\'"]', re.MULTILINE
        )
        self.export_pattern = re.compile(r"^export\s+.*", re.MULTILINE)
        self.jsx_expression_pattern = re.compile(r"\{([^}]+)\}")

    def _process_file_impl(
        self, file_path: str, relative_path: str, **kwargs
    ) -> List[ProcessedContent]:
        """
        Process MDX file and extract content and components.

        Args:
            file_path: Absolute path to the file
            relative_path: Relative path from repository root
            **kwargs: Additional processing parameters

        Returns:
            List of ProcessedContent instances
        """
        try:
            # Use base class file reading with error handling
            content = self.read_file_content(file_path)

            # Validate content with same criteria as markdown
            if not self.validate_content(content, min_length=50):
                return []

            filename = os.path.basename(file_path)
            extracted_items = []

            # Clean content while preserving semantic meaning
            cleaned_content = self._clean_mdx_content(content)

            # Create main content entry using base class helper
            processed_content = self.create_processed_content(
                content=cleaned_content,
                content_type="mdx",
                name=filename,
                line_number=1,
                language="mdx",
            )
            extracted_items.append(processed_content)

            # Extract JSX components for enhanced metadata
            jsx_components = self._extract_jsx_components(content)
            extracted_items.extend(jsx_components)

            return extracted_items

        except Exception as e:
            # Use project exception types with file context
            raise ProcessingError(
                f"Error processing MDX file {file_path}: {e}",
                file_path=file_path,
                processor_name=self.name,
            )

    def _clean_mdx_content(self, content: str) -> str:
        """Clean MDX content for text indexing while preserving semantic meaning."""
        try:
            # Try to extract frontmatter if present
            if content.startswith("---"):
                try:
                    # Find the closing --- delimiter
                    end_marker = content.find("\n---\n", 3)
                    if end_marker != -1:
                        # Skip frontmatter, use only body content
                        body = content[end_marker + 4 :].lstrip("\n")
                    else:
                        # No closing delimiter found, use full content
                        body = content
                except Exception:
                    # If frontmatter parsing fails, use full content
                    body = content
            else:
                body = content

            # Strip JSX components but preserve inner text content
            def replace_jsx_component(match):
                component_name = match.group(1)
                inner_content = match.group(3) or ""
                # Keep meaningful text content
                return f"{component_name} {inner_content}".strip()

            cleaned = self.jsx_component_pattern.sub(replace_jsx_component, body)

            # Remove imports/exports
            cleaned = self.import_pattern.sub("", cleaned)
            cleaned = self.export_pattern.sub("", cleaned)

            # Handle JSX expressions - keep variable names
            cleaned = self.jsx_expression_pattern.sub(r"\1", cleaned)

            # Clean up whitespace
            cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned).strip()

            return cleaned

        except Exception as e:
            # If cleaning fails, log warning and return original content
            self.logger.warning(f"Failed to clean MDX content: {e}")
            return content

    def _extract_jsx_components(self, content: str) -> List[ProcessedContent]:
        """Extract JSX components as separate ProcessedContent entities."""
        components = []
        seen_components = set()

        try:
            for match in self.jsx_component_pattern.finditer(content):
                component_name = match.group(1)

                # Avoid duplicate components
                if component_name not in seen_components:
                    seen_components.add(component_name)

                    line_number = content[: match.start()].count("\n") + 1
                    props = match.group(2) or ""
                    inner_content = match.group(3) or ""

                    # Create signature for the component
                    signature = f"<{component_name}"
                    if props.strip():
                        signature += f" {props.strip()}"
                    signature += ">"

                    # Create content description
                    component_content = f"JSX Component: {component_name}"
                    if inner_content.strip():
                        component_content += f" - {inner_content.strip()[:100]}"

                    component_processed = self.create_processed_content(
                        content=component_content,
                        content_type="jsx_component",
                        name=component_name,
                        signature=signature,
                        line_number=line_number,
                        language="jsx",
                    )
                    components.append(component_processed)

        except Exception as e:
            # If component extraction fails, log warning but continue
            self.logger.warning(f"Failed to extract JSX components: {e}")

        return components
