"""
Backward compatibility tests for GitHub repository processing.

Tests that the default behavior (markdown-only) still works correctly
after refactoring to use the unified indexing architecture.
"""
# ruff: noqa: E402

import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from features.github.discovery import MultiFileDiscovery, MarkdownDiscovery
from features.github.config.settings import SUPPORTED_EXTENSIONS, FILE_SIZE_LIMITS


class TestBackwardCompatibility:
    """Test backward compatibility of multi-file changes."""

    def test_multifile_discovery_defaults_to_markdown(self):
        """Test that MultiFileDiscovery works with default markdown behavior."""
        # Create temporary test repository
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple markdown file
            readme_content = """# Test Repository

This is a test markdown file for backward compatibility testing.

## Features

- Feature 1
- Feature 2

## Installation

```bash
pip install test-package
```
"""
            with open(os.path.join(temp_dir, "README.md"), "w") as f:
                f.write(readme_content)

            # Create non-markdown files that should be ignored by default
            with open(os.path.join(temp_dir, "script.py"), "w") as f:
                f.write("def test(): pass")

            with open(os.path.join(temp_dir, "config.json"), "w") as f:
                f.write('{"name": "test"}')

            # Test with MultiFileDiscovery using default markdown-only
            multi_discovery = MultiFileDiscovery()
            result = multi_discovery.discover_files(temp_dir, file_types=[".md"])

            # Should find only markdown files
            assert len(result) == 1
            assert result[0]["filename"] == "README.md"
            assert result[0]["file_type"] == ".md"
            assert result[0]["is_readme"] is True

    def test_multifile_discovery_vs_original_markdown_discovery(self):
        """Test that MultiFileDiscovery produces similar results to MarkdownDiscovery for markdown files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple markdown files with sufficient content to meet size requirements
            files_content = {
                "README.md": "# Main README\n\nThis is the main readme file with enough content to meet the minimum size requirements for the original MarkdownDiscovery class. It includes multiple paragraphs and sufficient detail to pass the 100-byte minimum threshold.",
                "docs/guide.md": "# User Guide\n\nThis is a comprehensive user guide that provides detailed instructions and information for users. It contains multiple sections and enough content to meet the size requirements.",
                "docs/api.md": "# API Documentation\n\nThis is the comprehensive API reference documentation that includes detailed information about all available endpoints, parameters, and responses. It provides examples and usage instructions.",
            }

            # Create directory structure
            os.makedirs(os.path.join(temp_dir, "docs"))

            for file_path, content in files_content.items():
                full_path = os.path.join(temp_dir, file_path)
                with open(full_path, "w") as f:
                    f.write(content)

            # Test original MarkdownDiscovery
            original_discovery = MarkdownDiscovery()
            original_result = original_discovery.discover_markdown_files(temp_dir)

            # Test MultiFileDiscovery with markdown only
            multi_discovery = MultiFileDiscovery()
            multi_result = multi_discovery.discover_files(temp_dir, file_types=[".md"])

            # Should find the same number of files
            assert len(original_result) == len(multi_result)

            # Should find the same files (by filename)
            original_filenames = {r["filename"] for r in original_result}
            multi_filenames = {r["filename"] for r in multi_result}
            assert original_filenames == multi_filenames

            # Both should prioritize README.md first
            assert original_result[0]["is_readme"] is True
            assert multi_result[0]["is_readme"] is True
            assert original_result[0]["filename"] == multi_result[0]["filename"]

    def test_supported_extensions_include_all_markdown_variants(self):
        """Test that all markdown file extensions are supported."""
        discovery = MultiFileDiscovery()

        markdown_extensions = [".md", ".markdown", ".mdown", ".mkd"]

        for ext in markdown_extensions:
            assert ext in SUPPORTED_EXTENSIONS

            # Test file type detection
            assert discovery._is_supported_file(f"test{ext}", [ext]) is True
            assert discovery._is_supported_file(f"README{ext}", [ext]) is True

    def test_file_size_limits_maintained(self):
        """Test that file size limits are properly maintained."""
        MultiFileDiscovery()

        # Check that size limits exist for all supported file types
        expected_limits = {
            ".md": 1_000_000,  # 1MB
            ".py": 1_000_000,  # 1MB
            ".ts": 1_000_000,  # 1MB
            ".tsx": 1_000_000,  # 1MB
            ".json": 100_000,  # 100KB
            ".yaml": 100_000,  # 100KB
            ".yml": 100_000,  # 100KB
            ".toml": 100_000,  # 100KB
        }

        for ext, expected_limit in expected_limits.items():
            assert ext in FILE_SIZE_LIMITS
            assert FILE_SIZE_LIMITS[ext] == expected_limit

    def test_excluded_directories_maintained(self):
        """Test that directory exclusion patterns are maintained."""
        discovery = MultiFileDiscovery()

        # Should inherit exclusion patterns from parent class
        common_excluded = {
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "build",
            "dist",
        }

        for excluded_dir in common_excluded:
            assert discovery._should_exclude_dir(excluded_dir) is True

        # Should not exclude common directories
        common_included = {"src", "docs", "lib", "tests", "examples"}

        for included_dir in common_included:
            assert discovery._should_exclude_dir(included_dir) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
