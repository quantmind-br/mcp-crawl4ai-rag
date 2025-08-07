"""

Performance validation tests for multi-file processing.

Tests processing speed, memory usage, and error handling performance.
"""
# ruff: noqa: E402

import pytest
import time
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from features.github_processor import (
    PythonProcessor,
    TypeScriptProcessor,
    ConfigProcessor,
    MarkdownProcessor,
    MultiFileDiscovery,
)


class TestPerformanceValidation:
    """Performance validation tests."""

    def test_python_processor_performance(self):
        """Test Python processor performance with realistic file."""

        processor = PythonProcessor()

        # Create a moderately sized Python file with multiple docstrings
        base_content = '''"""

Large Python module for performance testing.

This module contains multiple classes and functions to test
the performance of AST-based docstring extraction.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Union

'''

        # Generate multiple classes and functions
        class_and_function_template = '''
class TestClass{i}:
    """

    Test class number {i}.
    
    This class demonstrates performance with multiple methods
    and comprehensive docstrings for testing purposes.
    """

    
    def __init__(self, name: str, value: int = 0):
        """Initialize the test class with name and value."""

        self.name = name
        self.value = value
    
    def process_data(self, data: List[str]) -> Dict[str, Any]:
        """

        Process a list of strings and return statistics.
        
        Args:
            data: List of strings to process
            
        Returns:
            Dictionary containing processing statistics
            
        Raises:
            ValueError: When data is empty or invalid
        """

        if not data:
            raise ValueError("Data cannot be empty")
        
        return {{
            "count": len(data),
            "total_length": sum(len(s) for s in data),
            "average_length": sum(len(s) for s in data) / len(data)
        }}
    
    async def async_method(self, items: List[Any]) -> Optional[Dict[str, Union[str, int]]]:
        """

        Asynchronously process items with complex type annotations.
        
        This method demonstrates complex type annotations and async processing
        for performance testing of the AST parser.
        
        Args:
            items: List of items to process asynchronously
            
        Returns:
            Optional dictionary with processed results, or None if no items
        """

        if not items:
            return None
        
        # Simulate async processing
        processed = []
        for item in items:
            processed.append(str(item))
        
        return {{
            "processed_count": len(processed),
            "first_item": processed[0] if processed else None,
            "status": "completed"
        }}

def utility_function_{i}(param1: str, param2: int = 42) -> bool:
    """

    Utility function number {i} for testing performance.
    
    Args:
        param1: String parameter for processing
        param2: Integer parameter with default value
        
    Returns:
        Boolean result of processing
    """

    return len(param1) > param2

'''

        # Create content with 10 classes and functions
        python_content = base_content
        for i in range(10):
            python_content += class_and_function_template.format(i=i)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_content)
            temp_path = f.name

        try:
            # Measure processing time
            start_time = time.time()
            result = processor.process_file(temp_path, "test_performance.py")
            end_time = time.time()

            processing_time = end_time - start_time

            # Validate results
            assert len(result) > 0
            assert processing_time < 2.0  # Should process within 2 seconds

            # Check that we extracted docstrings from multiple sources
            doc_types = {item["type"] for item in result}
            assert "module" in doc_types
            assert "class" in doc_types
            assert "function" in doc_types

            print(f"Python processing: {len(result)} items in {processing_time:.3f}s")

        finally:
            os.unlink(temp_path)

    def test_typescript_processor_performance(self):
        """Test TypeScript processor performance with realistic file."""

        processor = TypeScriptProcessor()

        # Create a TypeScript file with multiple JSDoc comments
        base_typescript_content = """

/**
 * Performance test TypeScript module.
 * @module PerformanceTest
 */

"""

        typescript_template = """

/**
 * Interface definition for test data {i}.
 * @interface TestData{i}
 */
export interface TestData{i} {{
    id: number;
    name: string;
    value: string;
    metadata: Record<string, any>;
}}

/**
 * Service class for handling test operations {i}.
 * @class TestService{i}
 */
export class TestService{i} {{
    private data: TestData{i}[] = [];
    
    /**
     * Retrieves data by ID for service {i}.
     * @param id - The unique identifier
     * @returns The data object if found
     * @example
     * ```typescript
     * const service = new TestService{i}();
     * const data = service.getData(123);
     * ```
     */
    getData(id: number): TestData{i} | undefined {{
        return this.data.find(item => item.id === id);
    }}
    
    /**
     * Processes multiple data items asynchronously.
     * @param items - Array of data items to process
     * @returns Promise resolving to processed results
     * @throws {{ValidationError}} When items are invalid
     */
    async processItems(items: TestData{i}[]): Promise<ProcessedResult[]> {{
        const results: ProcessedResult[] = [];
        
        for (const item of items) {{
            const processed = await this.processItem(item);
            results.push(processed);
        }}
        
        return results;
    }}
}}

/**
 * Utility function for data validation {i}.
 * @param data - Data to validate
 * @returns True if valid, false otherwise
 */
export function validateData{i}(data: TestData{i}): boolean {{
    return data && data.id > 0 && data.name.length > 0;
}}

"""

        # Create content with 5 sets of interfaces, classes, and functions
        typescript_content = base_typescript_content
        for i in range(5):
            typescript_content += typescript_template.format(i=i)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
            f.write(typescript_content)
            temp_path = f.name

        try:
            start_time = time.time()
            result = processor.process_file(temp_path, "test_performance.ts")
            end_time = time.time()

            processing_time = end_time - start_time

            assert processing_time < 3.0  # Should process within 3 seconds
            print(
                f"TypeScript processing: {len(result)} items in {processing_time:.3f}s"
            )

        finally:
            os.unlink(temp_path)

    def test_multifile_discovery_performance(self):
        """Test MultiFileDiscovery performance with many files."""

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple directories and files
            file_count = 0

            # Create directory structure
            dirs = ["src", "docs", "config", "tests", "examples"]
            for dir_name in dirs:
                os.makedirs(os.path.join(temp_dir, dir_name))

            # Create files of different types
            file_types = {
                ".md": "markdown content with sufficient length for testing performance of file discovery and processing",
                ".py": '"""Python file for testing."""\ndef test(): pass',
                ".ts": "// TypeScript file\ninterface Test { id: number; }",
                ".json": '{"name": "test", "version": "1.0.0"}',
                ".yaml": "name: test\nversion: 1.0.0\ndescription: test file",
            }

            # Create 20 files of each type across directories
            for dir_name in dirs:
                for ext, content in file_types.items():
                    for i in range(4):  # 4 files per type per directory
                        filename = f"test_{i}{ext}"
                        filepath = os.path.join(temp_dir, dir_name, filename)
                        with open(filepath, "w") as f:
                            f.write(content)
                        file_count += 1

            # Test discovery performance
            discovery = MultiFileDiscovery()

            start_time = time.time()
            result = discovery.discover_files(
                temp_dir,
                file_types=[".md", ".py", ".ts", ".json", ".yaml"],
                max_files=100,
            )
            end_time = time.time()

            processing_time = end_time - start_time

            assert len(result) > 0
            assert processing_time < 5.0  # Should discover files within 5 seconds

            print(
                f"File discovery: {len(result)} files from {file_count} total in {processing_time:.3f}s"
            )

    def test_error_handling_performance(self):
        """Test performance when handling files with errors."""

        processors = [
            PythonProcessor(),
            TypeScriptProcessor(),
            ConfigProcessor(),
            MarkdownProcessor(),
        ]

        # Create problematic files
        problematic_files = {
            "syntax_error.py": "def broken(\n  # Missing closing parenthesis",
            "large_file.py": "x = 1\n" * 100000,  # Very large file
            "binary.json": "\x00\x01\x02\x03\x04",  # Binary content
            "empty.md": "",  # Empty file
            "minified.ts": "function test(){return true;}"
            + "x" * 1000,  # Minified-like
        }

        temp_files = []

        try:
            # Create temporary files
            for filename, content in problematic_files.items():
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=os.path.splitext(filename)[1], delete=False
                ) as f:
                    if filename == "binary.json":
                        f.close()  # Close first for binary write
                        with open(f.name, "wb") as bf:
                            bf.write(content.encode("latin1"))
                    else:
                        f.write(content)
                    temp_files.append(f.name)

            # Test error handling performance
            start_time = time.time()
            total_results = 0

            for temp_file in temp_files:
                for processor in processors:
                    try:
                        result = processor.process_file(
                            temp_file, os.path.basename(temp_file)
                        )
                        total_results += len(result)
                    except Exception:
                        # Errors should be handled gracefully
                        pass

            end_time = time.time()
            processing_time = end_time - start_time

            # Should handle errors quickly without hanging
            assert processing_time < 2.0

            print(
                f"Error handling: {len(temp_files)} problematic files processed in {processing_time:.3f}s"
            )

        finally:
            # Cleanup
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

    def test_memory_efficiency_estimate(self):
        """Test memory efficiency with moderately sized content."""

        # This is a basic test since we can't easily measure memory in unit tests
        processor = PythonProcessor()

        # Create content that would use significant memory if not handled efficiently
        large_docstring = (
            '"""\\n' + "This is a test line for memory efficiency.\\n" * 1000 + '"""'
        )

        python_content = f"""

{large_docstring}

def test_function():
    {large_docstring}
    pass

class TestClass:
    {large_docstring}
    
    def method(self):
        {large_docstring}
        pass
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_content)
            temp_path = f.name

        try:
            # Process large content
            result = processor.process_file(temp_path, "large_test.py")

            # Should handle large content without errors
            assert len(result) > 0

            # Verify content was processed (not just truncated)
            total_content_length = sum(len(item["content"]) for item in result)
            assert total_content_length > 1000  # Should have substantial content

            print(
                f"Memory efficiency: Processed {total_content_length} characters in {len(result)} items"
            )

        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements
