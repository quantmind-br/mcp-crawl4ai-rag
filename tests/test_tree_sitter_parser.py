#!/usr/bin/env python3
"""
Unit tests for Tree-sitter parser components.

Tests the core Tree-sitter parsing functionality including:
- Language detection
- Parser factory operations
- Query pattern execution
- Multi-language parsing results
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

try:
    from knowledge_graphs.parser_factory import ParserFactory, get_global_factory
    from knowledge_graphs.tree_sitter_parser import TreeSitterParser
    from knowledge_graphs.language_parser import ParseResult
except Exception:
    ParserFactory = None
    get_global_factory = None
    TreeSitterParser = None
    ParseResult = None

pytestmark = pytest.mark.skipif(
    ParserFactory is None, reason="Tree-sitter not available in test env"
)


class TestParserFactory(unittest.TestCase):
    """Test the ParserFactory class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.factory = ParserFactory()

    def test_language_detection(self):
        """Test language detection from file extensions."""
        test_cases = [
            ("test.py", "python"),
            ("app.js", "javascript"),
            ("component.tsx", "typescript"),
            ("Main.java", "java"),
            ("main.go", "go"),
            ("lib.rs", "rust"),
            ("utils.c", "c"),
            ("helper.cpp", "cpp"),
            ("Service.cs", "c_sharp"),
            ("index.php", "php"),
            ("model.rb", "ruby"),
            ("App.kt", "kotlin"),
            ("unknown.xyz", None),
        ]

        for filename, expected_lang in test_cases:
            with self.subTest(filename=filename):
                detected = self.factory.detect_language(filename)
                self.assertEqual(
                    detected,
                    expected_lang,
                    f"Expected {expected_lang}, got {detected} for {filename}",
                )

    def test_supported_extensions(self):
        """Test supported extension listing."""
        extensions = self.factory.get_supported_extensions()
        self.assertIsInstance(extensions, set)
        self.assertGreaterEqual(len(extensions), 20)  # Should support many extensions

        # Check some key extensions
        expected_extensions = {".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp"}
        self.assertTrue(expected_extensions.issubset(extensions))

    def test_supported_languages(self):
        """Test supported language listing."""
        languages = self.factory.get_supported_languages()
        self.assertIsInstance(languages, set)
        self.assertEqual(len(languages), 12)  # Should support exactly 12 languages

        expected_languages = {
            "python",
            "javascript",
            "typescript",
            "java",
            "go",
            "rust",
            "c",
            "cpp",
            "c_sharp",
            "php",
            "ruby",
            "kotlin",
        }
        self.assertEqual(languages, expected_languages)

    def test_parser_creation(self):
        """Test parser creation for different languages."""
        test_languages = ["python", "javascript", "java", "go"]

        for language in test_languages:
            with self.subTest(language=language):
                parser = self.factory.get_parser(language)
                self.assertIsInstance(parser, TreeSitterParser)
                self.assertEqual(parser.language, language)

    def test_parser_creation_invalid_language(self):
        """Test parser creation for invalid language."""
        parser = self.factory.get_parser("invalid_language")
        self.assertIsNone(parser)

    def test_global_factory_singleton(self):
        """Test that global factory returns same instance."""
        factory1 = get_global_factory()
        factory2 = get_global_factory()
        self.assertIs(factory1, factory2)


class TestTreeSitterParser(unittest.TestCase):
    """Test the TreeSitterParser class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.factory = ParserFactory()

    def test_python_parsing(self):
        """Test parsing Python code."""
        python_code = """
import os
from pathlib import Path

class TestClass:
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value

def test_function(x, y):
    return x + y

# Test usage
obj = TestClass("hello")
result = obj.get_value()
"""

        parser = self.factory.get_parser("python")
        self.assertIsNotNone(parser)

        result = parser.parse(python_code, "test.py")
        self.assertIsInstance(result, ParseResult)

        # Check basic structure
        self.assertGreater(len(result.imports), 0)
        self.assertGreater(len(result.classes), 0)
        self.assertGreater(len(result.functions), 0)

        # Check specific content
        import_names = {imp for imp in result.imports}
        self.assertIn("os", import_names)
        self.assertIn("Path", import_names)

        class_names = {cls["name"] for cls in result.classes}
        self.assertIn("TestClass", class_names)

        function_names = {func["name"] for func in result.functions}
        self.assertIn("test_function", function_names)

    def test_javascript_parsing(self):
        """Test parsing JavaScript code."""
        js_code = """
const fs = require('fs');
import { Path } from './utils';

class TestClass {
    constructor(value) {
        this.value = value;
    }
    
    getValue() {
        return this.value;
    }
}

function testFunction(x, y) {
    return x + y;
}

// Test usage
const obj = new TestClass("hello");
const result = obj.getValue();
"""

        parser = self.factory.get_parser("javascript")
        self.assertIsNotNone(parser)

        result = parser.parse(js_code, "test.js")
        self.assertIsInstance(result, ParseResult)

        # Check basic structure
        self.assertGreaterEqual(len(result.classes), 1)
        self.assertGreaterEqual(len(result.functions), 1)

        # Check class detection
        class_names = {cls["name"] for cls in result.classes}
        self.assertIn("TestClass", class_names)

        # Check function detection
        function_names = {func["name"] for func in result.functions}
        self.assertIn("testFunction", function_names)

    def test_java_parsing(self):
        """Test parsing Java code."""
        java_code = """
import java.util.List;
import java.util.Arrays;

public class TestClass {
    private String value;
    
    public TestClass(String value) {
        this.value = value;
    }
    
    public String getValue() {
        return this.value;
    }
}

class Utils {
    public static int testFunction(int x, int y) {
        return x + y;
    }
}
"""

        parser = self.factory.get_parser("java")
        self.assertIsNotNone(parser)

        result = parser.parse(java_code, "Test.java")
        self.assertIsInstance(result, ParseResult)

        # Check basic structure
        self.assertGreaterEqual(len(result.classes), 1)
        self.assertGreaterEqual(len(result.functions), 0)  # Methods are part of classes

        # Check class detection
        class_names = {cls["name"] for cls in result.classes}
        self.assertIn("TestClass", class_names)
        self.assertIn("Utils", class_names)

    def test_unsupported_language_handling(self):
        """Test handling of unsupported languages."""
        parser = self.factory.get_parser("unsupported")
        self.assertIsNone(parser)

    def test_parsing_empty_file(self):
        """Test parsing empty file."""
        parser = self.factory.get_parser("python")
        self.assertIsNotNone(parser)

        result = parser.parse("", "empty.py")
        self.assertIsInstance(result, ParseResult)

        # Empty file should have empty results
        self.assertEqual(len(result.classes), 0)
        self.assertEqual(len(result.functions), 0)
        self.assertEqual(len(result.imports), 0)

    def test_parsing_syntax_error(self):
        """Test parsing file with syntax errors."""
        invalid_python = """
def broken_function(
    # Missing closing parenthesis and body
"""

        parser = self.factory.get_parser("python")
        self.assertIsNotNone(parser)

        # Should not crash, might return partial results
        try:
            result = parser.parse(invalid_python, "broken.py")
            self.assertIsInstance(result, ParseResult)
        except Exception as e:
            self.fail(f"Parser should handle syntax errors gracefully: {e}")


class TestMultiLanguageIntegration(unittest.TestCase):
    """Test multi-language parsing integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.factory = ParserFactory()

    def test_file_based_parsing(self):
        """Test parsing from actual files."""
        test_files = {
            "test.py": """
class PythonClass:
    def method(self):
        pass

def python_function():
    return "python"
""",
            "test.js": """
class JavaScriptClass {
    method() {
        return "javascript";
    }
}

function jsFunction() {
    return "js";
}
""",
            "Test.java": """
public class JavaClass {
    public void method() {
        // Java method
    }
}
""",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results = {}

            for filename, content in test_files.items():
                file_path = temp_path / filename
                file_path.write_text(content)

                # Get parser for file
                parser = self.factory.get_parser_for_file(str(file_path))
                self.assertIsNotNone(parser, f"No parser found for {filename}")

                # Parse file
                result = parser.parse(content, str(file_path))
                results[filename] = result

                # Basic validation
                self.assertIsInstance(result, ParseResult)
                self.assertGreaterEqual(
                    len(result.classes), 1, f"No classes found in {filename}"
                )

            # Verify all files were processed
            self.assertEqual(len(results), len(test_files))

    def test_performance_with_large_file(self):
        """Test parsing performance with larger files."""
        # Generate a large Python file
        large_python = """import os\nfrom pathlib import Path\n\n"""

        # Add many classes and functions
        for i in range(50):
            large_python += f"""
class TestClass{i}:
    def __init__(self):
        self.value = {i}
    
    def get_value(self):
        return self.value
    
    def process(self, x):
        return x * {i}

def test_function_{i}(a, b):
    return a + b + {i}
"""

        parser = self.factory.get_parser("python")
        self.assertIsNotNone(parser)

        # Time the parsing (should complete reasonably quickly)
        import time

        start_time = time.time()
        result = parser.parse(large_python, "large_test.py")
        parse_time = time.time() - start_time

        # Verify results
        self.assertIsInstance(result, ParseResult)
        self.assertEqual(len(result.classes), 50)
        self.assertEqual(len(result.functions), 50)

        # Performance check (should parse in under 2 seconds)
        self.assertLess(
            parse_time, 2.0, f"Parsing took too long: {parse_time:.2f} seconds"
        )


if __name__ == "__main__":
    print("Running Tree-sitter Parser Unit Tests...")
    print("=" * 60)

    # Run tests with verbose output
    unittest.main(verbosity=2, exit=False)

    print("\n" + "=" * 60)
    print("Tree-sitter Parser Unit Tests Complete")
