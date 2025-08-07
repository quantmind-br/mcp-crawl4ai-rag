"""
Tests for utility validation functions.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestValidationUtils:
    """Test validation utility functions."""

    def test_validate_url_valid_http(self):
        """Test URL validation with valid HTTP URLs."""
        from src.utils.validation import validate_url
        
        valid_urls = [
            "http://example.com",
            "https://example.com",
            "https://www.example.com/path?param=value",
            "http://localhost:8080",
            "https://sub.domain.com/path/to/resource.html",
        ]
        
        for url in valid_urls:
            assert validate_url(url) is True

    def test_validate_url_invalid(self):
        """Test URL validation with invalid URLs."""
        from src.utils.validation import validate_url
        
        invalid_urls = [
            "",
            "not-a-url",
            "ftp://example.com",  # Not HTTP/HTTPS
            "javascript:alert('xss')",
            "file:///local/file.txt",
            "data:text/plain;base64,SGVsbG8=",
            "   ",  # Whitespace only
            "http://",  # Incomplete
            "https://",  # Incomplete
        ]
        
        for url in invalid_urls:
            assert validate_url(url) is False

    def test_validate_url_edge_cases(self):
        """Test URL validation edge cases."""
        from src.utils.validation import validate_url
        
        edge_cases = [
            ("http://example.com:80", True),  # Default HTTP port
            ("https://example.com:443", True),  # Default HTTPS port
            ("http://192.168.1.1", True),  # IP address
            ("https://[::1]:8080", True),  # IPv6 localhost
            ("http://example.com/path with spaces", False),  # Spaces not encoded
            ("https://example.com/path%20with%20spaces", True),  # Properly encoded
        ]
        
        for url, expected in edge_cases:
            assert validate_url(url) is expected

    def test_validate_source_id_valid(self):
        """Test source ID validation with valid IDs."""
        from src.utils.validation import validate_source_id
        
        valid_ids = [
            "example.com",
            "sub.domain.com",
            "github.com",
            "stackoverflow.com",
            "docs.python.org",
        ]
        
        for source_id in valid_ids:
            assert validate_source_id(source_id) is True

    def test_validate_source_id_invalid(self):
        """Test source ID validation with invalid IDs."""
        from src.utils.validation import validate_source_id
        
        invalid_ids = [
            "",
            "   ",
            "invalid id with spaces",
            "id-with-protocol://example.com",
            "id/with/slashes",
            "id?with=query",
            "id#with-fragment",
        ]
        
        for source_id in invalid_ids:
            assert validate_source_id(source_id) is False

    def test_validate_chunk_size_valid(self):
        """Test chunk size validation with valid sizes."""
        from src.utils.validation import validate_chunk_size
        
        valid_sizes = [100, 1000, 5000, 2048, 512]
        
        for size in valid_sizes:
            assert validate_chunk_size(size) is True

    def test_validate_chunk_size_invalid(self):
        """Test chunk size validation with invalid sizes."""
        from src.utils.validation import validate_chunk_size
        
        invalid_sizes = [0, -1, -100, 10, 50, 100001, 999999]
        
        for size in invalid_sizes:
            assert validate_chunk_size(size) is False

    def test_validate_chunk_size_edge_cases(self):
        """Test chunk size validation edge cases."""
        from src.utils.validation import validate_chunk_size
        
        edge_cases = [
            (100, True),   # Minimum valid
            (99, False),   # Just below minimum
            (100000, True),  # Maximum valid
            (100001, False),  # Just above maximum
        ]
        
        for size, expected in edge_cases:
            assert validate_chunk_size(size) is expected

    def test_validate_max_depth_valid(self):
        """Test max depth validation with valid depths."""
        from src.utils.validation import validate_max_depth
        
        valid_depths = [1, 2, 3, 5, 10]
        
        for depth in valid_depths:
            assert validate_max_depth(depth) is True

    def test_validate_max_depth_invalid(self):
        """Test max depth validation with invalid depths."""
        from src.utils.validation import validate_max_depth
        
        invalid_depths = [0, -1, -5, 11, 20, 100]
        
        for depth in invalid_depths:
            assert validate_max_depth(depth) is False

    def test_validate_max_concurrent_valid(self):
        """Test max concurrent validation with valid values."""
        from src.utils.validation import validate_max_concurrent
        
        valid_values = [1, 5, 10, 20, 50]
        
        for value in valid_values:
            assert validate_max_concurrent(value) is True

    def test_validate_max_concurrent_invalid(self):
        """Test max concurrent validation with invalid values."""
        from src.utils.validation import validate_max_concurrent
        
        invalid_values = [0, -1, -10, 101, 1000]
        
        for value in invalid_values:
            assert validate_max_concurrent(value) is False

    def test_validate_file_types_valid(self):
        """Test file types validation with valid types."""
        from src.utils.validation import validate_file_types
        
        valid_types = [
            [".md"],
            [".py", ".ts", ".js"],
            [".md", ".py", ".ts", ".tsx", ".json"],
            [".yaml", ".yml", ".toml"],
        ]
        
        for file_types in valid_types:
            assert validate_file_types(file_types) is True

    def test_validate_file_types_invalid(self):
        """Test file types validation with invalid types."""
        from src.utils.validation import validate_file_types
        
        invalid_types = [
            [],  # Empty list
            [".exe"],  # Unsupported type
            [".md", ".exe"],  # Mix of valid and invalid
            ["md"],  # Missing dot prefix
            [".MD"],  # Wrong case
            None,  # Not a list
            "string",  # Not a list
        ]
        
        for file_types in invalid_types:
            assert validate_file_types(file_types) is False

    def test_sanitize_filename_basic(self):
        """Test filename sanitization with basic cases."""
        from src.utils.validation import sanitize_filename
        
        test_cases = [
            ("normal_file.txt", "normal_file.txt"),
            ("file with spaces.txt", "file_with_spaces.txt"),
            ("file/with/slashes.txt", "file_with_slashes.txt"),
            ("file?with:special<chars>.txt", "file_with_special_chars_.txt"),
        ]
        
        for input_name, expected in test_cases:
            assert sanitize_filename(input_name) == expected

    def test_sanitize_filename_edge_cases(self):
        """Test filename sanitization edge cases."""
        from src.utils.validation import sanitize_filename
        
        edge_cases = [
            ("", "unnamed_file"),  # Empty string
            ("   ", "unnamed_file"),  # Whitespace only
            ("file" * 100 + ".txt", lambda x: len(x) <= 255),  # Too long
            ("..hidden_file", "_hidden_file"),  # Leading dots
            ("file.", "file_"),  # Trailing dot
        ]
        
        for input_name, expected in edge_cases:
            result = sanitize_filename(input_name)
            if callable(expected):
                assert expected(result)
            else:
                assert result == expected

    def test_validate_query_string_valid(self):
        """Test query string validation with valid queries."""
        from src.utils.validation import validate_query_string
        
        valid_queries = [
            "python programming",
            "how to use async await",
            "machine learning algorithms",
            "react hooks tutorial",
            "database design patterns",
        ]
        
        for query in valid_queries:
            assert validate_query_string(query) is True

    def test_validate_query_string_invalid(self):
        """Test query string validation with invalid queries."""
        from src.utils.validation import validate_query_string
        
        invalid_queries = [
            "",  # Empty string
            "   ",  # Whitespace only
            "a",  # Too short
            "x" * 1001,  # Too long
        ]
        
        for query in invalid_queries:
            assert validate_query_string(query) is False

    def test_validate_match_count_valid(self):
        """Test match count validation with valid counts."""
        from src.utils.validation import validate_match_count
        
        valid_counts = [1, 5, 10, 20, 50, 100]
        
        for count in valid_counts:
            assert validate_match_count(count) is True

    def test_validate_match_count_invalid(self):
        """Test match count validation with invalid counts."""
        from src.utils.validation import validate_match_count
        
        invalid_counts = [0, -1, -10, 101, 1000]
        
        for count in invalid_counts:
            assert validate_match_count(count) is False

    def test_validation_error_messages(self):
        """Test that validation functions provide helpful error context."""
        from src.utils.validation import ValidationError
        
        # Test custom validation error if it exists
        try:
            error = ValidationError("Test validation failed", field="test_field", value="test_value")
            assert str(error) == "Test validation failed"
            assert error.field == "test_field"
            assert error.value == "test_value"
        except ImportError:
            # ValidationError might not be implemented yet
            pass

    def test_validate_config_dict(self):
        """Test configuration dictionary validation."""
        from src.utils.validation import validate_config
        
        valid_config = {
            "chunk_size": 1000,
            "max_depth": 3,
            "max_concurrent": 10,
            "file_types": [".md", ".py"],
        }
        
        assert validate_config(valid_config) is True
        
        invalid_config = {
            "chunk_size": 0,  # Invalid
            "max_depth": 20,  # Invalid
            "max_concurrent": 200,  # Invalid
            "file_types": [],  # Invalid
        }
        
        assert validate_config(invalid_config) is False