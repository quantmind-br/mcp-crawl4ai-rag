import pytest
from src.utils.validation import validate_github_url, normalize_github_url


class TestValidation:
    def test_validate_github_url_valid(self):
        """Test that validate_github_url returns True for valid URLs"""
        valid_urls = [
            "https://github.com/owner/repo",
            "https://github.com/owner/repo.git",
            "http://github.com/owner/repo",
            "https://www.github.com/owner/repo",
        ]

        for url in valid_urls:
            is_valid, error = validate_github_url(url)
            assert is_valid is True, f"URL {url} should be valid but got error: {error}"
            assert error == ""

    def test_validate_github_url_invalid(self):
        """Test that validate_github_url returns False for invalid URLs"""
        invalid_urls = [
            None,
            "",
            "not-a-url",
            "https://gitlab.com/owner/repo",
            "https://github.com/",
            "https://github.com/owner",
            "https://github.com/owner/repo/invalid/path",
        ]

        for url in invalid_urls:
            is_valid, error = validate_github_url(url)
            assert is_valid is False, f"URL {url} should be invalid"
            assert error != ""

    def test_normalize_github_url_valid(self):
        """Test that normalize_github_url returns correct format for valid URLs"""
        test_cases = [
            ("https://github.com/owner/repo", "https://github.com/owner/repo.git"),
            ("https://github.com/owner/repo.git", "https://github.com/owner/repo.git"),
            ("http://github.com/owner/repo", "https://github.com/owner/repo.git"),
        ]

        for input_url, expected in test_cases:
            normalized = normalize_github_url(input_url)
            assert normalized == expected

    def test_normalize_github_url_invalid(self):
        """Test that normalize_github_url raises ValueError for invalid URLs"""
        invalid_urls = [None, "", "not-a-url", "https://gitlab.com/owner/repo"]

        for url in invalid_urls:
            with pytest.raises(ValueError):
                normalize_github_url(url)
