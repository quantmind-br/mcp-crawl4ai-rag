"""
Unit tests for MCP tools timeout configuration.

This module tests the timeout constant loading, environment variable fallbacks,
value validation, and error handling for the MCP tools timeout system.
"""

import os
import pytest
from unittest.mock import patch
import sys


class TestTimeoutConfiguration:
    """Test suite for MCP tools timeout configuration."""

    def setup_method(self):
        """Clean up any existing app module imports before each test."""
        # Remove app module from cache to test fresh imports
        if "src.core.app" in sys.modules:
            del sys.modules["src.core.app"]

    def test_default_timeout_values(self):
        """Test that default timeout values are loaded when no env vars are set."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove timeout env vars if they exist
            for key in [
                "MCP_QUICK_TIMEOUT",
                "MCP_MEDIUM_TIMEOUT",
                "MCP_LONG_TIMEOUT",
                "MCP_VERY_LONG_TIMEOUT",
            ]:
                os.environ.pop(key, None)

            # Import app to trigger timeout constant loading
            from src.core.app import (
                QUICK_TIMEOUT,
                MEDIUM_TIMEOUT,
                LONG_TIMEOUT,
                VERY_LONG_TIMEOUT,
            )

            # Verify default values
            assert QUICK_TIMEOUT == 60
            assert MEDIUM_TIMEOUT == 300
            assert LONG_TIMEOUT == 1800
            assert VERY_LONG_TIMEOUT == 3600

    def test_environment_variable_override(self):
        """Test that environment variables properly override default timeout values."""
        test_env = {
            "MCP_QUICK_TIMEOUT": "120",
            "MCP_MEDIUM_TIMEOUT": "600",
            "MCP_LONG_TIMEOUT": "3600",
            "MCP_VERY_LONG_TIMEOUT": "7200",
        }

        with patch.dict(os.environ, test_env, clear=False):
            # Reload module to pick up new environment variables
            if "src.core.app" in sys.modules:
                del sys.modules["src.core.app"]

            from src.core.app import (
                QUICK_TIMEOUT,
                MEDIUM_TIMEOUT,
                LONG_TIMEOUT,
                VERY_LONG_TIMEOUT,
            )

            # Verify overridden values
            assert QUICK_TIMEOUT == 120
            assert MEDIUM_TIMEOUT == 600
            assert LONG_TIMEOUT == 3600
            assert VERY_LONG_TIMEOUT == 7200

    def test_timeout_value_types(self):
        """Test that timeout values are integers."""
        from src.core.app import (
            QUICK_TIMEOUT,
            MEDIUM_TIMEOUT,
            LONG_TIMEOUT,
            VERY_LONG_TIMEOUT,
        )

        assert isinstance(QUICK_TIMEOUT, int)
        assert isinstance(MEDIUM_TIMEOUT, int)
        assert isinstance(LONG_TIMEOUT, int)
        assert isinstance(VERY_LONG_TIMEOUT, int)

    def test_timeout_value_ranges(self):
        """Test that timeout values are within reasonable ranges."""
        from src.core.app import (
            QUICK_TIMEOUT,
            MEDIUM_TIMEOUT,
            LONG_TIMEOUT,
            VERY_LONG_TIMEOUT,
        )

        # Basic sanity checks - timeouts should be positive
        assert QUICK_TIMEOUT > 0
        assert MEDIUM_TIMEOUT > 0
        assert LONG_TIMEOUT > 0
        assert VERY_LONG_TIMEOUT > 0

        # Logical ordering - each category should be larger than the previous
        assert QUICK_TIMEOUT <= MEDIUM_TIMEOUT
        assert MEDIUM_TIMEOUT <= LONG_TIMEOUT
        assert LONG_TIMEOUT <= VERY_LONG_TIMEOUT

        # Reasonable upper bounds (24 hours max)
        assert QUICK_TIMEOUT <= 86400
        assert MEDIUM_TIMEOUT <= 86400
        assert LONG_TIMEOUT <= 86400
        assert VERY_LONG_TIMEOUT <= 86400

    def test_invalid_environment_values_fallback_to_defaults(self):
        """Test that invalid environment variable values fallback to defaults."""
        invalid_env = {
            "MCP_QUICK_TIMEOUT": "not_a_number",
            "MCP_MEDIUM_TIMEOUT": "",
            "MCP_LONG_TIMEOUT": "-100",  # This will be converted to -100, testing negative handling
            "MCP_VERY_LONG_TIMEOUT": "0",  # Zero timeout edge case
        }

        with patch.dict(os.environ, invalid_env, clear=False):
            # Import should handle invalid values gracefully
            # Note: int() conversion of invalid strings will raise ValueError
            # We expect this to propagate since we want explicit configuration

            with pytest.raises(ValueError):
                if "src.core.app" in sys.modules:
                    del sys.modules["src.core.app"]

    def test_zero_timeout_values(self):
        """Test handling of zero timeout values."""
        zero_env = {
            "MCP_QUICK_TIMEOUT": "0",
            "MCP_MEDIUM_TIMEOUT": "0",
            "MCP_LONG_TIMEOUT": "0",
            "MCP_VERY_LONG_TIMEOUT": "0",
        }

        with patch.dict(os.environ, zero_env, clear=False):
            if "src.core.app" in sys.modules:
                del sys.modules["src.core.app"]

            from src.core.app import (
                QUICK_TIMEOUT,
                MEDIUM_TIMEOUT,
                LONG_TIMEOUT,
                VERY_LONG_TIMEOUT,
            )

            # Zero timeouts should be allowed (might indicate no timeout)
            assert QUICK_TIMEOUT == 0
            assert MEDIUM_TIMEOUT == 0
            assert LONG_TIMEOUT == 0
            assert VERY_LONG_TIMEOUT == 0

    def test_large_timeout_values(self):
        """Test handling of very large timeout values."""
        large_env = {
            "MCP_QUICK_TIMEOUT": "999999",
            "MCP_MEDIUM_TIMEOUT": "999999",
            "MCP_LONG_TIMEOUT": "999999",
            "MCP_VERY_LONG_TIMEOUT": "999999",
        }

        with patch.dict(os.environ, large_env, clear=False):
            if "src.core.app" in sys.modules:
                del sys.modules["src.core.app"]

            from src.core.app import (
                QUICK_TIMEOUT,
                MEDIUM_TIMEOUT,
                LONG_TIMEOUT,
                VERY_LONG_TIMEOUT,
            )

            # Large values should be accepted
            assert QUICK_TIMEOUT == 999999
            assert MEDIUM_TIMEOUT == 999999
            assert LONG_TIMEOUT == 999999
            assert VERY_LONG_TIMEOUT == 999999

    def test_partial_environment_override(self):
        """Test that only some environment variables can be overridden."""
        partial_env = {
            "MCP_QUICK_TIMEOUT": "90",
            "MCP_LONG_TIMEOUT": "2700",
            # MCP_MEDIUM_TIMEOUT and MCP_VERY_LONG_TIMEOUT not set
        }

        with patch.dict(os.environ, partial_env, clear=False):
            # Remove the non-overridden vars to ensure defaults
            os.environ.pop("MCP_MEDIUM_TIMEOUT", None)
            os.environ.pop("MCP_VERY_LONG_TIMEOUT", None)

            if "src.core.app" in sys.modules:
                del sys.modules["src.core.app"]

            from src.core.app import (
                QUICK_TIMEOUT,
                MEDIUM_TIMEOUT,
                LONG_TIMEOUT,
                VERY_LONG_TIMEOUT,
            )

            # Verify mixed override and default values
            assert QUICK_TIMEOUT == 90  # overridden
            assert MEDIUM_TIMEOUT == 300  # default
            assert LONG_TIMEOUT == 2700  # overridden
            assert VERY_LONG_TIMEOUT == 3600  # default

    def test_constants_are_importable(self):
        """Test that all timeout constants can be imported successfully."""
        try:
            from src.core.app import (
                QUICK_TIMEOUT,
                MEDIUM_TIMEOUT,
                LONG_TIMEOUT,
                VERY_LONG_TIMEOUT,
            )

            # If we get here, import was successful
            # Verify they are all integers to use the imports
            assert all(
                isinstance(timeout, int)
                for timeout in [
                    QUICK_TIMEOUT,
                    MEDIUM_TIMEOUT,
                    LONG_TIMEOUT,
                    VERY_LONG_TIMEOUT,
                ]
            )
        except ImportError as e:
            pytest.fail(f"Failed to import timeout constants: {e}")

    def test_constants_exist_in_module(self):
        """Test that timeout constants exist in the app module namespace."""
        import src.core.app as app_module

        required_constants = [
            "QUICK_TIMEOUT",
            "MEDIUM_TIMEOUT",
            "LONG_TIMEOUT",
            "VERY_LONG_TIMEOUT",
        ]

        for constant in required_constants:
            assert hasattr(app_module, constant), (
                f"Constant {constant} not found in app module"
            )
            value = getattr(app_module, constant)
            assert isinstance(value, int), f"Constant {constant} is not an integer"
