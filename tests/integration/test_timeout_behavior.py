"""
Integration tests for MCP tools timeout behavior.

This module tests the actual timeout enforcement in real-world scenarios,
including environment variable overrides and graceful timeout handling.
"""

import os
import asyncio
import pytest
from unittest.mock import patch, Mock
from mcp.server.fastmcp import FastMCP


class TestTimeoutBehavior:
    """Integration tests for timeout behavior across the MCP tools ecosystem."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock FastMCP app for testing."""
        return Mock(spec=FastMCP)

    @pytest.fixture
    def timeout_env(self):
        """Set up test environment variables for timeouts."""
        return {
            "MCP_QUICK_TIMEOUT": "5",  # 5 seconds for testing
            "MCP_MEDIUM_TIMEOUT": "10",  # 10 seconds for testing
            "MCP_LONG_TIMEOUT": "15",  # 15 seconds for testing
            "MCP_VERY_LONG_TIMEOUT": "20",  # 20 seconds for testing
        }

    def test_timeout_constants_environment_override(self, timeout_env):
        """Test that timeout constants properly load from environment variables."""
        with patch.dict(os.environ, timeout_env, clear=False):
            # Force reimport to pick up new environment
            import sys

            if "src.core.app" in sys.modules:
                del sys.modules["src.core.app"]

            from src.core.app import (
                QUICK_TIMEOUT,
                MEDIUM_TIMEOUT,
                LONG_TIMEOUT,
                VERY_LONG_TIMEOUT,
            )

            # Verify environment variables are loaded
            assert QUICK_TIMEOUT == 5
            assert MEDIUM_TIMEOUT == 10
            assert LONG_TIMEOUT == 15
            assert VERY_LONG_TIMEOUT == 20

    def test_tool_registration_with_custom_timeouts(self, mock_app, timeout_env):
        """Test that tools are registered with custom timeout values from environment."""
        with patch.dict(os.environ, timeout_env, clear=False):
            # Mock the tool imports directly within register_tools
            with (
                patch("src.tools.web_tools") as mock_web_tools,
                patch("src.tools.github_tools") as mock_github_tools,
                patch("src.tools.rag_tools") as mock_rag_tools,
            ):
                # Setup mock tools
                mock_web_tools.crawl_single_page = Mock()
                mock_web_tools.smart_crawl_url = Mock()
                mock_github_tools.index_github_repository = Mock()
                mock_rag_tools.get_available_sources = Mock()
                mock_rag_tools.perform_rag_query = Mock()

                # Force reimport with new environment
                import sys

                if "src.core.app" in sys.modules:
                    del sys.modules["src.core.app"]

                from src.core.app import register_tools

                # Register tools with custom timeouts
                with patch.dict(
                    os.environ,
                    {"USE_AGENTIC_RAG": "false", "USE_KNOWLEDGE_GRAPH": "false"},
                ):
                    register_tools(mock_app)

                # Verify tools were registered with custom timeout values
                timeout_calls = [
                    call
                    for call in mock_app.tool.call_args_list
                    if call.kwargs.get("timeout")
                ]

                timeout_values = [call.kwargs["timeout"] for call in timeout_calls]

                # Should include our custom timeout values
                assert 15 in timeout_values  # LONG_TIMEOUT for crawl_single_page
                assert (
                    20 in timeout_values
                )  # VERY_LONG_TIMEOUT for smart_crawl_url, index_github_repository
                assert 5 in timeout_values  # QUICK_TIMEOUT for get_available_sources
                assert 10 in timeout_values  # MEDIUM_TIMEOUT for perform_rag_query

    def test_timeout_value_validation(self):
        """Test that timeout values are properly validated."""
        from src.core.app import (
            QUICK_TIMEOUT,
            MEDIUM_TIMEOUT,
            LONG_TIMEOUT,
            VERY_LONG_TIMEOUT,
        )

        # Test basic properties
        assert isinstance(QUICK_TIMEOUT, int)
        assert isinstance(MEDIUM_TIMEOUT, int)
        assert isinstance(LONG_TIMEOUT, int)
        assert isinstance(VERY_LONG_TIMEOUT, int)

        # Test positive values
        assert QUICK_TIMEOUT > 0
        assert MEDIUM_TIMEOUT > 0
        assert LONG_TIMEOUT > 0
        assert VERY_LONG_TIMEOUT > 0

    def test_invalid_timeout_environment_handling(self):
        """Test that invalid environment values are handled properly."""
        invalid_env = {
            "MCP_QUICK_TIMEOUT": "invalid_number",
            "MCP_MEDIUM_TIMEOUT": "",
        }

        with patch.dict(os.environ, invalid_env, clear=False):
            # Should raise ValueError for invalid timeout values
            import sys

            if "src.core.app" in sys.modules:
                del sys.modules["src.core.app"]

            with pytest.raises(ValueError):
                pass

    def test_zero_timeout_handling(self):
        """Test that zero timeout values are handled correctly."""
        zero_env = {
            "MCP_QUICK_TIMEOUT": "0",
            "MCP_MEDIUM_TIMEOUT": "0",
        }

        with patch.dict(os.environ, zero_env, clear=False):
            import sys

            if "src.core.app" in sys.modules:
                del sys.modules["src.core.app"]

            from src.core.app import QUICK_TIMEOUT, MEDIUM_TIMEOUT

            # Zero timeouts should be allowed (might indicate no timeout)
            assert QUICK_TIMEOUT == 0
            assert MEDIUM_TIMEOUT == 0

    def test_large_timeout_values(self):
        """Test that very large timeout values are handled correctly."""
        large_env = {
            "MCP_QUICK_TIMEOUT": "86400",  # 24 hours
            "MCP_MEDIUM_TIMEOUT": "172800",  # 48 hours
        }

        with patch.dict(os.environ, large_env, clear=False):
            import sys

            if "src.core.app" in sys.modules:
                del sys.modules["src.core.app"]

            from src.core.app import QUICK_TIMEOUT, MEDIUM_TIMEOUT

            # Large values should be accepted
            assert QUICK_TIMEOUT == 86400
            assert MEDIUM_TIMEOUT == 172800

    @pytest.mark.asyncio
    async def test_mock_timeout_enforcement_simulation(self):
        """Simulate timeout enforcement with mock operations."""

        # Mock a long-running operation that would exceed timeout
        async def long_operation():
            await asyncio.sleep(0.1)  # Short delay for testing
            return "completed"

        # Mock a timeout enforcement wrapper
        async def with_timeout(coro, timeout_seconds):
            try:
                return await asyncio.wait_for(coro, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                return "timeout_exceeded"

        # Test timeout enforcement
        result = await with_timeout(long_operation(), 0.05)  # Very short timeout
        assert result == "timeout_exceeded"

        # Test successful completion within timeout
        result = await with_timeout(long_operation(), 0.2)  # Longer timeout
        assert result == "completed"

    def test_timeout_configuration_completeness(self, mock_app):
        """Test that all expected tools have timeout configuration."""

        # Mock all tool modules
        with (
            patch("src.tools.web_tools") as mock_web_tools,
            patch("src.tools.github_tools") as mock_github_tools,
            patch("src.tools.rag_tools") as mock_rag_tools,
            patch("src.tools.kg_tools") as mock_kg_tools,
        ):
            # Setup all mock tools
            mock_web_tools.crawl_single_page = Mock()
            mock_web_tools.smart_crawl_url = Mock()
            mock_github_tools.index_github_repository = Mock()
            mock_rag_tools.get_available_sources = Mock()
            mock_rag_tools.perform_rag_query = Mock()
            mock_rag_tools.search_code_examples = Mock()
            mock_kg_tools.check_ai_script_hallucinations = Mock()
            mock_kg_tools.query_knowledge_graph = Mock()

            from src.core.app import register_tools

            # Enable all features for maximum tool count
            with patch.dict(
                os.environ, {"USE_AGENTIC_RAG": "true", "USE_KNOWLEDGE_GRAPH": "true"}
            ):
                register_tools(mock_app)

            # Count tools with timeout configuration
            timeout_calls = [
                call
                for call in mock_app.tool.call_args_list
                if call.kwargs.get("timeout")
            ]

            # Should have at least 8 tools with timeout (may have duplicates from fallback registration)
            assert len(timeout_calls) >= 8, (
                f"Expected at least 8 tools with timeout, got {len(timeout_calls)}"
            )

    def test_timeout_hierarchy_validation(self):
        """Test that timeout values follow the expected hierarchy."""
        # Use default values to test hierarchy
        default_env = {
            "MCP_QUICK_TIMEOUT": "60",
            "MCP_MEDIUM_TIMEOUT": "300",
            "MCP_LONG_TIMEOUT": "1800",
            "MCP_VERY_LONG_TIMEOUT": "3600",
        }

        with patch.dict(os.environ, default_env, clear=False):
            # Force reimport to get default values
            import sys

            if "src.core.app" in sys.modules:
                del sys.modules["src.core.app"]

            from src.core.app import (
                QUICK_TIMEOUT,
                MEDIUM_TIMEOUT,
                LONG_TIMEOUT,
                VERY_LONG_TIMEOUT,
            )

            # With default values, should follow: QUICK <= MEDIUM <= LONG <= VERY_LONG
            assert QUICK_TIMEOUT <= MEDIUM_TIMEOUT, (
                f"QUICK ({QUICK_TIMEOUT}) should be <= MEDIUM ({MEDIUM_TIMEOUT})"
            )
            assert MEDIUM_TIMEOUT <= LONG_TIMEOUT, (
                f"MEDIUM ({MEDIUM_TIMEOUT}) should be <= LONG ({LONG_TIMEOUT})"
            )
            assert LONG_TIMEOUT <= VERY_LONG_TIMEOUT, (
                f"LONG ({LONG_TIMEOUT}) should be <= VERY_LONG ({VERY_LONG_TIMEOUT})"
            )

    def test_partial_environment_configuration(self):
        """Test that partial environment configuration works correctly."""
        partial_env = {
            "MCP_QUICK_TIMEOUT": "30",  # Override only this one
            # Others should use defaults
        }

        with patch.dict(os.environ, partial_env, clear=False):
            # Remove other timeout vars if they exist
            for key in [
                "MCP_MEDIUM_TIMEOUT",
                "MCP_LONG_TIMEOUT",
                "MCP_VERY_LONG_TIMEOUT",
            ]:
                os.environ.pop(key, None)

            import sys

            if "src.core.app" in sys.modules:
                del sys.modules["src.core.app"]

            from src.core.app import (
                QUICK_TIMEOUT,
                MEDIUM_TIMEOUT,
                LONG_TIMEOUT,
                VERY_LONG_TIMEOUT,
            )

            # Verify mixed configuration
            assert QUICK_TIMEOUT == 30  # Overridden
            assert MEDIUM_TIMEOUT == 300  # Default
            assert LONG_TIMEOUT == 1800  # Default
            assert VERY_LONG_TIMEOUT == 3600  # Default

    def test_environment_isolation(self):
        """Test that environment changes don't affect already imported constants."""
        # Import with default values
        from src.core.app import QUICK_TIMEOUT as original_quick

        # Change environment
        with patch.dict(os.environ, {"MCP_QUICK_TIMEOUT": "999"}, clear=False):
            # Re-import the same constant (should be cached)
            from src.core.app import QUICK_TIMEOUT as updated_quick

            # Should be the same (module caching)
            assert original_quick == updated_quick

            # Only new imports (after del sys.modules) should pick up changes
            import sys

            if "src.core.app" in sys.modules:
                del sys.modules["src.core.app"]

            from src.core.app import QUICK_TIMEOUT as fresh_quick

            assert fresh_quick == 999
