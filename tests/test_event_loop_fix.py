"""
Tests for the Windows ConnectionResetError event loop fix.

These tests validate the event loop configuration utility that fixes
ConnectionResetError [WinError 10054] on Windows by using SelectorEventLoop
instead of the default ProactorEventLoop.
"""

import asyncio
import platform
import pytest
from unittest.mock import patch, MagicMock

# Import the module under test
try:
    from src.event_loop_fix import (
        is_windows,
        has_selector_event_loop_policy,
        should_use_selector_loop,
        get_current_event_loop_policy,
        setup_event_loop,
        validate_event_loop_setup,
        print_event_loop_info,
    )
except ImportError:
    from event_loop_fix import (
        is_windows,
        has_selector_event_loop_policy,
        should_use_selector_loop,
        get_current_event_loop_policy,
        setup_event_loop,
        validate_event_loop_setup,
        print_event_loop_info,
    )


class TestPlatformDetection:
    """Test platform detection functions."""

    def test_is_windows_actual_platform(self):
        """Test is_windows returns correct value for current platform."""
        expected = platform.system().lower() == "windows"
        assert is_windows() == expected

    @patch("event_loop_fix.platform.system")
    def test_is_windows_mocked_windows(self, mock_system):
        """Test is_windows returns True when platform is Windows."""
        mock_system.return_value = "Windows"
        assert is_windows() is True

    @patch("event_loop_fix.platform.system")
    def test_is_windows_mocked_linux(self, mock_system):
        """Test is_windows returns False when platform is Linux."""
        mock_system.return_value = "Linux"
        assert is_windows() is False

    @patch("event_loop_fix.platform.system")
    def test_is_windows_mocked_darwin(self, mock_system):
        """Test is_windows returns False when platform is Darwin (macOS)."""
        mock_system.return_value = "Darwin"
        assert is_windows() is False


class TestEventLoopPolicyDetection:
    """Test event loop policy detection functions."""

    def test_has_selector_event_loop_policy_actual(self):
        """Test has_selector_event_loop_policy with actual platform."""
        expected = platform.system().lower() == "windows" and hasattr(
            asyncio, "WindowsSelectorEventLoopPolicy"
        )
        assert has_selector_event_loop_policy() == expected

    @patch("event_loop_fix.is_windows")
    def test_has_selector_event_loop_policy_non_windows(self, mock_is_windows):
        """Test has_selector_event_loop_policy returns False on non-Windows."""
        mock_is_windows.return_value = False
        assert has_selector_event_loop_policy() is False

    @patch("event_loop_fix.is_windows")
    @patch("event_loop_fix.asyncio")
    def test_has_selector_event_loop_policy_windows_with_policy(
        self, mock_asyncio, mock_is_windows
    ):
        """Test has_selector_event_loop_policy returns True on Windows with policy."""
        mock_is_windows.return_value = True
        mock_asyncio.WindowsSelectorEventLoopPolicy = MagicMock()

        # Mock hasattr to return True
        with patch("builtins.hasattr", return_value=True):
            assert has_selector_event_loop_policy() is True

    @patch("event_loop_fix.is_windows")
    @patch("event_loop_fix.asyncio")
    def test_has_selector_event_loop_policy_windows_without_policy(
        self, mock_asyncio, mock_is_windows
    ):
        """Test has_selector_event_loop_policy returns False on Windows without policy."""
        mock_is_windows.return_value = True

        # Mock hasattr to return False (policy not available)
        with patch("builtins.hasattr", return_value=False):
            assert has_selector_event_loop_policy() is False

    def test_should_use_selector_loop(self):
        """Test should_use_selector_loop returns same as has_selector_event_loop_policy."""
        assert should_use_selector_loop() == has_selector_event_loop_policy()


class TestEventLoopPolicyConfiguration:
    """Test event loop policy configuration functions."""

    def test_get_current_event_loop_policy_normal(self):
        """Test get_current_event_loop_policy returns policy name."""
        policy_name = get_current_event_loop_policy()
        assert isinstance(policy_name, str)
        assert len(policy_name) > 0
        assert "Policy" in policy_name or policy_name == "Unknown"

    @patch("event_loop_fix.asyncio.get_event_loop_policy")
    def test_get_current_event_loop_policy_exception(self, mock_get_policy):
        """Test get_current_event_loop_policy handles exceptions."""
        mock_get_policy.side_effect = Exception("Test exception")
        assert get_current_event_loop_policy() == "Unknown"

    def test_setup_event_loop_preserves_functionality(self):
        """Test setup_event_loop doesn't break asyncio functionality."""
        # Store original policy
        original_policy = asyncio.get_event_loop_policy()

        try:
            # Apply fix
            result = setup_event_loop()

            # Test that asyncio still works
            async def test_async():
                return "test_result"

            # This should work regardless of event loop policy
            loop_result = asyncio.run(test_async())
            assert loop_result == "test_result"

            # Verify return value is appropriate
            if platform.system().lower() == "windows":
                if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
                    assert result == "WindowsSelectorEventLoopPolicy"
                else:
                    assert result is None
            else:
                assert result is None

        finally:
            # Restore original policy
            asyncio.set_event_loop_policy(original_policy)

    @patch("event_loop_fix.should_use_selector_loop")
    @patch("event_loop_fix.asyncio.set_event_loop_policy")
    def test_setup_event_loop_windows_success(self, mock_set_policy, mock_should_use):
        """Test setup_event_loop success path on Windows."""
        mock_should_use.return_value = True

        with patch(
            "event_loop_fix.asyncio.WindowsSelectorEventLoopPolicy"
        ) as mock_policy_class:
            result = setup_event_loop()

            mock_set_policy.assert_called_once_with(mock_policy_class())
            assert result == "WindowsSelectorEventLoopPolicy"

    @patch("event_loop_fix.should_use_selector_loop")
    def test_setup_event_loop_non_windows(self, mock_should_use):
        """Test setup_event_loop on non-Windows platforms."""
        mock_should_use.return_value = False

        result = setup_event_loop()
        assert result is None

    @patch("event_loop_fix.should_use_selector_loop")
    @patch("event_loop_fix.asyncio.set_event_loop_policy")
    def test_setup_event_loop_exception_handling(
        self, mock_set_policy, mock_should_use
    ):
        """Test setup_event_loop handles exceptions gracefully."""
        mock_should_use.return_value = True
        mock_set_policy.side_effect = Exception("Test exception")

        # Should not raise exception, but return None
        result = setup_event_loop()
        assert result is None


class TestValidationFunctions:
    """Test validation and information functions."""

    def test_validate_event_loop_setup_structure(self):
        """Test validate_event_loop_setup returns proper structure."""
        info = validate_event_loop_setup()

        required_keys = [
            "platform",
            "python_version",
            "current_policy",
            "is_windows",
            "has_selector_policy",
            "should_use_selector",
            "fix_applied",
            "recommendations",
        ]

        for key in required_keys:
            assert key in info

        assert isinstance(info["platform"], str)
        assert isinstance(info["python_version"], str)
        assert isinstance(info["current_policy"], str)
        assert isinstance(info["is_windows"], bool)
        assert isinstance(info["has_selector_policy"], bool)
        assert isinstance(info["should_use_selector"], bool)
        assert isinstance(info["fix_applied"], bool)
        assert isinstance(info["recommendations"], list)

    def test_validate_event_loop_setup_windows_detection(self):
        """Test validate_event_loop_setup properly detects Windows scenarios."""
        info = validate_event_loop_setup()

        if platform.system().lower() == "windows":
            assert info["is_windows"] is True

            if info["current_policy"] == "WindowsSelectorEventLoopPolicy":
                assert info["fix_applied"] is True
                assert any("✅" in rec for rec in info["recommendations"])
            else:
                assert info["fix_applied"] is False
                assert any("⚠️" in rec for rec in info["recommendations"])
        else:
            assert info["is_windows"] is False
            assert any("ℹ️" in rec for rec in info["recommendations"])

    def test_print_event_loop_info_no_exception(self, capsys):
        """Test print_event_loop_info runs without exception."""
        # Should not raise any exceptions
        print_event_loop_info()

        # Capture output to verify it produces some output
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert "Event Loop Configuration Information" in captured.out


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def test_import_works_from_package(self):
        """Test that imports work when called from package context."""
        # This test validates that the import structure works correctly
        try:
            from src.event_loop_fix import setup_event_loop

            assert callable(setup_event_loop)
        except ImportError:
            # Fallback import should work
            from event_loop_fix import setup_event_loop

            assert callable(setup_event_loop)

    def test_entry_point_integration_crawl4ai_mcp(self):
        """Test that the fix integrates properly with crawl4ai_mcp entry point."""
        # Test the import pattern used in crawl4ai_mcp.py
        try:
            from src.event_loop_fix import setup_event_loop
        except ImportError:
            from event_loop_fix import setup_event_loop

        # Should be callable
        assert callable(setup_event_loop)

        # Should return appropriate value
        result = setup_event_loop()
        assert result is None or isinstance(result, str)

    def test_entry_point_integration_main(self):
        """Test that the fix integrates properly with __main__ entry point."""
        # Test the import pattern used in __main__.py
        from src.event_loop_fix import setup_event_loop

        # Should be callable
        assert callable(setup_event_loop)

        # Should return appropriate value
        result = setup_event_loop()
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_http_operations_still_work(self):
        """Test that HTTP operations work after applying the fix."""
        # Apply the fix
        setup_event_loop()

        # Test basic async operation (simulating HTTP operation pattern)
        async def mock_http_operation():
            # Simulate some async work like HTTP request
            await asyncio.sleep(0.01)
            return {"success": True, "data": "test"}

        result = await mock_http_operation()
        assert result["success"] is True
        assert result["data"] == "test"

    def test_multiple_setup_calls_safe(self):
        """Test that calling setup_event_loop multiple times is safe."""
        # Store original policy
        original_policy = get_current_event_loop_policy()

        try:
            # Call setup multiple times
            result1 = setup_event_loop()
            result2 = setup_event_loop()
            result3 = setup_event_loop()

            # All should return same result or None
            assert result1 == result2 == result3

            # Policy should be consistent
            final_policy = get_current_event_loop_policy()
            assert isinstance(final_policy, str)

        finally:
            # Restore original policy for other tests
            if original_policy != "Unknown":
                # Try to restore, but don't fail if we can't
                try:
                    if original_policy == "DefaultEventLoopPolicy":
                        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
                    elif (
                        hasattr(asyncio, "WindowsProactorEventLoopPolicy")
                        and original_policy == "WindowsProactorEventLoopPolicy"
                    ):
                        asyncio.set_event_loop_policy(
                            asyncio.WindowsProactorEventLoopPolicy()
                        )
                except Exception:
                    pass  # Best effort restore


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
