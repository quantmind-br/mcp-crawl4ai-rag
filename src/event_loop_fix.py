"""
Event loop configuration utility for fixing Windows ConnectionResetError.

This module provides platform-aware event loop configuration to eliminate
the ConnectionResetError [WinError 10054] that occurs during HTTP client cleanup
in Windows asyncio ProactorEventLoop. The fix uses SelectorEventLoop on Windows
which has more graceful connection cleanup behavior.

The issue occurs due to a race condition between client/server socket cleanup
during Windows ProactorEventLoop cleanup in _ProactorBasePipeTransport._call_connection_lost().
This is a cosmetic error that doesn't affect functionality but creates confusing logs.
"""

import sys
import asyncio
import platform
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def is_windows() -> bool:
    """
    Check if running on Windows platform.

    Returns:
        bool: True if running on Windows, False otherwise
    """
    return platform.system().lower() == "windows"


def has_selector_event_loop_policy() -> bool:
    """
    Check if WindowsSelectorEventLoopPolicy is available.

    This policy is available in Python 3.7+ on Windows platforms.

    Returns:
        bool: True if WindowsSelectorEventLoopPolicy is available, False otherwise
    """
    return is_windows() and hasattr(asyncio, "WindowsSelectorEventLoopPolicy")


def should_use_selector_loop() -> bool:
    """
    Determine if SelectorEventLoop should be used.

    Uses SelectorEventLoop on Windows when available to avoid
    ConnectionResetError during HTTP client cleanup.

    Returns:
        bool: True if SelectorEventLoop should be used, False otherwise
    """
    return has_selector_event_loop_policy()


def get_current_event_loop_policy() -> str:
    """
    Get the name of the current event loop policy.

    Returns:
        str: Name of the current event loop policy class
    """
    try:
        policy = asyncio.get_event_loop_policy()
        return type(policy).__name__
    except Exception:
        return "Unknown"


def setup_event_loop() -> Optional[str]:
    """
    Configure appropriate event loop policy for platform.

    On Windows, configures SelectorEventLoop to avoid ConnectionResetError
    during HTTP client cleanup. On other platforms, uses default policy.

    Returns:
        Optional[str]: Name of the configured policy, or None if no change made

    Raises:
        RuntimeError: If event loop policy configuration fails
    """
    try:
        if should_use_selector_loop():
            # Store original policy for debugging
            original_policy = get_current_event_loop_policy()

            # Set WindowsSelectorEventLoopPolicy for better cleanup behavior
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

            new_policy = get_current_event_loop_policy()

            logger.info(
                f"Applied Windows ConnectionResetError fix: "
                f"Changed event loop policy from {original_policy} to {new_policy}"
            )

            print(f"DEBUG: Applied Windows event loop fix - using {new_policy}")

            return new_policy

        else:
            # Use default policy on non-Windows platforms or if SelectorEventLoop unavailable
            current_policy = get_current_event_loop_policy()

            if is_windows():
                logger.info(
                    f"WindowsSelectorEventLoopPolicy not available, "
                    f"using default policy: {current_policy}"
                )
                print(
                    f"DEBUG: WindowsSelectorEventLoopPolicy not available, using {current_policy}"
                )
            else:
                logger.debug(
                    f"Non-Windows platform detected, using default policy: {current_policy}"
                )
                print(
                    f"DEBUG: Non-Windows platform, using default event loop policy: {current_policy}"
                )

            return None

    except Exception as e:
        error_msg = f"Failed to configure event loop policy: {e}"
        logger.error(error_msg)
        print(f"ERROR: {error_msg}")

        # For safety, continue with default policy rather than failing
        print("DEBUG: Continuing with default event loop policy")
        return None


def validate_event_loop_setup() -> dict:
    """
    Validate current event loop configuration.

    Returns a dictionary with configuration information for debugging.

    Returns:
        dict: Configuration details including platform, policy, and recommendations
    """
    info = {
        "platform": platform.system(),
        "python_version": sys.version,
        "current_policy": get_current_event_loop_policy(),
        "is_windows": is_windows(),
        "has_selector_policy": has_selector_event_loop_policy(),
        "should_use_selector": should_use_selector_loop(),
        "fix_applied": False,
        "recommendations": [],
    }

    # Check if fix is properly applied
    if is_windows():
        if info["current_policy"] == "WindowsSelectorEventLoopPolicy":
            info["fix_applied"] = True
            info["recommendations"].append(
                "✅ Windows ConnectionResetError fix is active"
            )
        else:
            info["recommendations"].append(
                "⚠️  Windows detected but SelectorEventLoop not active. "
                "Call setup_event_loop() before asyncio.run()"
            )
    else:
        info["recommendations"].append("ℹ️  Non-Windows platform, no fix needed")

    return info


def print_event_loop_info():
    """
    Print detailed event loop configuration information for debugging.
    """
    info = validate_event_loop_setup()

    print("\n" + "=" * 60)
    print("Event Loop Configuration Information")
    print("=" * 60)
    print(f"Platform: {info['platform']}")
    print(f"Python Version: {info['python_version']}")
    print(f"Current Event Loop Policy: {info['current_policy']}")
    print(f"Windows Platform: {info['is_windows']}")
    print(f"WindowsSelectorEventLoopPolicy Available: {info['has_selector_policy']}")
    print(f"Should Use Selector Loop: {info['should_use_selector']}")
    print(f"Fix Applied: {info['fix_applied']}")
    print("\nRecommendations:")
    for rec in info["recommendations"]:
        print(f"  {rec}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # When run directly, display configuration information
    print_event_loop_info()

    # Test the setup function
    print("Testing setup_event_loop()...")
    result = setup_event_loop()
    if result:
        print(f"✅ Event loop policy configured: {result}")
    else:
        print("ℹ️  No event loop policy change needed")

    # Display final configuration
    print("\nFinal configuration:")
    print_event_loop_info()
