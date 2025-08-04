"""
Event loop configuration utility for fixing Windows ConnectionResetError.

This module provides platform-aware event loop configuration to eliminate
the ConnectionResetError [WinError 10054] that occurs during HTTP client cleanup
in Windows asyncio ProactorEventLoop. The fix uses SelectorEventLoop on Windows
which has more graceful connection cleanup behavior, but falls back to ProactorEventLoop
when Playwright is detected since it requires subprocess support.

The issue occurs due to a race condition between client/server socket cleanup
during Windows ProactorEventLoop cleanup in _ProactorBasePipeTransport._call_connection_lost().
This is a cosmetic error that doesn't affect functionality but creates confusing logs.

Playwright Compatibility:
- Playwright requires ProactorEventLoop for subprocess support on Windows
- When Playwright is detected in imports, uses ProactorEventLoop to prevent NotImplementedError
- In this case, ConnectionResetError may still occur but functionality is preserved
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


def is_playwright_imported() -> bool:
    """
    Check if Playwright is imported in the current process.
    
    Playwright requires ProactorEventLoop for subprocess support on Windows,
    so we need to detect its presence to avoid NotImplementedError.
    
    Returns:
        bool: True if Playwright is imported, False otherwise
    """
    playwright_modules = [
        'playwright',
        'playwright.async_api',
        'playwright._impl',
        'crawl4ai'  # crawl4ai uses playwright internally
    ]
    
    for module_name in playwright_modules:
        if module_name in sys.modules:
            return True
    
    return False


def should_use_selector_loop() -> bool:
    """
    Determine if SelectorEventLoop should be used.

    Uses SelectorEventLoop on Windows when available to avoid
    ConnectionResetError during HTTP client cleanup, but falls back
    to ProactorEventLoop when Playwright is detected.

    Returns:
        bool: True if SelectorEventLoop should be used, False otherwise
    """
    if not has_selector_event_loop_policy():
        return False
    
    # Don't use SelectorEventLoop if Playwright is imported
    # since it requires ProactorEventLoop for subprocess support
    if is_playwright_imported():
        return False
    
    return True


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


def configure_http_client_limits():
    """
    Configure HTTP client connection limits to reduce ConnectionResetError.
    
    Sets environment variables to limit concurrent connections and improve
    connection cleanup behavior during intensive HTTP operations.
    """
    import os
    
    # Set connection pool limits to reduce connection pressure
    os.environ.setdefault('HTTPX_HTTP2', 'false')  # Disable HTTP/2 to reduce complexity
    os.environ.setdefault('HTTPCORE_MAX_CONNECTIONS', '50')  # Limit total connections
    os.environ.setdefault('HTTPCORE_MAX_KEEPALIVE_CONNECTIONS', '10')  # Limit keepalive
    os.environ.setdefault('HTTPCORE_KEEPALIVE_EXPIRY', '5.0')  # Shorter keepalive expiry
    
    logger.debug("Configured HTTP client connection limits for better stability")


def setup_event_loop() -> Optional[str]:
    """
    Configure appropriate event loop policy for platform and HTTP client settings.

    On Windows, configures SelectorEventLoop to avoid ConnectionResetError
    during HTTP client cleanup, but uses ProactorEventLoop if Playwright is detected
    to prevent subprocess NotImplementedError. On other platforms, uses default policy.
    
    Also configures HTTP client connection limits to reduce connection pressure.

    Returns:
        Optional[str]: Name of the configured policy, or None if no change made

    Raises:
        RuntimeError: If event loop policy configuration fails
    """
    try:
        # Configure HTTP client limits on all platforms to reduce connection pressure
        configure_http_client_limits()
        
        if is_windows():
            original_policy = get_current_event_loop_policy()
            playwright_detected = is_playwright_imported()
            
            if should_use_selector_loop():
                # Set WindowsSelectorEventLoopPolicy for better cleanup behavior
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                new_policy = get_current_event_loop_policy()

                logger.info(
                    f"Applied Windows ConnectionResetError fix: "
                    f"Changed event loop policy from {original_policy} to {new_policy}"
                )
                print(f"DEBUG: Applied Windows event loop fix - using {new_policy}")
                return new_policy
                
            elif playwright_detected:
                # Playwright detected - ensure ProactorEventLoop is used
                if hasattr(asyncio, "WindowsProactorEventLoopPolicy"):
                    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                    new_policy = get_current_event_loop_policy()
                    
                    logger.info(
                        f"Playwright detected on Windows: Using ProactorEventLoop for subprocess support. "
                        f"Changed from {original_policy} to {new_policy}. "
                        f"ConnectionResetError may still occur but functionality is preserved."
                    )
                    print(
                        f"DEBUG: Playwright detected - using {new_policy} for subprocess support. "
                        f"ConnectionResetError may still occur."
                    )
                    return new_policy
                else:
                    logger.warning("Playwright detected but WindowsProactorEventLoopPolicy not available")
                    print("DEBUG: Playwright detected but ProactorEventLoop not available")
            else:
                # Windows but SelectorEventLoop not suitable and no Playwright
                logger.info(
                    f"Windows detected but SelectorEventLoop not suitable, "
                    f"using default policy: {original_policy}"
                )
                print(f"DEBUG: Windows detected, using default policy: {original_policy}")
        else:
            # Non-Windows platform
            current_policy = get_current_event_loop_policy()
            logger.debug(f"Non-Windows platform detected, using default policy: {current_policy}")
            print(f"DEBUG: Non-Windows platform, using default event loop policy: {current_policy}")

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
    playwright_detected = is_playwright_imported()
    current_policy = get_current_event_loop_policy()
    
    info = {
        "platform": platform.system(),
        "python_version": sys.version,
        "current_policy": current_policy,
        "is_windows": is_windows(),
        "has_selector_policy": has_selector_event_loop_policy(),
        "should_use_selector": should_use_selector_loop(),
        "playwright_detected": playwright_detected,
        "fix_applied": False,
        "recommendations": [],
    }

    # Check if configuration is appropriate
    if is_windows():
        if playwright_detected:
            if current_policy == "WindowsProactorEventLoopPolicy":
                info["fix_applied"] = True
                info["recommendations"].append(
                    "OK - Playwright detected: Using ProactorEventLoop for subprocess support"
                )
                info["recommendations"].append(
                    "WARNING - ConnectionResetError may still occur but functionality is preserved"
                )
            else:
                info["recommendations"].append(
                    "WARNING - Playwright detected but ProactorEventLoop not active. "
                    "Call setup_event_loop() before asyncio.run()"
                )
        else:
            if current_policy == "WindowsSelectorEventLoopPolicy":
                info["fix_applied"] = True
                info["recommendations"].append(
                    "OK - Windows ConnectionResetError fix is active"
                )
            else:
                info["recommendations"].append(
                    "WARNING - Windows detected but SelectorEventLoop not active. "
                    "Call setup_event_loop() before asyncio.run()"
                )
    else:
        info["recommendations"].append("INFO - Non-Windows platform, no fix needed")

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
    print(f"Playwright Detected: {info['playwright_detected']}")
    print(f"Should Use Selector Loop: {info['should_use_selector']}")
    print(f"Configuration Applied: {info['fix_applied']}")
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
