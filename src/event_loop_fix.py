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
    try:
        system = platform.system()
        return isinstance(system, str) and system.lower() == "windows"
    except Exception:
        return False


def has_selector_event_loop_policy() -> bool:
    """
    Check if WindowsSelectorEventLoopPolicy is available.

    This policy is available in Python 3.7+ on Windows platforms.

    Returns:
        bool: True if WindowsSelectorEventLoopPolicy is available, False otherwise
    """
    try:
        if not is_windows():
            return False
        # Must be present and not None
        if not hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
            return False
        return getattr(asyncio, "WindowsSelectorEventLoopPolicy", None) is not None
    except Exception:
        return False


def is_playwright_imported() -> bool:
    """
    Check if Playwright is imported in the current process.

    Playwright requires ProactorEventLoop for subprocess support on Windows,
    so we need to detect its presence to avoid NotImplementedError.

    Returns:
        bool: True if Playwright is imported, False otherwise
    """
    # Consider Playwright detected if either playwright or crawl4ai is imported
    return ("playwright" in sys.modules) or ("crawl4ai" in sys.modules)


def will_playwright_be_imported() -> bool:
    """
    Detect if Playwright will be imported by checking for Crawl4AI availability.

    This is used during early event loop configuration to predict Playwright usage
    without actually importing heavy modules.

    Returns:
        bool: True if Playwright will be imported (Crawl4AI is available), False otherwise
    """
    try:
        # Try to find crawl4ai without importing it
        import importlib.util

        spec = importlib.util.find_spec("crawl4ai")
        return spec is not None
    except Exception:
        return False


def should_use_selector_loop() -> bool:
    """
    Determine if SelectorEventLoop should be used.

    Uses SelectorEventLoop on Windows when available to avoid
    ConnectionResetError during HTTP client cleanup, but falls back
    to ProactorEventLoop when Playwright is detected or will be imported.

    Returns:
        bool: True if SelectorEventLoop should be used, False otherwise
    """
    try:
        # Only consider selector loop on Windows
        if not is_windows():
            return False

        if not has_selector_event_loop_policy():
            return False

        # Don't use SelectorEventLoop if Playwright is imported or will be imported
        # since it requires ProactorEventLoop for subprocess support
        if is_playwright_imported() or will_playwright_be_imported():
            return False

        return True
    except Exception:
        return False


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

    Sets environment variables to balance connection stability with performance.
    Uses more generous limits to maintain crawling speed while preventing errors.

    Configuration can be overridden by setting environment variables before startup:
    - HTTPX_HTTP2: 'true'/'false' to enable/disable HTTP/2
    - HTTPCORE_MAX_CONNECTIONS: Max total connections (default: 200)
    - HTTPCORE_MAX_KEEPALIVE_CONNECTIONS: Max keepalive connections (default: 50)
    - HTTPCORE_KEEPALIVE_EXPIRY: Keepalive expiry in seconds (default: 30.0)
    """
    import os

    # Balanced connection pool limits for stability + performance
    # Use setdefault so existing env vars take precedence
    os.environ.setdefault(
        "HTTPX_HTTP2", "true"
    )  # Re-enable HTTP/2 for better performance
    os.environ.setdefault(
        "HTTPCORE_MAX_CONNECTIONS", "200"
    )  # Increased for better throughput
    os.environ.setdefault(
        "HTTPCORE_MAX_KEEPALIVE_CONNECTIONS", "50"
    )  # More keepalive connections
    os.environ.setdefault(
        "HTTPCORE_KEEPALIVE_EXPIRY", "30.0"
    )  # Longer keepalive for reuse

    # Log current configuration for debugging
    current_config = {
        "HTTPX_HTTP2": os.environ.get("HTTPX_HTTP2"),
        "HTTPCORE_MAX_CONNECTIONS": os.environ.get("HTTPCORE_MAX_CONNECTIONS"),
        "HTTPCORE_MAX_KEEPALIVE_CONNECTIONS": os.environ.get(
            "HTTPCORE_MAX_KEEPALIVE_CONNECTIONS"
        ),
        "HTTPCORE_KEEPALIVE_EXPIRY": os.environ.get("HTTPCORE_KEEPALIVE_EXPIRY"),
    }

    logger.debug(f"HTTP client configuration: {current_config}")
    logging.debug(f"HTTP client config - {current_config}")


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
                # Return the intended policy name for determinism in tests
                new_policy = "WindowsSelectorEventLoopPolicy"

                logger.info(
                    f"Applied Windows ConnectionResetError fix: "
                    f"Changed event loop policy from {original_policy} to {new_policy}"
                )
                logging.debug(f"Applied Windows event loop fix - using {new_policy}")
                return new_policy

            elif (
                playwright_detected
                or will_playwright_be_imported()
                or not has_selector_event_loop_policy()
            ):
                # Playwright detected or will be imported - ensure ProactorEventLoop is used
                if hasattr(asyncio, "WindowsProactorEventLoopPolicy"):
                    asyncio.set_event_loop_policy(
                        asyncio.WindowsProactorEventLoopPolicy()
                    )
                    # Return the intended policy name for determinism in tests
                    new_policy = "WindowsProactorEventLoopPolicy"

                    if playwright_detected:
                        logger.info(
                            f"Playwright detected on Windows: Using ProactorEventLoop for optimal performance. "
                            f"Changed from {original_policy} to {new_policy}."
                        )
                        logging.debug(
                            f"Playwright detected - using {new_policy} for optimal performance."
                        )
                    elif will_playwright_be_imported():
                        logger.info(
                            f"Crawl4AI/Playwright will be imported: Using ProactorEventLoop for optimal performance. "
                            f"Changed from {original_policy} to {new_policy}."
                        )
                        logging.debug(
                            f"Crawl4AI/Playwright will be imported - using {new_policy} for optimal performance."
                        )
                    return new_policy
                else:
                    logger.warning(
                        "Playwright detected/needed but WindowsProactorEventLoopPolicy not available"
                    )
                    logging.debug(
                        "Playwright detected/needed but ProactorEventLoop not available"
                    )
            else:
                # Windows but SelectorEventLoop not suitable and no Playwright
                logger.info(
                    f"Windows detected but SelectorEventLoop not suitable, "
                    f"using default policy: {original_policy}"
                )
                logging.debug(
                    f"Windows detected, using default policy: {original_policy}"
                )
        else:
            # Non-Windows platform
            current_policy = get_current_event_loop_policy()
            logger.debug(
                f"Non-Windows platform detected, using default policy: {current_policy}"
            )
            logging.debug(
                f"Non-Windows platform, using default event loop policy: {current_policy}"
            )

        return None

    except Exception as e:
        error_msg = f"Failed to configure event loop policy: {e}"
        logger.error(error_msg)
        print(f"ERROR: {error_msg}")

        # For safety, continue with default policy rather than failing
        logging.debug("Continuing with default event loop policy")
        return None


def validate_event_loop_setup() -> dict:
    """
    Validate current event loop configuration.

    Returns a dictionary with configuration information for debugging.

    Returns:
        dict: Configuration details including platform, policy, and recommendations
    """
    try:
        playwright_detected = is_playwright_imported()
        current_policy = get_current_event_loop_policy()
        plat = "Windows" if is_windows() else "Other"
        info = {
            "platform": plat,
            "python_version": sys.version,
            "current_policy": current_policy,
            "policy": current_policy,
            "is_windows": is_windows(),
            "has_selector_policy": has_selector_event_loop_policy(),
            "should_use_selector": should_use_selector_loop(),
            "playwright_detected": playwright_detected,
            "fix_applied": False,
            "recommendations": [],
        }
    except Exception as e:
        return {"error": str(e), "platform": str(platform.system())}

    # Check if configuration is appropriate
    if info.get("is_windows"):
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
    print(f"Platform: {info.get('platform', 'Unknown')}")
    print(f"Python Version: {info.get('python_version', sys.version)}")
    print(f"Current Event Loop Policy: {info.get('current_policy', 'Unknown')}")
    print(f"Windows Platform: {info.get('is_windows', False)}")
    print(
        f"WindowsSelectorEventLoopPolicy Available: {info.get('has_selector_policy', False)}"
    )
    print(f"Playwright Detected: {info.get('playwright_detected', False)}")
    print(f"Should Use Selector Loop: {info.get('should_use_selector', False)}")
    print(f"Configuration Applied: {info.get('fix_applied', False)}")
    print("\nRecommendations:")
    for rec in info.get("recommendations", []):
        print(f"  {rec}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # When run directly, display configuration information
    print_event_loop_info()

    # Test the setup function
    print("Testing setup_event_loop()...")
    result = setup_event_loop()
    if result:
        print(f"SUCCESS: Event loop policy configured: {result}")
    else:
        print("INFO: No event loop policy change needed")

    # Display final configuration
    print("\nFinal configuration:")
    print_event_loop_info()
