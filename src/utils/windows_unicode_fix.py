"""
Windows Unicode compatibility fixes for Crawl4AI.

This module provides patches and workarounds for Windows console Unicode encoding issues
that occur when Crawl4AI tries to output Unicode characters to cp1252 console.
"""

import os
import sys
import logging
import warnings


def patch_crawl4ai_logging():
    """
    Patch Crawl4AI logging to prevent Unicode encoding errors on Windows.

    This patches the colorama/ansitowin32 module and crawl4ai logging to handle
    Unicode characters gracefully on Windows console environments.
    """
    # Set console encoding environment variables
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # Monkey patch print function for safer Unicode handling
    original_print = print

    def safe_print(*args, **kwargs):
        """Safe print that handles Unicode encoding errors."""
        try:
            return original_print(*args, **kwargs)
        except UnicodeEncodeError:
            # Fallback: encode problematic characters as ASCII
            safe_args = []
            for arg in args:
                if isinstance(arg, str):
                    # Replace problematic Unicode chars with ASCII equivalents
                    safe_arg = arg.encode("ascii", "replace").decode("ascii")
                    safe_args.append(safe_arg)
                else:
                    safe_args.append(arg)
            return original_print(*safe_args, **kwargs)

    # Replace built-in print
    import builtins

    builtins.print = safe_print


def patch_colorama_output():
    """
    Patch colorama to handle Unicode encoding errors gracefully.

    This prevents the UnicodeEncodeError that occurs when colorama tries to
    write Unicode characters to Windows console.
    """
    try:
        import colorama.ansitowin32 as ansitowin32

        # Store original write method
        original_write = ansitowin32.AnsiToWin32.write

        def safe_write(self, text):
            """Safe write method that handles Unicode encoding errors."""
            try:
                return original_write(self, text)
            except UnicodeEncodeError:
                # Fallback: replace problematic characters
                safe_text = text.encode("ascii", "replace").decode("ascii")
                return original_write(self, safe_text)

        # Apply patch
        ansitowin32.AnsiToWin32.write = safe_write

    except ImportError:
        # colorama not available, skip patch
        pass


def configure_safe_logging():
    """
    Configure logging to handle Unicode characters safely on Windows.

    This sets up logging handlers with appropriate encoding and error handling
    to prevent UnicodeEncodeError during log output.
    """
    import platform

    if platform.system().lower() != "windows":
        return

    # Configure root logger with safe encoding
    root_logger = logging.getLogger()

    # Remove existing handlers that might cause Unicode issues
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            root_logger.removeHandler(handler)

    # Add safe console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Use a formatter that handles Unicode safely
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Set encoding explicitly for the handler
    if hasattr(console_handler.stream, "reconfigure"):
        try:
            console_handler.stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    root_logger.addHandler(console_handler)


def setup_windows_unicode_compatibility():
    """
    Set up all Windows Unicode compatibility fixes.

    This should be called early in the application startup process,
    before any Crawl4AI imports or AsyncWebCrawler initialization.
    """
    import platform

    if platform.system().lower() != "windows":
        return

    # Apply all patches
    patch_crawl4ai_logging()
    patch_colorama_output()
    configure_safe_logging()

    # Suppress specific Unicode-related warnings
    warnings.filterwarnings("ignore", category=UnicodeWarning)

    # Set environment variables for safer Unicode handling
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONLEGACYWINDOWSSTDIO", "1")


def create_safe_crawler_config():
    """
    Create AsyncWebCrawler configuration that avoids Unicode issues.

    Returns:
        dict: Safe configuration parameters for AsyncWebCrawler
    """
    try:
        from crawl4ai import BrowserConfig

        return BrowserConfig(
            headless=True,
            verbose=False,  # Disable verbose to prevent Unicode output
            # Additional safety configurations
            browser_type="chromium",
            accept_downloads=False,
            java_script_enabled=True,
            use_managed_browser=True,
        )
    except ImportError:
        # Fallback if crawl4ai not available
        return {
            "headless": True,
            "verbose": False,
        }


if __name__ == "__main__":
    # Test the fixes
    print("Testing Windows Unicode compatibility fixes...")
    setup_windows_unicode_compatibility()
    print("SUCCESS: Unicode compatibility fixes applied")
