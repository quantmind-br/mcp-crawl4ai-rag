"""
Entry point for running the Crawl4AI MCP server as a module.
This allows us to use relative imports properly.
"""

import asyncio

if __name__ == "__main__":
    # Apply Windows Unicode compatibility fixes FIRST
    try:
        from .utils.windows_unicode_fix import setup_windows_unicode_compatibility

        setup_windows_unicode_compatibility()
    except ImportError:
        pass  # Module not available, continue without fixes

    # Import the new core application runner
    from .core.app import run_server, configure_windows_logging

    # Apply Windows ConnectionResetError fix after imports
    from .event_loop_fix import setup_event_loop

    # Configure Windows logging to suppress ConnectionResetError messages
    configure_windows_logging()

    # Apply event loop fix - now correctly detects Crawl4AI availability
    # to use ProactorEventLoop for optimal Playwright/Crawl4AI performance
    setup_event_loop()

    # Run the server
    asyncio.run(run_server())
