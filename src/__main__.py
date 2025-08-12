"""
Entry point for running the Crawl4AI MCP server as a module.
This allows us to use relative imports properly.
"""

import asyncio

if __name__ == "__main__":
    # Import the new core application runner first
    from .core.app import run_server, configure_windows_logging

    # Apply Windows ConnectionResetError fix after imports (so Playwright is detected)
    from .event_loop_fix import setup_event_loop

    setup_event_loop()
    configure_windows_logging()

    # Run the server
    asyncio.run(run_server())
