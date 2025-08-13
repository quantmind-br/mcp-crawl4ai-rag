#!/usr/bin/env python3
"""
Simple entry point for the Crawl4AI MCP server.
This avoids relative import issues by running from the project root.
"""

import sys
import asyncio
from pathlib import Path

# Add src directory to Python path before importing from src
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    # Apply Windows Unicode compatibility fixes FIRST
    try:
        from utils.windows_unicode_fix import setup_windows_unicode_compatibility
        setup_windows_unicode_compatibility()
    except ImportError:
        pass  # Module not available, continue without fixes

    # Import after path setup and Unicode fixes - this is intentional for standalone scripts
    from event_loop_fix import setup_event_loop  # noqa: E402
    from core.app import run_server, configure_windows_logging  # noqa: E402

    # Configure Windows logging to suppress ConnectionResetError messages
    configure_windows_logging()

    # Apply Windows ConnectionResetError fix - now correctly detects Crawl4AI availability
    # to use ProactorEventLoop for optimal Playwright/Crawl4AI performance
    setup_event_loop()

    # Run the server with the new core structure
    asyncio.run(run_server())
