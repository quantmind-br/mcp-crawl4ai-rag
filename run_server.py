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

# Import after path setup - this is intentional for standalone scripts
from crawl4ai_mcp import main  # noqa: E402

if __name__ == "__main__":
    asyncio.run(main())
