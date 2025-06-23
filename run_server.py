#!/usr/bin/env python3
"""
Entry point script for running the MCP Crawl4AI RAG server.

This script can be run directly and handles the module imports correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import and run the server
from src.crawl4ai_mcp import run_server

if __name__ == "__main__":
    asyncio.run(run_server())