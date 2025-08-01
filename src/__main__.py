"""
Entry point for running the Crawl4AI MCP server as a module.
This allows us to use relative imports properly.
"""
import asyncio
from .crawl4ai_mcp import main

if __name__ == "__main__":
    asyncio.run(main())