"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
Also includes AI hallucination detection and repository parsing tools using Neo4j knowledge graphs.

Refactored modular version using service and client abstractions.
"""

import asyncio

from .core.server import create_mcp_server
from .tools.crawl_tools import register_crawl_tools
from .tools.search_tools import register_search_tools
from .tools.source_tools import register_source_tools
from .tools.knowledge_tools import register_knowledge_tools
from .api.health import register_health_endpoints
from .config import config


def main():
    """Main entry point for the MCP server."""
    # Create the MCP server with lifespan management
    mcp = create_mcp_server()
    
    # Register all tool groups
    register_crawl_tools(mcp)
    register_search_tools(mcp)
    register_source_tools(mcp)
    
    # Register knowledge graph tools only if enabled
    if config.USE_KNOWLEDGE_GRAPH:
        register_knowledge_tools(mcp)
    
    # Register health check endpoints
    register_health_endpoints(mcp)
    
    return mcp


async def run_server():
    """Run the MCP server with the configured transport."""
    mcp = main()
    
    if config.TRANSPORT == 'sse':
        # Run the MCP server with SSE transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(run_server())