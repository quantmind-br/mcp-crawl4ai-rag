"""
Core module for MCP Crawl4AI RAG server.

This module contains the fundamental components for the server:
- Server initialization and configuration
- Application context management
- Lifespan management
"""

from .context import Crawl4AIContext
from .server import create_mcp_server

__all__ = ["Crawl4AIContext", "create_mcp_server"]