"""
MCP Tools for Crawl4AI server.

This package contains all MCP tool implementations organized by functionality.
"""

from .web_tools import crawl_single_page, smart_crawl_url

__all__ = [
    "crawl_single_page",
    "smart_crawl_url",
]
