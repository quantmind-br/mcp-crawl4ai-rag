"""
API module for MCP Crawl4AI RAG server.

This module contains HTTP API endpoints:
- Health check endpoints
- Status monitoring endpoints
"""

from .health import register_health_endpoints

__all__ = ["register_health_endpoints"]