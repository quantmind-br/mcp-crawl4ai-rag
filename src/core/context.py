"""
Application context for the Crawl4AI MCP server.

This module defines the context dataclass that holds all the main components
needed throughout the application lifecycle.
"""

from dataclasses import dataclass
from typing import Optional, Any
from crawl4ai import AsyncWebCrawler
from supabase import Client
from sentence_transformers import CrossEncoder


@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    supabase_client: Client
    reranking_model: Optional[CrossEncoder] = None
    knowledge_validator: Optional[Any] = None  # KnowledgeGraphValidator when available
    repo_extractor: Optional[Any] = None       # DirectNeo4jExtractor when available