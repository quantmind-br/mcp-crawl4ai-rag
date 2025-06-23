"""
Client module for MCP Crawl4AI RAG server.

This module contains client abstractions and implementations:
- Base client interfaces
- Chat model clients with fallback support
- Embedding model clients with fallback support
- Supabase client operations
"""

from .base import BaseClient, BaseOpenAIClient
from .chat_client import ChatClient
from .embedding_client import EmbeddingClient
from .supabase_client import SupabaseService

__all__ = ["BaseClient", "BaseOpenAIClient", "ChatClient", "EmbeddingClient", "SupabaseService"]