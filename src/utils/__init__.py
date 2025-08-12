"""
Utility modules for the Crawl4AI MCP server.
"""

# Updated imports for modular architecture
from src.features.github import (
    GitRepository as GitHubRepoManager,
    MarkdownDiscovery,
    MetadataExtractor as GitHubMetadataExtractor,
)
from .validation import validate_github_url
from .grammar_initialization import initialize_grammars_if_needed

# Re-export LLM client getters for tests expecting them here
from src.clients.llm_api_client import (
    get_chat_client,
    get_embeddings_client,
    get_chat_fallback_client,
)

__all__ = [
    "GitHubRepoManager",
    "MarkdownDiscovery",
    "GitHubMetadataExtractor",
    "validate_github_url",
    "initialize_grammars_if_needed",
    "get_chat_client",
    "get_embeddings_client",
    "get_chat_fallback_client",
]
