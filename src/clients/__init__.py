# External service client adapters

from .qdrant_client import get_qdrant_client, QdrantClientWrapper
from .llm_api_client import (
    get_chat_client,
    get_embeddings_client,
    get_chat_fallback_client,
    get_embeddings_fallback_client,
    get_adaptive_chat_client,
    validate_chat_config,
    validate_chat_fallback_config,
    validate_embeddings_fallback_config,
    get_effective_fallback_config,
)

__all__ = [
    "get_qdrant_client",
    "QdrantClientWrapper",
    "get_chat_client",
    "get_embeddings_client",
    "get_chat_fallback_client",
    "get_embeddings_fallback_client",
    "get_adaptive_chat_client",
    "validate_chat_config",
    "validate_chat_fallback_config",
    "validate_embeddings_fallback_config",
    "get_effective_fallback_config",
]