"""
Utility modules for the Crawl4AI MCP server.
"""

from src.features.github_processor import (
    GitHubRepoManager,
    MarkdownDiscovery,
    GitHubMetadataExtractor,
)
from .validation import validate_github_url

# Import functions from new modular structure for backward compatibility
try:
    from ..clients.qdrant_client import get_qdrant_client, QdrantClientWrapper
    from ..clients.llm_api_client import (
        get_embeddings_client,
        get_chat_client,
        get_chat_fallback_client,
        get_embeddings_fallback_client,
        get_adaptive_chat_client,
        validate_chat_config,
        validate_chat_fallback_config,
        validate_embeddings_fallback_config,
        get_effective_fallback_config,
    )
    from ..services.embedding_service import (
        create_embeddings_batch,
        create_embedding,
        create_sparse_embedding,
        create_sparse_embeddings_batch,
        generate_contextual_embedding,
        process_chunk_with_context,
        health_check_gpu_acceleration,
    )
    from ..services.rag_service import (
        add_documents_to_vector_db,
        search_documents,
        search_code_examples,
        update_source_info,
        add_code_examples_to_vector_db,
    )
    from ..device_manager import (
        get_device_info,
        cleanup_gpu_memory,
        get_optimal_device,
    )
    from ..sparse_vector_types import SparseVectorConfig
except ImportError:
    from clients.qdrant_client import get_qdrant_client, QdrantClientWrapper
    from clients.llm_api_client import (
        get_embeddings_client,
        get_chat_client,
        get_chat_fallback_client,
        get_embeddings_fallback_client,
        get_adaptive_chat_client,
        validate_chat_config,
        validate_chat_fallback_config,
        validate_embeddings_fallback_config,
        get_effective_fallback_config,
    )
    from services.embedding_service import (
        create_embeddings_batch,
        create_embedding,
        create_sparse_embedding,
        create_sparse_embeddings_batch,
        generate_contextual_embedding,
        process_chunk_with_context,
        health_check_gpu_acceleration,
    )
    from services.rag_service import (
        add_documents_to_vector_db,
        search_documents,
        search_code_examples,
        update_source_info,
        add_code_examples_to_vector_db,
    )
    from device_manager import (
        get_device_info,
        cleanup_gpu_memory,
        get_optimal_device,
    )
    from sparse_vector_types import SparseVectorConfig

# Additional utility functions
try:
    from ..utils import extract_code_blocks, generate_code_example_summary, extract_source_summary
except ImportError:
    # These functions were moved to features or services, provide fallback
    def extract_code_blocks(*args, **kwargs):
        raise NotImplementedError("extract_code_blocks moved to features/github_processor")
    
    def generate_code_example_summary(*args, **kwargs):
        raise NotImplementedError("generate_code_example_summary moved to services")
    
    def extract_source_summary(*args, **kwargs):
        raise NotImplementedError("extract_source_summary moved to services")

# Create SparseVectorEncoder reference for backward compatibility
try:
    from ..services.embedding_service import SparseVectorEncoder
except ImportError:
    from services.embedding_service import SparseVectorEncoder

__all__ = [
    "GitHubRepoManager",
    "MarkdownDiscovery", 
    "GitHubMetadataExtractor",
    "validate_github_url",
    # Backward compatibility exports
    "get_embeddings_client",
    "get_qdrant_client", 
    "QdrantClientWrapper",
    "health_check_gpu_acceleration",
    "get_chat_client",
    "get_chat_fallback_client",
    "get_embeddings_fallback_client",
    "get_adaptive_chat_client",
    "validate_chat_config",
    "validate_chat_fallback_config", 
    "validate_embeddings_fallback_config",
    "get_effective_fallback_config",
    "create_embeddings_batch",
    "create_embedding",
    "add_documents_to_vector_db",
    "search_documents",
    "extract_code_blocks",
    "add_code_examples_to_vector_db",
    "search_code_examples", 
    "update_source_info",
    "extract_source_summary",
    "generate_contextual_embedding",
    "get_device_info",
    "cleanup_gpu_memory",
    "get_optimal_device",
    # Sparse vector functions
    "create_sparse_embedding",
    "create_sparse_embeddings_batch", 
    "SparseVectorConfig",
    "SparseVectorEncoder",
    # Code example functions
    "generate_code_example_summary",
    "process_chunk_with_context",
]