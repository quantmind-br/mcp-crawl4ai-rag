"""
Utility modules for the Crawl4AI MCP server.
"""

from .github_processor import (
    GitHubRepoManager,
    MarkdownDiscovery,
    GitHubMetadataExtractor,
)
from .validation import validate_github_url

# Import functions from the main utils.py for backward compatibility
import sys
from pathlib import Path

# Add parent directory to path to import from utils.py
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from utils import (
        get_embeddings_client,
        get_supabase_client,
        health_check_gpu_acceleration,
        health_check_reranking_model,
        cleanup_gpu_memory,
        get_chat_client,
        create_embeddings_batch,
        create_embedding,
        add_documents_to_supabase,
        search_documents,
        extract_code_blocks,
        add_code_examples_to_supabase,
        search_code_examples,
        update_source_info,
        extract_source_summary,
        generate_contextual_embedding,
        get_device_info,
        log_device_status,
        cleanup_compute_memory,
        get_optimal_compute_device,
    )
except ImportError:
    # If we can't import from utils, define minimal stubs
    def get_embeddings_client():
        raise NotImplementedError("get_embeddings_client not available")

    def get_supabase_client():
        raise NotImplementedError("get_supabase_client not available")

    def health_check_gpu_acceleration():
        raise NotImplementedError("health_check_gpu_acceleration not available")

    def health_check_reranking_model():
        raise NotImplementedError("health_check_reranking_model not available")

    def cleanup_gpu_memory():
        raise NotImplementedError("cleanup_gpu_memory not available")


__all__ = [
    "GitHubRepoManager",
    "MarkdownDiscovery",
    "GitHubMetadataExtractor",
    "validate_github_url",
    # Backward compatibility exports
    "get_embeddings_client",
    "get_supabase_client",
    "health_check_gpu_acceleration",
    "health_check_reranking_model",
    "cleanup_gpu_memory",
    "get_chat_client",
    "create_embeddings_batch",
    "create_embedding",
    "add_documents_to_supabase",
    "search_documents",
    "extract_code_blocks",
    "add_code_examples_to_supabase",
    "search_code_examples",
    "update_source_info",
    "extract_source_summary",
    "generate_contextual_embedding",
    "get_device_info",
    "log_device_status",
    "cleanup_compute_memory",
    "get_optimal_compute_device",
]
