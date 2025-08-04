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

# Import directly from utils.py file to avoid circular imports
utils_file = Path(__file__).parent.parent / "utils.py"

# Load utils.py module directly
import importlib.util
spec = importlib.util.spec_from_file_location("utils_module", utils_file)
utils_module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(utils_module)
    # Extract the functions we need
    get_embeddings_client = getattr(utils_module, 'get_embeddings_client', None)
    get_supabase_client = getattr(utils_module, 'get_supabase_client', None)
    health_check_gpu_acceleration = getattr(utils_module, 'health_check_gpu_acceleration', None)
    health_check_reranking_model = getattr(utils_module, 'health_check_reranking_model', None)
    cleanup_gpu_memory = getattr(utils_module, 'cleanup_gpu_memory', None)
    get_chat_client = getattr(utils_module, 'get_chat_client', None)
    create_embeddings_batch = getattr(utils_module, 'create_embeddings_batch', None)
    create_embedding = getattr(utils_module, 'create_embedding', None)
    add_documents_to_supabase = getattr(utils_module, 'add_documents_to_supabase', None)
    search_documents = getattr(utils_module, 'search_documents', None)
    extract_code_blocks = getattr(utils_module, 'extract_code_blocks', None)
    add_code_examples_to_supabase = getattr(utils_module, 'add_code_examples_to_supabase', None)
    search_code_examples = getattr(utils_module, 'search_code_examples', None)
    update_source_info = getattr(utils_module, 'update_source_info', None)
    extract_source_summary = getattr(utils_module, 'extract_source_summary', None)
    generate_contextual_embedding = getattr(utils_module, 'generate_contextual_embedding', None)
    get_device_info = getattr(utils_module, 'get_device_info', None)
    log_device_status = getattr(utils_module, 'log_device_status', None)
    cleanup_compute_memory = getattr(utils_module, 'cleanup_compute_memory', None)
    get_optimal_compute_device = getattr(utils_module, 'get_optimal_compute_device', None)
    # Sparse vector functions
    create_sparse_embedding = getattr(utils_module, 'create_sparse_embedding', None)
    create_sparse_embeddings_batch = getattr(utils_module, 'create_sparse_embeddings_batch', None)
    SparseVectorEncoder = getattr(utils_module, 'SparseVectorEncoder', None)
    
    # Import SparseVectorConfig from separate module to avoid circular imports
    try:
        from ..sparse_vector_types import SparseVectorConfig
    except ImportError:
        SparseVectorConfig = None
    
    
except Exception as e:
    print(f"DEBUG: Error loading utils.py module: {e}")
    import traceback
    traceback.print_exc()
    # If we can't load from utils.py, define minimal stubs
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
    # Sparse vector functions
    "create_sparse_embedding",
    "create_sparse_embeddings_batch",
    "SparseVectorConfig",
    "SparseVectorEncoder",
]
