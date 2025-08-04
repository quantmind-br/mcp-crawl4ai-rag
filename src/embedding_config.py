"""
Embedding configuration utilities.

This module provides configuration utilities for embedding dimensions
and model-specific settings.
"""

import os


def get_embedding_dimensions() -> int:
    """
    Get embedding dimensions based on configuration.

    Returns:
        int: The number of dimensions for embeddings

    Raises:
        ValueError: If configured dimensions are invalid
    """
    # Check for explicit dimension configuration first
    explicit_dims = os.getenv("EMBEDDINGS_DIMENSIONS")
    if explicit_dims:
        try:
            dims = int(explicit_dims)
            if dims <= 0:
                raise ValueError("EMBEDDINGS_DIMENSIONS must be positive")
            return dims
        except ValueError as e:
            if "positive" in str(e):
                raise
            raise ValueError("EMBEDDINGS_DIMENSIONS must be a valid integer")

    # Auto-detect based on model
    embeddings_model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")

    # Model dimension mappings
    model_dimensions = {
        # OpenAI models
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        # DeepInfra models
        "Qwen/Qwen3-Embedding-0.6B": 1024,
        "BAAI/bge-large-en-v1.5": 1024,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        # Hugging Face models commonly used with DeepInfra
        "intfloat/e5-large-v2": 1024,
        "intfloat/e5-base-v2": 768,
        "intfloat/e5-small-v2": 384,
    }

    # Check if we have a known model
    if embeddings_model in model_dimensions:
        detected_dims = model_dimensions[embeddings_model]
        print(
            f"Auto-detected embedding dimensions for {embeddings_model}: {detected_dims}"
        )
        return detected_dims

    # Default fallback
    default_dims = 1536
    print(
        f"Unknown embedding model '{embeddings_model}'. Using default dimensions: {default_dims}"
    )
    return default_dims


def validate_embeddings_config() -> bool:
    """
    Validate embeddings configuration.

    Returns:
        bool: True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    # Check API key
    api_key = os.getenv("EMBEDDINGS_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key configured for embeddings. Please set EMBEDDINGS_API_KEY"
        )

    # Validate dimensions configuration
    try:
        dims = get_embedding_dimensions()
        print(f"Embeddings configuration validated - dimensions: {dims}")
        return True
    except Exception as e:
        raise ValueError(f"Invalid embeddings configuration: {e}")
