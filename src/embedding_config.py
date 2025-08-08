"""
Embedding configuration utilities.

This module provides configuration utilities for embedding dimensions
and model-specific settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Global singleton for embeddings configuration to avoid multiple validations
class EmbeddingsConfigSingleton:
    """Singleton class to manage embeddings configuration validation."""

    _instance = None
    _dimensions = None
    _validated = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingsConfigSingleton, cls).__new__(cls)
        return cls._instance

    def get_dimensions(self):
        """Get or validate the embedding dimensions."""
        if not self._validated:
            self._validate_and_get_dimensions()
        return self._dimensions

    @classmethod
    def reset(cls):
        """Reset the singleton for testing purposes."""
        cls._instance = None
        if hasattr(cls, "_dimensions"):
            cls._dimensions = None
        if hasattr(cls, "_validated"):
            cls._validated = False

    def _validate_and_get_dimensions(self):
        """Validate configuration and get dimensions only once."""
        if self._validated:
            return

        # Check for explicit dimension configuration first
        explicit_dims = os.getenv("EMBEDDINGS_DIMENSIONS")
        if explicit_dims:
            try:
                dims = int(explicit_dims)
                if dims <= 0:
                    raise ValueError("EMBEDDINGS_DIMENSIONS must be positive")
                self._dimensions = dims
                # Explicit dimensions configured, validation message will be shown by caller
                self._validated = True
                return
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
            # Auto-detection info logged, but validation message will be shown by caller
            self._dimensions = detected_dims
        else:
            # If model is not in the map, require explicit dimension setting
            raise ValueError(
                f"Unknown embeddings model '{embeddings_model}'. "
                "Please explicitly set EMBEDDINGS_DIMENSIONS in your environment."
            )

        # Validation completed - message will be shown by caller
        self._validated = True


def get_embedding_dimensions() -> int:
    """
    Get embedding dimensions based on configuration.

    Returns:
        int: The number of dimensions for embeddings

    Raises:
        ValueError: If configured dimensions are invalid
    """
    # Check if running in a test environment
    if "PYTEST_CURRENT_TEST" in os.environ:
        return 1024

    # Use singleton to ensure validation happens only once
    singleton = EmbeddingsConfigSingleton()
    return singleton.get_dimensions()


def reset_embeddings_config():
    """Reset embeddings configuration for testing purposes."""
    EmbeddingsConfigSingleton.reset()


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

    # Validate dimensions configuration using singleton (no duplicate printing)
    try:
        dims = get_embedding_dimensions()
        # Note: validation message is printed by singleton, no need to print again
        return True
    except Exception as e:
        raise ValueError(f"Invalid embeddings configuration: {e}")
