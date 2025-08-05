"""
Utility functions for the Crawl4AI MCP server with Qdrant integration.
"""

import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple, Union
import openai
import time
import logging

from qdrant_client.models import PointStruct

# Import embedding configuration utilities
try:
    from .embedding_config import get_embedding_dimensions
except ImportError:
    try:
        from embedding_config import get_embedding_dimensions
    except ImportError:
        # Fallback for when utils.py is loaded through complex import paths
        import sys
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        from embedding_config import get_embedding_dimensions

# Import our Qdrant client wrapper
try:
    from .qdrant_wrapper import QdrantClientWrapper, get_qdrant_client
except ImportError:
    from qdrant_wrapper import QdrantClientWrapper, get_qdrant_client

# Import embedding cache
try:
    from .embedding_cache import get_embedding_cache
except ImportError:
    from embedding_cache import get_embedding_cache


# Import sparse vector configuration (avoiding circular import)
try:
    from .sparse_vector_types import SparseVectorConfig
except ImportError:
    from sparse_vector_types import SparseVectorConfig


class SparseVectorEncoder:
    """
    Singleton encoder for generating BM25 sparse vectors using FastEmbed.

    Handles lazy loading and training of the FastBM25 encoder to avoid
    unnecessary initialization overhead.
    """

    _instance = None
    _encoder = None
    _trained = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_encoder(self):
        """Lazy load the FastEmbed sparse text embedding model."""
        if self._encoder is None:
            try:
                from fastembed import SparseTextEmbedding

                logging.info("Initializing FastBM25 sparse encoder (Qdrant/bm25)")
                self._encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
            except ImportError as e:
                logging.error(f"FastEmbed not available: {e}")
                raise ImportError(
                    "FastEmbed is required for sparse vectors. Install with: pip install fastembed"
                )
            except Exception as e:
                logging.error(f"Failed to initialize FastBM25 encoder: {e}")
                raise

    def encode(self, text: str) -> SparseVectorConfig:
        """
        Generate sparse vector for the given text using BM25.

        Args:
            text: Input text to encode

        Returns:
            SparseVectorConfig: Sparse vector with indices and values

        Raises:
            ImportError: If FastEmbed is not available
            ValueError: If text is empty or encoding fails
        """
        if not text or not text.strip():
            # Return empty sparse vector for empty text
            return SparseVectorConfig(indices=[], values=[])

        self._ensure_encoder()

        try:
            # CRITICAL: Handle encoder training on first use
            if not self._trained:
                logging.info("Training BM25 encoder on first use")
                # GOTCHA: Must call embed once to initialize BM25 statistics
                _ = list(self._encoder.embed([text]))
                self._trained = True
                logging.info("BM25 encoder training completed")

            # Generate sparse embedding
            sparse_embeddings = list(self._encoder.embed([text]))
            if not sparse_embeddings:
                logging.warning("FastBM25 encoder returned no embeddings")
                return SparseVectorConfig(indices=[], values=[])

            sparse_embedding = sparse_embeddings[0]

            # Convert to SparseVectorConfig format
            return SparseVectorConfig(
                indices=sparse_embedding.indices.tolist(),
                values=sparse_embedding.values.tolist(),
            )

        except Exception as e:
            logging.error(f"Failed to encode sparse vector: {e}")
            # Graceful fallback: return empty sparse vector
            return SparseVectorConfig(indices=[], values=[])

    def encode_batch(self, texts: List[str]) -> List[SparseVectorConfig]:
        """
        Generate sparse vectors for multiple texts in batch.

        Args:
            texts: List of input texts to encode

        Returns:
            List[SparseVectorConfig]: List of sparse vectors
        """
        if not texts:
            return []

        self._ensure_encoder()

        try:
            # Handle training on first batch
            if not self._trained:
                logging.info("Training BM25 encoder on first batch")
                _ = list(self._encoder.embed(texts[:1]))  # Train on first text
                self._trained = True
                logging.info("BM25 encoder training completed")

            # Generate embeddings for all texts
            sparse_embeddings = list(self._encoder.embed(texts))

            results = []
            for i, sparse_embedding in enumerate(sparse_embeddings):
                if sparse_embedding is None:
                    logging.warning(f"Empty embedding for text {i}")
                    results.append(SparseVectorConfig(indices=[], values=[]))
                else:
                    results.append(
                        SparseVectorConfig(
                            indices=sparse_embedding.indices.tolist(),
                            values=sparse_embedding.values.tolist(),
                        )
                    )

            return results

        except Exception as e:
            logging.error(f"Failed to encode sparse vector batch: {e}")
            # Graceful fallback: return empty sparse vectors
            return [SparseVectorConfig(indices=[], values=[]) for _ in texts]


# Global sparse encoder instance
_sparse_encoder = SparseVectorEncoder()


def create_sparse_embedding(text: str) -> SparseVectorConfig:
    """
    Create a sparse vector embedding for the given text using BM25.

    This is a convenience function that uses the global sparse encoder instance.

    Args:
        text: Input text to encode

    Returns:
        SparseVectorConfig: Sparse vector configuration
    """
    return _sparse_encoder.encode(text)


def create_sparse_embeddings_batch(texts: List[str]) -> List[SparseVectorConfig]:
    """
    Create sparse vector embeddings for multiple texts using BM25.

    Args:
        texts: List of input texts to encode

    Returns:
        List[SparseVectorConfig]: List of sparse vector configurations
    """
    return _sparse_encoder.encode_batch(texts)


def get_chat_client():
    """
    Get a configured OpenAI client for chat/completion operations.

    Supports flexible configuration through environment variables:
    - CHAT_API_KEY: API key for chat model
    - CHAT_API_BASE: Base URL for chat API (defaults to OpenAI)

    Returns:
        openai.OpenAI: Configured OpenAI client for chat operations

    Raises:
        ValueError: If no API key is configured
    """
    # Get configuration
    api_key = os.getenv("CHAT_API_KEY")
    base_url = os.getenv("CHAT_API_BASE")

    if not api_key:
        raise ValueError(
            "No API key configured for chat model. Please set CHAT_API_KEY"
        )

    # Log configuration for debugging (without exposing API key)
    if base_url:
        logging.debug(f"Using custom chat API endpoint: {base_url}")
    else:
        logging.debug("Using default OpenAI API endpoint")

    # Create client with optional base_url
    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        return openai.OpenAI(api_key=api_key)


def get_embeddings_client():
    """
    Get a configured OpenAI client for embeddings operations.

    Supports flexible configuration through environment variables:
    - EMBEDDINGS_API_KEY: API key for embeddings
    - EMBEDDINGS_API_BASE: Base URL for embeddings API (defaults to OpenAI)

    Returns:
        openai.OpenAI: Configured OpenAI client for embeddings operations

    Raises:
        ValueError: If no API key is configured
    """
    # Get configuration
    api_key = os.getenv("EMBEDDINGS_API_KEY")
    base_url = os.getenv("EMBEDDINGS_API_BASE")

    if not api_key:
        raise ValueError(
            "No API key configured for embeddings. Please set EMBEDDINGS_API_KEY"
        )

    # Create client with optional base_url
    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        return openai.OpenAI(api_key=api_key)


def get_chat_fallback_client():
    """
    Get a configured OpenAI client for fallback chat/completion operations.

    Supports flexible configuration through environment variables with inheritance:
    - CHAT_FALLBACK_API_KEY: API key for fallback chat (inherits CHAT_API_KEY if not set)
    - CHAT_FALLBACK_API_BASE: Base URL for fallback chat API (inherits CHAT_API_BASE if not set)

    Returns:
        openai.OpenAI: Configured OpenAI client for fallback chat operations

    Raises:
        ValueError: If no API key is configured (primary or fallback)
    """
    # Get fallback configuration with inheritance
    api_key = os.getenv("CHAT_FALLBACK_API_KEY") or os.getenv("CHAT_API_KEY")
    base_url = os.getenv("CHAT_FALLBACK_API_BASE") or os.getenv("CHAT_API_BASE")

    if not api_key:
        raise ValueError(
            "No API key configured for fallback chat model. Please set CHAT_FALLBACK_API_KEY or CHAT_API_KEY"
        )

    # Log configuration for debugging (without exposing API key)
    if base_url:
        logging.debug(f"Using fallback chat API endpoint: {base_url}")
    else:
        logging.debug("Using default OpenAI API endpoint for fallback chat")

    # Create client with optional base_url
    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        return openai.OpenAI(api_key=api_key)


def get_embeddings_fallback_client():
    """
    Get a configured OpenAI client for fallback embeddings operations.

    Supports flexible configuration through environment variables with inheritance:
    - EMBEDDINGS_FALLBACK_API_KEY: API key for fallback embeddings (inherits EMBEDDINGS_API_KEY if not set)
    - EMBEDDINGS_FALLBACK_API_BASE: Base URL for fallback embeddings API (inherits EMBEDDINGS_API_BASE if not set)

    Returns:
        openai.OpenAI: Configured OpenAI client for fallback embeddings operations

    Raises:
        ValueError: If no API key is configured (primary or fallback)
    """
    # Get fallback configuration with inheritance
    api_key = os.getenv("EMBEDDINGS_FALLBACK_API_KEY") or os.getenv(
        "EMBEDDINGS_API_KEY"
    )
    base_url = os.getenv("EMBEDDINGS_FALLBACK_API_BASE") or os.getenv(
        "EMBEDDINGS_API_BASE"
    )

    if not api_key:
        raise ValueError(
            "No API key configured for fallback embeddings. Please set EMBEDDINGS_FALLBACK_API_KEY or EMBEDDINGS_API_KEY"
        )

    # Log configuration for debugging (without exposing API key)
    if base_url:
        logging.debug(f"Using fallback embeddings API endpoint: {base_url}")
    else:
        logging.debug("Using default OpenAI API endpoint for fallback embeddings")

    # Create client with optional base_url
    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        return openai.OpenAI(api_key=api_key)


def get_adaptive_chat_client(model_preference=None):
    """
    Get an adaptive OpenAI client that tries primary first, then fallback.

    Args:
        model_preference: Optional model name preference (uses environment fallback if None)

    Returns:
        tuple: (client, model_used, is_fallback)
            - client: Configured OpenAI client
            - model_used: The model name that will be used
            - is_fallback: Boolean indicating if fallback configuration is being used

    Raises:
        ValueError: If neither primary nor fallback configuration is available
    """
    # Determine model to use
    if model_preference:
        model_used = model_preference
        is_fallback = False
    else:
        # Try primary model first
        primary_model = os.getenv("CHAT_MODEL")
        if primary_model:
            model_used = primary_model
            is_fallback = False
        else:
            # Fall back to fallback model
            fallback_model = os.getenv("CHAT_FALLBACK_MODEL") or "gpt-4o-mini"
            model_used = fallback_model
            is_fallback = True

    # Try to get appropriate client
    try:
        if not is_fallback and os.getenv("CHAT_API_KEY"):
            # Try primary client first
            client = get_chat_client()
            return (client, model_used, False)
        else:
            # Use fallback client
            client = get_chat_fallback_client()
            # If we're using fallback client, make sure we're using fallback model
            if not model_preference:
                fallback_model = os.getenv("CHAT_FALLBACK_MODEL") or "gpt-4o-mini"
                model_used = fallback_model
            return (client, model_used, True)
    except ValueError as e:
        # If primary fails, try fallback
        if not is_fallback:
            try:
                client = get_chat_fallback_client()
                fallback_model = os.getenv("CHAT_FALLBACK_MODEL") or "gpt-4o-mini"
                return (client, fallback_model, True)
            except ValueError:
                pass

        # Both failed
        raise ValueError(
            f"No valid API configuration available for chat model. "
            f"Please configure CHAT_API_KEY or CHAT_FALLBACK_API_KEY. Original error: {str(e)}"
        )


def validate_chat_config() -> bool:
    """
    Validate chat model configuration and provide helpful guidance.

    Returns:
        bool: True if configuration is valid, False otherwise

    Raises:
        ValueError: If critical configuration is missing
    """
    # Check for API key
    chat_api_key = os.getenv("CHAT_API_KEY")

    if not chat_api_key:
        raise ValueError(
            "No API key configured for chat model. Please set CHAT_API_KEY"
        )

    # Check for model configuration
    chat_model = os.getenv("CHAT_MODEL")

    if not chat_model:
        logging.warning(
            "No chat model specified. Please set CHAT_MODEL environment variable. "
            "Defaulting to configured fallback model."
        )

    # Log configuration being used
    effective_key_source = "CHAT_API_KEY"
    effective_model = chat_model or os.getenv("CHAT_FALLBACK_MODEL") or "default"
    base_url = os.getenv("CHAT_API_BASE", "default OpenAI")

    logging.debug(
        f"Chat configuration - Model: {effective_model}, Key source: {effective_key_source}, Base URL: {base_url}"
    )

    return True


def validate_chat_fallback_config() -> bool:
    """
    Validate chat fallback model configuration with inheritance support.

    Returns:
        bool: True if fallback configuration is valid (direct or inherited)

    Raises:
        ValueError: If no valid fallback configuration is available
    """
    # Check for fallback API key (direct or inherited)
    fallback_api_key = os.getenv("CHAT_FALLBACK_API_KEY") or os.getenv("CHAT_API_KEY")

    if not fallback_api_key:
        raise ValueError(
            "No API key configured for chat fallback. Please set CHAT_FALLBACK_API_KEY or CHAT_API_KEY"
        )

    # Validate base URL format if provided
    fallback_base_url = os.getenv("CHAT_FALLBACK_API_BASE") or os.getenv(
        "CHAT_API_BASE"
    )
    if fallback_base_url:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(fallback_base_url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid base URL format: {fallback_base_url}")
        except Exception as e:
            raise ValueError(f"Invalid fallback base URL configuration: {str(e)}")

    # Log effective configuration for debugging
    effective_key_source = (
        "CHAT_FALLBACK_API_KEY"
        if os.getenv("CHAT_FALLBACK_API_KEY")
        else "CHAT_API_KEY (inherited)"
    )
    effective_base_source = (
        "CHAT_FALLBACK_API_BASE"
        if os.getenv("CHAT_FALLBACK_API_BASE")
        else "CHAT_API_BASE (inherited)"
        if fallback_base_url
        else "default OpenAI"
    )
    fallback_model = os.getenv("CHAT_FALLBACK_MODEL", "gpt-4o-mini")

    logging.debug(
        f"Chat fallback configuration - Model: {fallback_model}, Key source: {effective_key_source}, Base URL source: {effective_base_source}"
    )

    return True


def validate_embeddings_fallback_config() -> bool:
    """
    Validate embeddings fallback model configuration with inheritance support.

    Returns:
        bool: True if fallback configuration is valid (direct or inherited)

    Raises:
        ValueError: If no valid fallback configuration is available
    """
    # Check for fallback API key (direct or inherited)
    fallback_api_key = os.getenv("EMBEDDINGS_FALLBACK_API_KEY") or os.getenv(
        "EMBEDDINGS_API_KEY"
    )

    if not fallback_api_key:
        raise ValueError(
            "No API key configured for embeddings fallback. Please set EMBEDDINGS_FALLBACK_API_KEY or EMBEDDINGS_API_KEY"
        )

    # Validate base URL format if provided
    fallback_base_url = os.getenv("EMBEDDINGS_FALLBACK_API_BASE") or os.getenv(
        "EMBEDDINGS_API_BASE"
    )
    if fallback_base_url:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(fallback_base_url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid base URL format: {fallback_base_url}")
        except Exception as e:
            raise ValueError(f"Invalid fallback base URL configuration: {str(e)}")

    # Log effective configuration for debugging
    effective_key_source = (
        "EMBEDDINGS_FALLBACK_API_KEY"
        if os.getenv("EMBEDDINGS_FALLBACK_API_KEY")
        else "EMBEDDINGS_API_KEY (inherited)"
    )
    effective_base_source = (
        "EMBEDDINGS_FALLBACK_API_BASE"
        if os.getenv("EMBEDDINGS_FALLBACK_API_BASE")
        else "EMBEDDINGS_API_BASE (inherited)"
        if fallback_base_url
        else "default OpenAI"
    )
    fallback_model = os.getenv("EMBEDDINGS_FALLBACK_MODEL", "text-embedding-3-small")

    logging.debug(
        f"Embeddings fallback configuration - Model: {fallback_model}, Key source: {effective_key_source}, Base URL source: {effective_base_source}"
    )

    return True


def get_effective_fallback_config():
    """
    Get the effective fallback configuration with inheritance resolved.
    Useful for debugging and monitoring.

    Returns:
        dict: Configuration details showing what will actually be used
    """
    config = {
        "chat_fallback": {
            "model": os.getenv("CHAT_FALLBACK_MODEL", "gpt-4o-mini"),
            "api_key_source": "CHAT_FALLBACK_API_KEY"
            if os.getenv("CHAT_FALLBACK_API_KEY")
            else "CHAT_API_KEY (inherited)"
            if os.getenv("CHAT_API_KEY")
            else None,
            "base_url": os.getenv("CHAT_FALLBACK_API_BASE")
            or os.getenv("CHAT_API_BASE"),
            "base_url_source": "CHAT_FALLBACK_API_BASE"
            if os.getenv("CHAT_FALLBACK_API_BASE")
            else "CHAT_API_BASE (inherited)"
            if os.getenv("CHAT_API_BASE")
            else "default OpenAI",
        },
        "embeddings_fallback": {
            "model": os.getenv("EMBEDDINGS_FALLBACK_MODEL", "text-embedding-3-small"),
            "api_key_source": "EMBEDDINGS_FALLBACK_API_KEY"
            if os.getenv("EMBEDDINGS_FALLBACK_API_KEY")
            else "EMBEDDINGS_API_KEY (inherited)"
            if os.getenv("EMBEDDINGS_API_KEY")
            else None,
            "base_url": os.getenv("EMBEDDINGS_FALLBACK_API_BASE")
            or os.getenv("EMBEDDINGS_API_BASE"),
            "base_url_source": "EMBEDDINGS_FALLBACK_API_BASE"
            if os.getenv("EMBEDDINGS_FALLBACK_API_BASE")
            else "EMBEDDINGS_API_BASE (inherited)"
            if os.getenv("EMBEDDINGS_API_BASE")
            else "default OpenAI",
        },
    }

    return config


def get_supabase_client():
    """
    DEPRECATED: Legacy function name maintained for compatibility.
    Returns Qdrant client wrapper instead.
    """
    return get_qdrant_client()


def create_embeddings_batch(
    texts: List[str],
) -> Union[List[List[float]], Tuple[List[List[float]], List[SparseVectorConfig]]]:
    """
    Create embeddings for multiple texts with Redis caching support and optional sparse vectors.

    This function implements a high-performance caching layer that:
    - Checks Redis cache for existing embeddings first
    - Only calls external APIs for cache misses
    - Stores new embeddings in cache for future use
    - Provides graceful degradation when cache is unavailable
    - Creates both dense and sparse vectors when USE_HYBRID_SEARCH=true

    Args:
        texts: List of texts to create embeddings for

    Returns:
        When USE_HYBRID_SEARCH=false: List of embeddings (each embedding is a list of floats)
        When USE_HYBRID_SEARCH=true: Tuple of (dense_vectors, sparse_vectors)
    """
    if not texts:
        # Return appropriate empty structure based on hybrid search mode
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"
        if use_hybrid_search:
            return ([], [])  # (dense_vectors, sparse_vectors)
        return []

    use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"
    embeddings_model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    final_embeddings = [None] * len(texts)
    final_sparse_vectors = [None] * len(texts) if use_hybrid_search else None

    # Try cache first if available
    cache = get_embedding_cache()
    if cache:
        cached_embeddings = cache.get_batch(texts, embeddings_model)

        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            if text in cached_embeddings:
                final_embeddings[i] = cached_embeddings[text]  # Cache hit
                logging.debug(f"Cache hit for text {i}")
            else:
                texts_to_embed.append(text)  # Cache miss
                indices_to_embed.append(i)

        if cached_embeddings:
            logging.info(
                f"Cache hits: {len(cached_embeddings)}/{len(texts)} embeddings"
            )
    else:
        # No cache available, embed all texts
        texts_to_embed = texts
        indices_to_embed = list(range(len(texts)))

    # Create embeddings for cache misses using existing retry logic
    if texts_to_embed:
        logging.info(f"Creating {len(texts_to_embed)} new embeddings via API")
        new_embeddings_list = _create_embeddings_api_call(texts_to_embed)

        # Store new embeddings in cache
        if cache and new_embeddings_list:
            new_to_cache = {
                text: emb for text, emb in zip(texts_to_embed, new_embeddings_list)
            }
            ttl = int(os.getenv("REDIS_EMBEDDING_TTL", "86400"))
            cache.set_batch(new_to_cache, embeddings_model, ttl)
            logging.debug(f"Cached {len(new_to_cache)} new embeddings")

        # Place new embeddings in correct positions
        for i, new_embedding in enumerate(new_embeddings_list):
            original_index = indices_to_embed[i]
            final_embeddings[original_index] = new_embedding

    # Generate sparse vectors if hybrid search is enabled
    if use_hybrid_search:
        # Create sparse vectors for all texts (not just cache misses)
        logging.info(f"Creating {len(texts)} sparse vectors for hybrid search")
        try:
            sparse_embeddings = create_sparse_embeddings_batch(texts)
            final_sparse_vectors = sparse_embeddings
        except Exception as e:
            logging.error(f"Failed to create sparse vectors: {e}")
            # Fallback to empty sparse vectors to maintain consistency
            final_sparse_vectors = [
                SparseVectorConfig(indices=[], values=[]) for _ in texts
            ]

    # Return appropriate format based on hybrid search mode
    if use_hybrid_search:
        return (final_embeddings, final_sparse_vectors)
    return final_embeddings


def _create_embeddings_api_call(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings via API call with retry logic (extracted from original function).

    Args:
        texts: List of texts to create embeddings for

    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []

    max_retries = 3
    retry_delay = 1.0  # Start with 1 second delay

    for retry in range(max_retries):
        try:
            embeddings_model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
            client = get_embeddings_client()
            response = client.embeddings.create(
                model=embeddings_model,
                input=texts,
                encoding_format="float",  # Explicitly set encoding format for DeepInfra compatibility
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            if retry < max_retries - 1:
                print(
                    f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}"
                )
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(
                    f"Failed to create batch embeddings after {max_retries} attempts: {e}"
                )
                # Try creating embeddings one by one as fallback
                print("Attempting to create embeddings individually...")
                embeddings = []
                successful_count = 0

                for i, text in enumerate(texts):
                    try:
                        embeddings_model = os.getenv(
                            "EMBEDDINGS_MODEL", "text-embedding-3-small"
                        )
                        client = get_embeddings_client()
                        individual_response = client.embeddings.create(
                            model=embeddings_model,
                            input=[text],
                            encoding_format="float",  # Explicitly set encoding format for DeepInfra compatibility
                        )
                        embeddings.append(individual_response.data[0].embedding)
                        successful_count += 1
                    except Exception as individual_error:
                        print(
                            f"Failed to create embedding for text {i}: {individual_error}"
                        )
                        # Add zero embedding as fallback
                        embeddings.append([0.0] * get_embedding_dimensions())

                print(
                    f"Successfully created {successful_count}/{len(texts)} embeddings individually"
                )
                return embeddings


def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using OpenAI's API.

    Args:
        text: Text to create an embedding for

    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * get_embedding_dimensions()
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
        return [0.0] * get_embedding_dimensions()


def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.

    Uses the chat model configured via CHAT_MODEL environment variable with robust fallback logic
    for handling API errors like rate limits (429), server errors (500/503), and overload scenarios.

    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for

    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    # Get chat model with modern fallback configuration
    model_choice = (
        os.getenv("CHAT_MODEL") or os.getenv("CHAT_FALLBACK_MODEL") or "gpt-4o-mini"
    )

    if not model_choice:
        print("Warning: No chat model configured. Set CHAT_MODEL environment variable.")
        return chunk, False

    # Optimize prompt for better token efficiency
    # Reduce document size for better context generation
    doc_limit = 15000 if "gemini" in model_choice.lower() else 25000

    # Create a more concise prompt for Gemini models
    if "gemini" in model_choice.lower():
        prompt = f"""Document excerpt: {full_document[:doc_limit]}

Chunk to contextualize: {chunk}

Provide 1-2 sentences of context for this chunk within the document. Be concise."""
    else:
        prompt = f"""<document> 
{full_document[:doc_limit]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

    # Adjust max_tokens based on model being used
    max_tokens = 150 if "gemini" in model_choice.lower() else 200

    try:
        # PATTERN: Get primary client explicitly (not adaptive client)
        client = get_chat_client()

        # PATTERN: Make API call with existing prompt logic
        response = client.chat.completions.create(
            model=model_choice,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides concise contextual information.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )

        # Validate response structure
        if not response.choices:
            print(f"Warning: API returned no choices. Model: {model_choice}")
            return chunk, False

        choice = response.choices[0]

        # Check if response was truncated due to length
        if choice.finish_reason == "length" and choice.message.content is None:
            print(
                f"Warning: Model {model_choice} hit token limit before generating content. Trying shorter prompt."
            )

            # Retry with much shorter prompt for Gemini
            short_prompt = f"Context for '{chunk[:100]}...' in document about: {full_document[:500]}... \nProvide brief context (1 sentence):"

            response = client.chat.completions.create(
                model=model_choice,
                messages=[{"role": "user", "content": short_prompt}],
                temperature=0.3,
                max_tokens=50,
            )

            if response.choices and response.choices[0].message.content:
                choice = response.choices[0]
            else:
                print(
                    f"Warning: Even shorter prompt failed for {model_choice}. Using original chunk."
                )
                return chunk, False

        # Extract the generated context with null check
        content = choice.message.content
        if content is None:
            print(
                f"Warning: API returned None content for contextual embedding. Model: {model_choice}"
            )
            print(f"Finish reason: {choice.finish_reason}")
            return chunk, False

        context = content.strip()
        if not context:
            print(
                f"Warning: API returned empty content for contextual embedding. Model: {model_choice}"
            )
            return chunk, False

        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"

        return contextual_text, True

    except (
        openai.APIStatusError,
        openai.RateLimitError,
        openai.InternalServerError,
    ) as primary_error:
        # CRITICAL: Log specific error type and attempt fallback
        print(
            f"Primary model {model_choice} failed ({type(primary_error).__name__}): {primary_error}"
        )
        print("Attempting fallback model...")

        try:
            # PATTERN: Try fallback client with same prompt structure
            fallback_client = get_chat_fallback_client()
            fallback_model = os.getenv("CHAT_FALLBACK_MODEL") or "gpt-4o-mini"

            # Adjust max_tokens based on fallback model being used
            fallback_max_tokens = 150 if "gemini" in fallback_model.lower() else 200

            response = fallback_client.chat.completions.create(
                model=fallback_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides concise contextual information.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=fallback_max_tokens,
            )

            # Validate response structure
            if not response.choices:
                print(
                    f"Warning: Fallback API returned no choices. Model: {fallback_model}"
                )
                return chunk, False

            choice = response.choices[0]

            # Check if response was truncated due to length
            if choice.finish_reason == "length" and choice.message.content is None:
                print(
                    f"Warning: Fallback model {fallback_model} hit token limit before generating content. Trying shorter prompt."
                )

                # Retry with much shorter prompt
                short_prompt = f"Context for '{chunk[:100]}...' in document about: {full_document[:500]}... \nProvide brief context (1 sentence):"

                response = fallback_client.chat.completions.create(
                    model=fallback_model,
                    messages=[{"role": "user", "content": short_prompt}],
                    temperature=0.3,
                    max_tokens=50,
                )

                if response.choices and response.choices[0].message.content:
                    choice = response.choices[0]
                else:
                    print(
                        f"Warning: Even shorter prompt failed for fallback model {fallback_model}. Using original chunk."
                    )
                    return chunk, False

            # Extract the generated context with null check
            content = choice.message.content
            if content is None:
                print(
                    f"Warning: Fallback API returned None content for contextual embedding. Model: {fallback_model}"
                )
                print(f"Finish reason: {choice.finish_reason}")
                return chunk, False

            context = content.strip()
            if not context:
                print(
                    f"Warning: Fallback API returned empty content for contextual embedding. Model: {fallback_model}"
                )
                return chunk, False

            # Combine the context with the original chunk
            contextual_text = f"{context}\n---\n{chunk}"

            # PATTERN: Process response identically
            print(f"Fallback model {fallback_model} succeeded")
            return contextual_text, True

        except Exception as fallback_error:
            # PATTERN: Log both errors and gracefully degrade
            print(f"Fallback model also failed: {fallback_error}")
            print("Using original chunk without contextual enhancement")
            return chunk, False

    except Exception as e:
        # PATTERN: Handle non-API errors (network, etc.)
        print(f"Non-API error generating contextual embedding: {e}")
        return chunk, False


def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.

    Args:
        args: Tuple containing (url, content, full_document)

    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)


def add_documents_to_supabase(
    client: QdrantClientWrapper,
    urls: List[str],
    chunk_numbers: List[int],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 100,
) -> None:
    """
    Add documents to Qdrant crawled_pages collection.
    LEGACY FUNCTION NAME: Maintained for compatibility.

    Args:
        client: Qdrant client wrapper
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
    """
    if not urls:
        return

    # Check if contextual embeddings are enabled
    use_contextual_embeddings = (
        os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
    )
    print(f"\n\nUse contextual embeddings: {use_contextual_embeddings}\n\n")

    # Get point batches from Qdrant client (this handles URL deletion)
    point_batches = list(
        client.add_documents_to_qdrant(
            urls, chunk_numbers, contents, metadatas, url_to_full_document, batch_size
        )
    )

    # Process each batch
    for batch_idx, points_batch in enumerate(point_batches):
        batch_contents = [point["content"] for point in points_batch]

        # Apply contextual embedding if enabled
        if use_contextual_embeddings:
            # Prepare arguments for parallel processing
            process_args = []
            for i, point in enumerate(points_batch):
                url = point["payload"]["url"]
                content = point["content"]
                full_document = url_to_full_document.get(url, "")
                process_args.append((url, content, full_document))

            # Process in parallel using ThreadPoolExecutor
            contextual_contents = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all tasks and collect results
                future_to_idx = {
                    executor.submit(process_chunk_with_context, arg): idx
                    for idx, arg in enumerate(process_args)
                }

                # Process results as they complete
                results = [None] * len(process_args)  # Pre-allocate to maintain order
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result, success = future.result()
                        results[idx] = result
                        if success:
                            points_batch[idx]["payload"]["contextual_embedding"] = True
                    except Exception as e:
                        print(f"Error processing chunk {idx}: {e}")
                        # Use original content as fallback
                        results[idx] = batch_contents[idx]

                contextual_contents = results
        else:
            # If not using contextual embeddings, use original contents
            contextual_contents = batch_contents

        # Create embeddings for the batch (supports both dense and sparse vectors)
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"

        if use_hybrid_search:
            batch_embeddings, batch_sparse_vectors = create_embeddings_batch(
                contextual_contents
            )
            logging.info(
                f"Created {len(batch_embeddings)} dense and {len(batch_sparse_vectors)} sparse vectors for batch"
            )
        else:
            batch_embeddings = create_embeddings_batch(contextual_contents)
            batch_sparse_vectors = None
            logging.info(f"Created {len(batch_embeddings)} dense vectors for batch")

        # Create PointStruct objects with appropriate vector configuration
        qdrant_points = []
        for i, point in enumerate(points_batch):
            if use_hybrid_search:
                # Create PointStruct with named vectors (dense + sparse)

                qdrant_points.append(
                    PointStruct(
                        id=point["id"],
                        vector={
                            "text-dense": batch_embeddings[i],
                            "text-sparse": batch_sparse_vectors[
                                i
                            ].to_qdrant_sparse_vector(),
                        },
                        payload=point["payload"],
                    )
                )
            else:
                # Create PointStruct with single vector (legacy mode)
                qdrant_points.append(
                    PointStruct(
                        id=point["id"],
                        vector=batch_embeddings[i],
                        payload=point["payload"],
                    )
                )

        # Upsert batch to Qdrant
        try:
            client.upsert_points("crawled_pages", qdrant_points)
            print(f"Successfully inserted batch {batch_idx + 1}/{len(point_batches)}")
        except Exception as e:
            print(f"Error inserting batch {batch_idx + 1}: {e}")
            # Try inserting points individually as fallback
            successful_inserts = 0
            for point in qdrant_points:
                try:
                    client.upsert_points("crawled_pages", [point])
                    successful_inserts += 1
                except Exception as individual_error:
                    print(
                        f"Failed to insert individual point {point.id}: {individual_error}"
                    )

            if successful_inserts > 0:
                print(
                    f"Successfully inserted {successful_inserts}/{len(qdrant_points)} points individually"
                )


def search_documents(
    client: QdrantClientWrapper,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Search for documents using Qdrant vector similarity.

    Args:
        client: Qdrant client wrapper
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter

    Returns:
        List of matching documents
    """
    # Create embedding for the query
    query_embedding = create_embedding(query)

    # Execute the search using Qdrant client
    try:
        results = client.search_documents(
            query_embedding=query_embedding,
            match_count=match_count,
            filter_metadata=filter_metadata,
        )
        return results
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []


def extract_code_blocks(
    markdown_content: str, min_length: int = 1000
) -> List[Dict[str, Any]]:
    """
    Extract code blocks from markdown content along with context.

    Args:
        markdown_content: The markdown content to extract code blocks from
        min_length: Minimum length of code blocks to extract (default: 1000 characters)

    Returns:
        List of dictionaries containing code blocks and their context
    """
    code_blocks = []

    # Skip if content starts with triple backticks (edge case for files wrapped in backticks)
    content = markdown_content.strip()
    start_offset = 0
    if content.startswith("```"):
        # Skip the first triple backticks
        start_offset = 3
        print("Skipping initial triple backticks")

    # Find all occurrences of triple backticks
    backtick_positions = []
    pos = start_offset
    while True:
        pos = markdown_content.find("```", pos)
        if pos == -1:
            break
        backtick_positions.append(pos)
        pos += 3

    # Process pairs of backticks
    i = 0
    while i < len(backtick_positions) - 1:
        start_pos = backtick_positions[i]
        end_pos = backtick_positions[i + 1]

        # Extract the content between backticks
        code_section = markdown_content[start_pos + 3 : end_pos]

        # Check if there's a language specifier on the first line
        lines = code_section.split("\n", 1)
        if len(lines) > 1:
            # Check if first line is a language specifier (no spaces, common language names)
            first_line = lines[0].strip()
            if first_line and " " not in first_line and len(first_line) < 20:
                language = first_line
                code_content = lines[1].strip() if len(lines) > 1 else ""
            else:
                language = ""
                code_content = code_section.strip()
        else:
            language = ""
            code_content = code_section.strip()

        # Skip if code block is too short
        if len(code_content) < min_length:
            i += 2  # Move to next pair
            continue

        # Extract context before (1000 chars)
        context_start = max(0, start_pos - 1000)
        context_before = markdown_content[context_start:start_pos].strip()

        # Extract context after (1000 chars)
        context_end = min(len(markdown_content), end_pos + 3 + 1000)
        context_after = markdown_content[end_pos + 3 : context_end].strip()

        code_blocks.append(
            {
                "code": code_content,
                "language": language,
                "context_before": context_before,
                "context_after": context_after,
                "full_context": f"{context_before}\n\n{code_content}\n\n{context_after}",
            }
        )

        # Move to next pair (skip the closing backtick we just processed)
        i += 2

    return code_blocks


def generate_code_example_summary(
    code: str, context_before: str, context_after: str
) -> str:
    """
    Generate a summary for a code example using its surrounding context.

    Uses the chat model configured via CHAT_MODEL environment variable with explicit fallback configuration.

    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code

    Returns:
        A summary of what the code example demonstrates
    """
    # Create the prompt
    prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated.
"""

    try:
        # Get adaptive client that can fallback to different provider
        client, actual_model, is_fallback = get_adaptive_chat_client()
        response = client.chat.completions.create(
            model=actual_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides concise code example summaries.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=100,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generating code example summary: {e}")
        return "Code example for demonstration purposes."


def add_code_examples_to_supabase(
    client: QdrantClientWrapper,
    urls: List[str],
    chunk_numbers: List[int],
    code_examples: List[str],
    summaries: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 100,
):
    """
    Add code examples to Qdrant code_examples collection.
    LEGACY FUNCTION NAME: Maintained for compatibility.

    Args:
        client: Qdrant client wrapper
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        code_examples: List of code example contents
        summaries: List of code example summaries
        metadatas: List of metadata dictionaries
        batch_size: Size of each batch for insertion
    """
    if not urls:
        return

    # Get point batches from Qdrant client (this handles URL deletion)
    point_batches = list(
        client.add_code_examples_to_qdrant(
            urls, chunk_numbers, code_examples, summaries, metadatas, batch_size
        )
    )

    # Process each batch
    for batch_idx, points_batch in enumerate(point_batches):
        # Create embeddings for combined text (code + summary)
        combined_texts = [point["combined_text"] for point in points_batch]

        # Support both dense and sparse vectors for hybrid search
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"

        if use_hybrid_search:
            embeddings, sparse_vectors = create_embeddings_batch(combined_texts)
            logging.info(
                f"Created {len(embeddings)} dense and {len(sparse_vectors)} sparse vectors for code examples batch"
            )
        else:
            embeddings = create_embeddings_batch(combined_texts)
            sparse_vectors = None
            logging.info(
                f"Created {len(embeddings)} dense vectors for code examples batch"
            )

        # Check if embeddings are valid (not all zeros)
        valid_embeddings = []
        valid_sparse_vectors = [] if use_hybrid_search else None

        for i, embedding in enumerate(embeddings):
            if embedding and not all(v == 0.0 for v in embedding):
                valid_embeddings.append(embedding)
                if use_hybrid_search:
                    valid_sparse_vectors.append(sparse_vectors[i])
            else:
                print(
                    "Warning: Zero or invalid embedding detected, creating new one..."
                )
                # Try to create a single embedding as fallback
                single_embedding = create_embedding(combined_texts[i])
                valid_embeddings.append(single_embedding)

                if use_hybrid_search:
                    # Create fallback sparse vector
                    try:
                        fallback_sparse = create_sparse_embedding(combined_texts[i])
                        valid_sparse_vectors.append(fallback_sparse)
                    except Exception as e:
                        logging.error(f"Failed to create fallback sparse vector: {e}")
                        valid_sparse_vectors.append(
                            SparseVectorConfig(indices=[], values=[])
                        )

        # Create PointStruct objects with appropriate vector configuration
        qdrant_points = []
        for i, point in enumerate(points_batch):
            if use_hybrid_search:
                # Create PointStruct with named vectors (dense + sparse)

                qdrant_points.append(
                    PointStruct(
                        id=point["id"],
                        vector={
                            "text-dense": valid_embeddings[i],
                            "text-sparse": valid_sparse_vectors[
                                i
                            ].to_qdrant_sparse_vector(),
                        },
                        payload=point["payload"],
                    )
                )
            else:
                # Create PointStruct with single vector (legacy mode)
                qdrant_points.append(
                    PointStruct(
                        id=point["id"],
                        vector=valid_embeddings[i],
                        payload=point["payload"],
                    )
                )

        # Upsert batch to Qdrant
        try:
            client.upsert_points("code_examples", qdrant_points)
            print(
                f"Inserted batch {batch_idx + 1} of {len(point_batches)} code examples"
            )
        except Exception as e:
            print(f"Error inserting code examples batch {batch_idx + 1}: {e}")
            # Try inserting points individually as fallback
            successful_inserts = 0
            for point in qdrant_points:
                try:
                    client.upsert_points("code_examples", [point])
                    successful_inserts += 1
                except Exception as individual_error:
                    print(
                        f"Failed to insert individual code example {point.id}: {individual_error}"
                    )

            if successful_inserts > 0:
                print(
                    f"Successfully inserted {successful_inserts}/{len(qdrant_points)} code examples individually"
                )


def update_source_info(
    client: QdrantClientWrapper, source_id: str, summary: str, word_count: int
):
    """
    Update source information using Qdrant client wrapper.

    Args:
        client: Qdrant client wrapper
        source_id: The source ID (domain)
        summary: Summary of the source
        word_count: Total word count for the source
    """
    try:
        client.update_source_info(source_id, summary, word_count)
    except Exception as e:
        print(f"Error updating source {source_id}: {e}")


def extract_source_summary(source_id: str, content: str, max_length: int = 500) -> str:
    """
    Extract a summary for a source from its content using an LLM.

    Uses the chat model configured via CHAT_MODEL environment variable with explicit fallback configuration.

    Args:
        source_id: The source ID (domain)
        content: The content to extract a summary from
        max_length: Maximum length of the summary

    Returns:
        A summary string
    """
    # Default summary if we can't extract anything meaningful
    default_summary = f"Content from {source_id}"

    if not content or len(content.strip()) == 0:
        return default_summary

    # Limit content length to avoid token limits
    truncated_content = content[:25000] if len(content) > 25000 else content

    # Create the prompt for generating the summary
    prompt = f"""<source_content>
{truncated_content}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose.
"""

    try:
        # Get adaptive client that can fallback to different provider
        client, actual_model, is_fallback = get_adaptive_chat_client()
        response = client.chat.completions.create(
            model=actual_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides concise library/tool/framework summaries.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=150,
        )

        # Extract the generated summary
        summary = response.choices[0].message.content.strip()

        # Ensure the summary is not too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."

        return summary

    except Exception as e:
        print(
            f"Error generating summary with LLM for {source_id}: {e}. Using default summary."
        )
        return default_summary


def search_code_examples(
    client: QdrantClientWrapper,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search for code examples using Qdrant vector similarity.

    Args:
        client: Qdrant client wrapper
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_id: Optional source ID to filter results

    Returns:
        List of matching code examples
    """
    # Create a more descriptive query for better embedding match
    enhanced_query = (
        f"Code example for {query}\n\nSummary: Example code showing {query}"
    )

    # Create embedding for the enhanced query
    query_embedding = create_embedding(enhanced_query)

    # Execute the search using Qdrant client
    try:
        results = client.search_code_examples(
            query_embedding=query_embedding,
            match_count=match_count,
            filter_metadata=filter_metadata,
            source_filter=source_id,
        )
        return results
    except Exception as e:
        print(f"Error searching code examples: {e}")
        return []


# Device Management and Diagnostics
# Import device management utilities
try:
    from .device_manager import (
        get_device_info as _get_device_info,
        cleanup_gpu_memory,
        get_optimal_device,
    )
except ImportError:
    from device_manager import (
        get_device_info as _get_device_info,
        cleanup_gpu_memory,
        get_optimal_device,
    )


def get_device_info() -> Dict[str, Any]:
    """
    Get comprehensive device information for diagnostics.

    Provides information about available compute devices (CPU, CUDA, MPS)
    including memory status, device capabilities, and availability.

    Returns:
        Dict with device capabilities and status information including:
        - torch_available: Whether PyTorch is available
        - cuda_available: Whether CUDA GPUs are available
        - mps_available: Whether Apple Silicon MPS is available
        - device_count: Number of available CUDA devices
        - devices: List of detailed device information
    """
    return _get_device_info()


def log_device_status() -> None:
    """
    Log comprehensive device information for debugging and monitoring.

    Logs device information including available GPUs, memory status,
    and device capabilities. Useful for troubleshooting GPU acceleration issues.
    """
    device_info = get_device_info()

    print("=== Device Status Report ===")
    print(f"PyTorch Available: {device_info['torch_available']}")
    print(f"CUDA Available: {device_info['cuda_available']}")
    print(f"MPS Available: {device_info['mps_available']}")
    print(f"CUDA Device Count: {device_info['device_count']}")

    if device_info["devices"]:
        print("Available Devices:")
        for device in device_info["devices"]:
            if "name" in device and "type" not in device:  # CUDA device
                print(f"  - CUDA {device['index']}: {device['name']}")
                print(f"    Total Memory: {device['memory_total_gb']:.2f} GB")
                print(f"    Allocated Memory: {device['memory_allocated_gb']:.2f} GB")
                print(f"    Current Device: {device['is_current']}")
            elif "type" in device:  # MPS device
                print(f"  - {device['name']} ({device['type']})")
    else:
        print("No GPU devices available")

    print("============================")


def monitor_gpu_memory() -> Optional[Dict[str, float]]:
    """
    Monitor GPU memory usage for the current device.

    Returns:
        Dict with memory information in GB, or None if CUDA not available:
        - allocated: Currently allocated memory
        - reserved: Reserved memory (includes allocated)
        - max_allocated: Peak allocated memory since last reset
        - max_reserved: Peak reserved memory since last reset
        - total: Total device memory
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        current_device = torch.cuda.current_device()
        memory_info = {
            "allocated": torch.cuda.memory_allocated(current_device) / (1024**3),
            "reserved": torch.cuda.memory_reserved(current_device) / (1024**3),
            "max_allocated": torch.cuda.max_memory_allocated(current_device)
            / (1024**3),
            "max_reserved": torch.cuda.max_memory_reserved(current_device) / (1024**3),
            "total": torch.cuda.get_device_properties(current_device).total_memory
            / (1024**3),
        }
        return memory_info
    except Exception as e:
        print(f"Error monitoring GPU memory: {e}")
        return None


def log_gpu_memory_status() -> None:
    """
    Log current GPU memory status for monitoring and debugging.

    Provides detailed memory usage information including allocated,
    reserved, and peak usage statistics.
    """
    memory_info = monitor_gpu_memory()

    if memory_info is None:
        print("GPU memory monitoring not available (CUDA not detected)")
        return

    print("=== GPU Memory Status ===")
    print(f"Allocated: {memory_info['allocated']:.2f} GB")
    print(f"Reserved: {memory_info['reserved']:.2f} GB")
    print(f"Max Allocated: {memory_info['max_allocated']:.2f} GB")
    print(f"Max Reserved: {memory_info['max_reserved']:.2f} GB")
    print(f"Total Memory: {memory_info['total']:.2f} GB")
    print(
        f"Utilization: {(memory_info['allocated'] / memory_info['total'] * 100):.1f}%"
    )
    print("=========================")


def get_optimal_compute_device(preference: str = "auto") -> str:
    """
    Get the optimal compute device for machine learning operations.

    Provides a simple interface to device selection with fallback to CPU.
    This function wraps the more comprehensive device_manager functionality.

    Args:
        preference: Device preference - "auto", "cuda", "cpu", "mps"

    Returns:
        String representation of the optimal device (e.g., "cuda:0", "cpu")
    """
    try:
        device = get_optimal_device(preference)
        return str(device)
    except Exception as e:
        print(f"Error getting optimal device: {e}. Falling back to CPU.")
        return "cpu"


def cleanup_compute_memory() -> None:
    """
    Clean up compute memory (GPU cache) to prevent memory leaks.

    Safe wrapper around GPU memory cleanup that handles cases where
    GPU is not available. Should be called after intensive compute operations.
    """
    try:
        cleanup_gpu_memory()
    except Exception as e:
        print(f"Error during memory cleanup: {e}")


def health_check_gpu_acceleration() -> Dict[str, Any]:
    """
    Comprehensive health check for GPU acceleration capabilities.

    Performs actual device testing to verify GPU acceleration is working
    correctly. Useful for monitoring and troubleshooting deployment issues.

    Returns:
        Dict with health check results including:
        - gpu_available: Whether GPU is detected and working
        - device_name: Name of the GPU device
        - memory_available: Available GPU memory
        - test_passed: Whether GPU operations test passed
        - error_message: Error details if test failed
    """
    health_status = {
        "gpu_available": False,
        "device_name": "CPU",
        "memory_available_gb": None,
        "test_passed": False,
        "error_message": None,
    }

    try:
        import torch

        if torch.cuda.is_available():
            # Test actual GPU operations
            device = torch.device("cuda:0")

            # Perform test operation
            test_tensor = torch.randn(100, 100, device=device)
            _ = test_tensor @ test_tensor.T  # Test GPU matrix operation

            # If we get here, GPU test passed
            health_status.update(
                {
                    "gpu_available": True,
                    "device_name": torch.cuda.get_device_name(device),
                    "memory_available_gb": torch.cuda.get_device_properties(
                        device
                    ).total_memory
                    / (1024**3),
                    "test_passed": True,
                }
            )

        elif (
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            # Test MPS (Apple Silicon)
            device = torch.device("mps")
            test_tensor = torch.randn(100, 100, device=device)
            _ = test_tensor.sum()  # Test MPS functionality

            health_status.update(
                {
                    "gpu_available": True,
                    "device_name": "Apple Silicon GPU (MPS)",
                    "memory_available_gb": None,  # MPS doesn't expose memory info
                    "test_passed": True,
                }
            )

    except Exception as e:
        health_status["error_message"] = str(e)

    return health_status


def health_check_reranking_model(model=None) -> Dict[str, Any]:
    """
    Comprehensive health check for reranking model functionality.

    Validates the reranking model with dummy inference to ensure it's working
    correctly. Tests model loading, device allocation, and inference capability.

    Args:
        model: Optional CrossEncoder model instance. If None, attempts to access
               from the current lifespan context.

    Returns:
        Dict with health check results including:
        - model_available: Whether reranking model is loaded
        - model_name: Name of the loaded model
        - device: Device the model is running on
        - inference_test_passed: Whether dummy inference succeeded
        - inference_latency_ms: Latency of dummy inference in milliseconds
        - error_message: Error details if test failed
    """
    health_status = {
        "model_available": False,
        "model_name": None,
        "device": None,
        "inference_test_passed": False,
        "inference_latency_ms": None,
        "error_message": None,
    }

    try:
        # Import CrossEncoder here to avoid circular imports
        from sentence_transformers import CrossEncoder
        import time

        # If no model provided, try to get from environment or return not available
        if model is None:
            if os.getenv("USE_RERANKING", "false") != "true":
                health_status["error_message"] = (
                    "Reranking not enabled (USE_RERANKING=false)"
                )
                return health_status

            # Model not provided and can't access it directly - return not available
            health_status["error_message"] = (
                "Reranking model not accessible for health check"
            )
            return health_status

        if not isinstance(model, CrossEncoder):
            health_status["error_message"] = (
                "Invalid model type - expected CrossEncoder"
            )
            return health_status

        # Model is available
        health_status["model_available"] = True

        # Get model information
        if hasattr(model, "model") and hasattr(model.model, "name_or_path"):
            health_status["model_name"] = model.model.name_or_path
        elif hasattr(model, "_model_name"):
            health_status["model_name"] = model._model_name
        else:
            health_status["model_name"] = "Unknown"

        # Get device information
        if hasattr(model, "device"):
            health_status["device"] = str(model.device)
        else:
            health_status["device"] = "Unknown"

        # Perform dummy inference test
        dummy_pairs = [
            ["health check query", "health check document"],
            ["test reranking", "sample content for validation"],
        ]

        start_time = time.time()
        scores = model.predict(dummy_pairs)
        end_time = time.time()

        # Validate inference results
        if (
            isinstance(scores, (list, tuple))
            and len(scores) == 2
            and all(isinstance(score, (int, float)) for score in scores)
        ):
            health_status["inference_test_passed"] = True
            health_status["inference_latency_ms"] = round(
                (end_time - start_time) * 1000, 2
            )

            # Clean up GPU memory after test
            cleanup_gpu_memory()
        else:
            health_status["error_message"] = (
                f"Invalid inference output: {type(scores)} with length {len(scores) if hasattr(scores, '__len__') else 'unknown'}"
            )

    except ImportError as e:
        health_status["error_message"] = f"Missing required dependencies: {e}"
    except Exception as e:
        health_status["error_message"] = f"Health check failed: {e}"

    return health_status
