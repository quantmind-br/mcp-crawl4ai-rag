"""
Embedding service for the Crawl4AI MCP application.

This service handles embedding generation, caching, and batch processing
for both dense and sparse vectors. It encapsulates the logic for creating
embeddings with Redis caching support and provides a clean interface for
the application layer.
"""

import os
import time
import logging
from typing import List, Dict, Any, Tuple, Union

import openai

# Import embedding configuration utilities
try:
    from ..embedding_config import get_embedding_dimensions
except ImportError:
    from embedding_config import get_embedding_dimensions

# Import embedding cache
try:
    from ..embedding_cache import get_embedding_cache
except ImportError:
    from embedding_cache import get_embedding_cache

# Import sparse vector configuration
try:
    from ..sparse_vector_types import SparseVectorConfig
except ImportError:
    from sparse_vector_types import SparseVectorConfig

# Import LLM API clients
try:
    from ..clients.llm_api_client import (
        get_chat_client,
        get_embeddings_client,
        get_chat_fallback_client,
    )
except ImportError:
    from clients.llm_api_client import (
        get_chat_client,
        get_embeddings_client,
        get_chat_fallback_client,
    )


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


class EmbeddingService:
    """
    Service for creating and managing embeddings with caching support.

    This service provides a high-level interface for embedding generation,
    handling both dense and sparse vectors, with Redis caching for performance
    optimization and cost reduction.
    """

    def __init__(self):
        """Initialize the embedding service."""
        self._sparse_encoder = SparseVectorEncoder()
        self._cache = get_embedding_cache()

    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for a single text using OpenAI's API.

        Args:
            text: Text to create an embedding for

        Returns:
            List of floats representing the embedding
        """
        try:
            result = self.create_embeddings_batch([text])

            # Handle both hybrid mode (tuple) and regular mode (list)
            if isinstance(result, tuple):
                # Hybrid mode: (dense_vectors, sparse_vectors)
                dense_vectors, _ = result
                return (
                    dense_vectors[0]
                    if dense_vectors
                    else [0.0] * get_embedding_dimensions()
                )
            else:
                # Regular mode: list of embeddings
                return result[0] if result else [0.0] * get_embedding_dimensions()
        except Exception as e:
            logging.error(f"Error creating embedding: {e}")
            # Return empty embedding if there's an error
            return [0.0] * get_embedding_dimensions()

    def create_sparse_embedding(self, text: str) -> SparseVectorConfig:
        """
        Create a sparse vector embedding for the given text using BM25.

        This is a convenience function that uses the sparse encoder instance.

        Args:
            text: Input text to encode

        Returns:
            SparseVectorConfig: Sparse vector configuration
        """
        return self._sparse_encoder.encode(text)

    def create_sparse_embeddings_batch(
        self, texts: List[str]
    ) -> List[SparseVectorConfig]:
        """
        Create sparse vector embeddings for multiple texts using BM25.

        Args:
            texts: List of input texts to encode

        Returns:
            List[SparseVectorConfig]: List of sparse vector configurations
        """
        return self._sparse_encoder.encode_batch(texts)

    def create_embeddings_batch(
        self, texts: List[str]
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
            use_hybrid_search = (
                os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"
            )
            if use_hybrid_search:
                return ([], [])  # (dense_vectors, sparse_vectors)
            return []

        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"
        embeddings_model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
        final_embeddings = [None] * len(texts)
        final_sparse_vectors = [None] * len(texts) if use_hybrid_search else None

        # Try cache first if available
        if self._cache:
            cached_embeddings = self._cache.get_batch(texts, embeddings_model)

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
            new_embeddings_list = self._create_embeddings_api_call(texts_to_embed)

            # Store new embeddings in cache
            if self._cache and new_embeddings_list:
                new_to_cache = {
                    text: emb for text, emb in zip(texts_to_embed, new_embeddings_list)
                }
                ttl = int(os.getenv("REDIS_EMBEDDING_TTL", "86400"))
                self._cache.set_batch(new_to_cache, embeddings_model, ttl)
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
                sparse_embeddings = self.create_sparse_embeddings_batch(texts)
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

    def _create_embeddings_api_call(self, texts: List[str]) -> List[List[float]]:
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
                embeddings_model = os.getenv(
                    "EMBEDDINGS_MODEL", "text-embedding-3-small"
                )
                client = get_embeddings_client()
                response = client.embeddings.create(
                    model=embeddings_model,
                    input=texts,
                    encoding_format="float",  # Explicitly set encoding format for DeepInfra compatibility
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if retry < max_retries - 1:
                    logging.warning(
                        f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}"
                    )
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logging.error(
                        f"Failed to create batch embeddings after {max_retries} attempts: {e}"
                    )
                    # Try creating embeddings one by one as fallback
                    logging.info("Attempting to create embeddings individually...")
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
                            logging.error(
                                f"Failed to create embedding for text {i}: {individual_error}"
                            )
                            # Add zero embedding as fallback
                            embeddings.append([0.0] * get_embedding_dimensions())

                    logging.info(
                        f"Successfully created {successful_count}/{len(texts)} embeddings individually"
                    )
                    return embeddings

    def generate_contextual_embedding(
        self, full_document: str, chunk: str
    ) -> Tuple[str, bool]:
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
            logging.warning(
                "No chat model configured. Set CHAT_MODEL environment variable."
            )
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
                logging.warning(f"API returned no choices. Model: {model_choice}")
                return chunk, False

            choice = response.choices[0]

            # Check if response was truncated due to length
            if choice.finish_reason == "length" and choice.message.content is None:
                logging.warning(
                    f"Model {model_choice} hit token limit before generating content. Trying shorter prompt."
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
                    logging.warning(
                        f"Even shorter prompt failed for {model_choice}. Using original chunk."
                    )
                    return chunk, False

            # Extract the generated context with null check
            content = choice.message.content
            if content is None:
                logging.warning(
                    f"API returned None content for contextual embedding. Model: {model_choice}"
                )
                logging.debug(f"Finish reason: {choice.finish_reason}")
                return chunk, False

            context = content.strip()
            if not context:
                logging.warning(
                    f"API returned empty content for contextual embedding. Model: {model_choice}"
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
            logging.error(
                f"Primary model {model_choice} failed ({type(primary_error).__name__}): {primary_error}"
            )
            logging.info("Attempting fallback model...")

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
                    logging.warning(
                        f"Fallback API returned no choices. Model: {fallback_model}"
                    )
                    return chunk, False

                choice = response.choices[0]

                # Check if response was truncated due to length
                if choice.finish_reason == "length" and choice.message.content is None:
                    logging.warning(
                        f"Fallback model {fallback_model} hit token limit before generating content. Trying shorter prompt."
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
                        logging.warning(
                            f"Even shorter prompt failed for fallback model {fallback_model}. Using original chunk."
                        )
                        return chunk, False

                # Extract the generated context with null check
                content = choice.message.content
                if content is None:
                    logging.warning(
                        f"Fallback API returned None content for contextual embedding. Model: {fallback_model}"
                    )
                    logging.debug(f"Finish reason: {choice.finish_reason}")
                    return chunk, False

                context = content.strip()
                if not context:
                    logging.warning(
                        f"Fallback API returned empty content for contextual embedding. Model: {fallback_model}"
                    )
                    return chunk, False

                # Combine the context with the original chunk
                contextual_text = f"{context}\n---\n{chunk}"

                # PATTERN: Process response identically
                logging.info(f"Fallback model {fallback_model} succeeded")
                return contextual_text, True

            except Exception as fallback_error:
                # PATTERN: Log both errors and gracefully degrade
                logging.error(f"Fallback model also failed: {fallback_error}")
                logging.info("Using original chunk without contextual enhancement")
                return chunk, False

        except Exception as e:
            # PATTERN: Handle non-API errors (network, etc.)
            logging.error(f"Non-API error generating contextual embedding: {e}")
            return chunk, False

    def process_chunk_with_context(self, args):
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
        return self.generate_contextual_embedding(full_document, content)


# Health check function for backward compatibility
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


# Global instance following existing codebase patterns
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """
    Get global embedding service instance (singleton pattern).

    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


# Convenience functions for backward compatibility
def create_embedding(text: str) -> List[float]:
    """Create an embedding for a single text using the global service instance."""
    return get_embedding_service().create_embedding(text)


def create_sparse_embedding(text: str) -> SparseVectorConfig:
    """Create a sparse vector embedding using the global service instance."""
    return get_embedding_service().create_sparse_embedding(text)


def create_embeddings_batch(
    texts: List[str],
) -> Union[List[List[float]], Tuple[List[List[float]], List[SparseVectorConfig]]]:
    """Create embeddings for multiple texts using the global service instance."""
    return get_embedding_service().create_embeddings_batch(texts)


def create_sparse_embeddings_batch(texts: List[str]) -> List[SparseVectorConfig]:
    """Create sparse vector embeddings for multiple texts using the global service instance."""
    return get_embedding_service().create_sparse_embeddings_batch(texts)


def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """Generate contextual information for a chunk using the global service instance."""
    return get_embedding_service().generate_contextual_embedding(full_document, chunk)


def process_chunk_with_context(args):
    """Process a single chunk with contextual embedding using the global service instance."""
    return get_embedding_service().process_chunk_with_context(args)
