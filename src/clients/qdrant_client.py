"""
Qdrant client wrapper for the Crawl4AI MCP server.

This module provides a wrapper around the Qdrant client that maintains compatibility
with the existing Supabase-based interface while using Qdrant as the vector database.

Note: This module is named 'qdrant_client' but provides a wrapper class
'QdrantClientWrapper' to avoid conflicts with the installed qdrant-client package.
"""

import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from datetime import datetime, timezone
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SparseVectorParams,
    SparseIndexParams,
    NamedVector,
    NamedSparseVector,
    SearchRequest,
    Modifier,
)
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Import dynamic dimension utilities
try:
    from ..embedding_config import get_embedding_dimensions
except ImportError:
    from embedding_config import get_embedding_dimensions

# Import sparse vector configuration (avoiding circular import)
try:
    from ..sparse_vector_types import SparseVectorConfig
except ImportError:
    SparseVectorConfig = None


def get_collections_config():
    """
    Generate collection configurations with dynamic embedding dimensions.

    Returns:
        dict: Collection configurations with current embedding dimensions
    """
    embedding_dims = get_embedding_dimensions()

    return {
        "crawled_pages": {
            "vectors_config": VectorParams(
                size=embedding_dims, distance=Distance.COSINE
            ),
            "payload_schema": {
                "url": str,
                "content": str,
                "chunk_number": int,
                "source_id": str,
                "metadata": dict,
                "created_at": str,
            },
        },
        "code_examples": {
            "vectors_config": VectorParams(
                size=embedding_dims, distance=Distance.COSINE
            ),
            "payload_schema": {
                "url": str,
                "content": str,
                "summary": str,
                "chunk_number": int,
                "source_id": str,
                "metadata": dict,
                "created_at": str,
            },
        },
        "sources": {
            "vectors_config": VectorParams(
                size=embedding_dims, distance=Distance.COSINE
            ),
            "payload_schema": {
                "source_id": str,
                "summary": str,
                "total_words": int,
                "created_at": str,
                "updated_at": str,
            },
        },
    }


def get_hybrid_collections_config():
    """
    Generate collection configurations with named vectors for hybrid search.

    Uses both dense vectors (semantic search) and sparse vectors (keyword search)
    for improved search accuracy and performance.

    Returns:
        dict: Collection configurations with named dense and sparse vectors
    """
    embedding_dims = get_embedding_dimensions()

    return {
        "crawled_pages": {
            # Named dense vectors for semantic search
            "vectors_config": {
                "text-dense": VectorParams(
                    size=embedding_dims, distance=Distance.COSINE
                )
            },
            # Named sparse vectors for keyword search
            "sparse_vectors_config": {
                "text-sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False  # REQUIRED for optimal performance
                    ),
                    modifier=Modifier.IDF,  # REQUIRED for BM25
                )
            },
            "payload_schema": {
                "url": str,
                "content": str,
                "chunk_number": int,
                "source_id": str,
                "metadata": dict,
                "created_at": str,
            },
        },
        "code_examples": {
            # Named dense vectors for semantic search
            "vectors_config": {
                "text-dense": VectorParams(
                    size=embedding_dims, distance=Distance.COSINE
                )
            },
            # Named sparse vectors for keyword search
            "sparse_vectors_config": {
                "text-sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False  # REQUIRED for optimal performance
                    ),
                    modifier=Modifier.IDF,  # REQUIRED for BM25
                )
            },
            "payload_schema": {
                "url": str,
                "content": str,
                "summary": str,
                "chunk_number": int,
                "source_id": str,
                "metadata": dict,
                "created_at": str,
            },
        },
        "sources": {
            # Named dense vectors for semantic search
            "vectors_config": {
                "text-dense": VectorParams(
                    size=embedding_dims, distance=Distance.COSINE
                )
            },
            # Named sparse vectors for keyword search
            "sparse_vectors_config": {
                "text-sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False  # REQUIRED for optimal performance
                    ),
                    modifier=Modifier.IDF,  # REQUIRED for BM25
                )
            },
            "payload_schema": {
                "source_id": str,
                "summary": str,
                "total_words": int,
                "created_at": str,
                "updated_at": str,
            },
        },
    }


def get_active_collections_config():
    """
    Get the appropriate collections configuration based on feature flags.

    Returns:
        dict: Either hybrid (named vectors) or legacy (single vector) configuration
    """
    use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"

    if use_hybrid_search:
        logging.info("Using hybrid search collections configuration with named vectors")
        return get_hybrid_collections_config()
    else:
        logging.info("Using legacy collections configuration with single vectors")
        return get_collections_config()


# Legacy global collections are now fully replaced by get_active_collections_config()
# This ensures that the configuration is always dynamically generated based on
# environment variables and feature flags, providing a single source of truth.

# Sources are now stored persistently in Qdrant 'sources' collection


class QdrantClientWrapper:
    """
    Wrapper class for Qdrant client that provides Supabase-compatible interface.
    """

    # Class-level cache for collection existence to avoid redundant checks
    _collections_verified = False

    def __init__(self, host: str = None, port: int = None):
        """
        Initialize Qdrant client wrapper.

        Args:
            host: Qdrant host (defaults to env QDRANT_HOST or localhost)
            port: Qdrant port (defaults to env QDRANT_PORT or 6333)
        """
        self.host = host or os.getenv("QDRANT_HOST") or "localhost"
        self.port = port or int(os.getenv("QDRANT_PORT") or 6333)

        # Initialize Qdrant client with retry logic
        self.client = self._create_client()

        # Set hybrid search mode based on environment variable
        self.use_hybrid_search = (
            os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"
        )

        # Ensure collections exist (only if not already verified)
        if not QdrantClientWrapper._collections_verified:
            self._ensure_collections_exist()
            QdrantClientWrapper._collections_verified = True
            logging.info("Qdrant collections verified and created if necessary")
        else:
            logging.debug("Qdrant collections already verified, skipping check")

        logging.info(f"Qdrant client initialized: {self.host}:{self.port}")

    def _create_client(self) -> QdrantClient:
        """Create Qdrant client with error handling."""
        try:
            client = QdrantClient(
                host=self.host,
                port=self.port,
                prefer_grpc=True,  # Better performance
                timeout=30,  # Longer timeout for large operations
            )
            # Test connection
            client.get_collections()
            return client
        except Exception as e:
            logging.error(
                f"Failed to connect to Qdrant at {self.host}:{self.port}: {e}"
            )
            raise ConnectionError(f"Cannot connect to Qdrant: {e}")

    def _collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        try:
            self.client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def _ensure_collections_exist(self):
        """Initialize collections, validating dimensions and creating if necessary."""
        collections_config = get_active_collections_config()

        for name, config in collections_config.items():
            if self._collection_exists(name):
                collection_info = self.client.get_collection(name)

                # Check for schema mismatch (legacy vs hybrid)
                is_config_hybrid = "sparse_vectors_config" in config
                is_qdrant_hybrid = (
                    hasattr(collection_info.config.params, "sparse_vectors")
                    and collection_info.config.params.sparse_vectors is not None
                )

                if is_config_hybrid != is_qdrant_hybrid:
                    error_message = (
                        f"Schema mismatch for collection '{name}'. "
                        f"Expected hybrid: {is_config_hybrid}, Found hybrid: {is_qdrant_hybrid}. "
                        "This can happen if you changed the USE_HYBRID_SEARCH flag. "
                        "Please delete the collections manually to have them recreated with the new schema. "
                        "You can use 'scripts/clean_qdrant.py' to do this, but be aware that THIS WILL DELETE ALL YOUR DATA."
                    )
                    logging.error(error_message)
                    raise ValueError(error_message)

                # Determine expected dimensions from config
                expected_vectors_config = config["vectors_config"]
                if isinstance(expected_vectors_config, dict):
                    expected_dims = expected_vectors_config.get("text-dense").size
                else:
                    expected_dims = expected_vectors_config.size

                # Determine current dimensions from Qdrant
                current_vectors_config = collection_info.config.params.vectors
                current_dims = -1
                if hasattr(current_vectors_config, "size"):  # Legacy single vector
                    current_dims = current_vectors_config.size
                elif isinstance(current_vectors_config, dict):  # Hybrid named vectors
                    if "text-dense" in current_vectors_config:
                        current_dims = current_vectors_config["text-dense"].size
                    elif len(current_vectors_config) > 0:
                        current_dims = list(current_vectors_config.values())[0].size

                if current_dims != -1 and current_dims != expected_dims:
                    error_message = (
                        f"Dimension mismatch for collection '{name}'. "
                        f"Expected: {expected_dims}, Found: {current_dims}. "
                        "This can happen if you changed the embedding model. "
                        "Please delete the collections manually to have them recreated with the new dimensions. "
                        "You can use 'scripts/define_qdrant_dimensions.py' to do this, but be aware that THIS WILL DELETE ALL YOUR DATA."
                    )
                    logging.error(error_message)
                    raise ValueError(error_message)
                elif current_dims == -1:
                    logging.warning(
                        f"Could not determine dimensions for collection '{name}'. Skipping validation."
                    )
                else:
                    logging.info(
                        f"Collection '{name}' dimensions are correct ({current_dims})."
                    )

            else:
                # Create new collection with appropriate configuration
                try:
                    vectors_config = config["vectors_config"]
                    sparse_vectors_config = config.get("sparse_vectors_config", None)

                    if sparse_vectors_config:
                        # Create hybrid collection with named vectors
                        logging.info(
                            f"Creating hybrid collection {name} with named vectors"
                        )
                        self.client.create_collection(
                            collection_name=name,
                            vectors_config=vectors_config,
                            sparse_vectors_config=sparse_vectors_config,
                        )
                        logging.info(f"✅ Created hybrid collection {name}")
                    else:
                        # Create legacy collection with single vector
                        logging.info(
                            f"Creating legacy collection {name} with single vector"
                        )
                        self.client.create_collection(
                            collection_name=name, vectors_config=vectors_config
                        )
                        logging.info(f"✅ Created legacy collection {name}")

                except Exception as e:
                    logging.error(f"Failed to create collection {name}: {e}")
                    raise

    def generate_point_id(self, url: str, chunk_number: int) -> str:
        """Generate consistent UUID from URL and chunk number."""
        # Create deterministic UUID from URL and chunk number
        namespace = uuid.uuid5(uuid.NAMESPACE_URL, url)
        return str(uuid.uuid5(namespace, str(chunk_number)))

    def normalize_search_results(
        self, qdrant_results: List[Any]
    ) -> List[Dict[str, Any]]:
        """Convert Qdrant results to Supabase-compatible format."""
        normalized = []
        for hit in qdrant_results:
            result = {
                "id": hit.id,
                "similarity": hit.score,
                **hit.payload,  # Include all payload fields at top level
            }
            normalized.append(result)
        return normalized

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _robust_operation(self, operation_func, *args, **kwargs):
        """Execute Qdrant operation with retry logic."""
        try:
            return operation_func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Qdrant operation failed: {e}")
            raise

    def add_documents_to_qdrant(
        self,
        urls: List[str],
        chunk_numbers: List[int],
        contents: List[str],
        metadatas: List[Dict[str, Any]],
        url_to_full_document: Dict[str, str],
        batch_size: int = 500,
    ) -> None:
        """
        Add documents to Qdrant crawled_pages collection.
        Maintains same signature as Supabase version.
        """
        if not urls:
            return

        # Get unique URLs for deletion (Qdrant equivalent of DELETE WHERE url IN ...)
        unique_urls = list(set(urls))

        # Delete existing records for these URLs
        for url in unique_urls:
            try:
                # Scroll through existing points with this URL
                existing_points = self.client.scroll(
                    collection_name="crawled_pages",
                    scroll_filter=Filter(
                        must=[FieldCondition(key="url", match=MatchValue(value=url))]
                    ),
                    limit=10000,  # Large limit to get all points
                    with_payload=False,  # Only need IDs
                )[0]

                if existing_points:
                    point_ids = [point.id for point in existing_points]
                    self.client.delete(
                        collection_name="crawled_pages", points_selector=point_ids
                    )
                    logging.info(
                        f"Deleted {len(point_ids)} existing points for URL: {url}"
                    )

            except Exception as e:
                logging.error(f"Error deleting existing records for URL {url}: {e}")

        # Process in batches
        total_points = len(urls)
        for i in range(0, total_points, batch_size):
            batch_end = min(i + batch_size, total_points)

            # Prepare batch points
            points = []
            for j in range(i, batch_end):
                # Generate consistent point ID
                point_id = self.generate_point_id(urls[j], chunk_numbers[j])

                # Extract source_id from URL
                parsed_url = urlparse(urls[j])
                source_id = parsed_url.netloc or parsed_url.path

                # Create point payload (embed all metadata)
                payload = {
                    "url": urls[j],
                    "content": contents[j],
                    "chunk_number": chunk_numbers[j],
                    "source_id": source_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **metadatas[j],  # Include all metadata fields
                }

                # Note: embedding will be handled by the calling function
                # This maintains compatibility with existing create_embeddings_batch logic
                points.append(
                    {
                        "id": point_id,
                        "payload": payload,
                        "content": contents[j],  # For embedding creation
                    }
                )

            yield points  # Yield batch for embedding creation by caller

    def upsert_points(self, collection_name: str, points: List[PointStruct]):
        """Upsert points to Qdrant collection with retry logic."""
        return self._robust_operation(
            self.client.upsert,
            collection_name=collection_name,
            points=points,
            wait=True,
        )

    def search_documents(
        self,
        query_embedding: List[float],
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search documents in Qdrant with same interface as Supabase RPC function.
        """
        # Build filter conditions
        filter_conditions = []

        if filter_metadata:
            for key, value in filter_metadata.items():
                filter_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        if source_filter:
            filter_conditions.append(
                FieldCondition(key="source", match=MatchValue(value=source_filter))
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        try:
            # Ensure query_embedding is a flat list (not nested)
            if (
                isinstance(query_embedding, list)
                and len(query_embedding) > 0
                and isinstance(query_embedding[0], list)
            ):
                query_embedding = query_embedding[0]  # Extract from nested list

            # Use named vectors for hybrid collections, regular vector for legacy
            if self.use_hybrid_search:
                results = self.client.search(
                    collection_name="crawled_pages",
                    query_vector=("text-dense", query_embedding),
                    query_filter=query_filter,
                    limit=match_count,
                    with_payload=True,
                )
            else:
                results = self.client.search(
                    collection_name="crawled_pages",
                    query_vector=query_embedding,
                    query_filter=query_filter,
                    limit=match_count,
                    with_payload=True,
                    score_threshold=0.0,  # Include all results like Supabase
                )

            return self.normalize_search_results(results)

        except Exception as e:
            logging.error(f"Error searching documents: {e}")
            return []

    def search_code_examples(
        self,
        query_embedding: List[float],
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search code examples in Qdrant with same interface as Supabase RPC function.
        """
        # Build filter conditions
        filter_conditions = []

        if filter_metadata:
            for key, value in filter_metadata.items():
                filter_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        if source_filter:
            filter_conditions.append(
                FieldCondition(key="source", match=MatchValue(value=source_filter))
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        try:
            # Ensure query_embedding is a flat list (not nested)
            if (
                isinstance(query_embedding, list)
                and len(query_embedding) > 0
                and isinstance(query_embedding[0], list)
            ):
                query_embedding = query_embedding[0]  # Extract from nested list

            # Use named vectors for hybrid collections, regular vector for legacy
            if self.use_hybrid_search:
                results = self.client.search(
                    collection_name="code_examples",
                    query_vector=("text-dense", query_embedding),
                    query_filter=query_filter,
                    limit=match_count,
                    with_payload=True,
                )
            else:
                results = self.client.search(
                    collection_name="code_examples",
                    query_vector=query_embedding,
                    query_filter=query_filter,
                    limit=match_count,
                    with_payload=True,
                    score_threshold=0.0,
                )

            return self.normalize_search_results(results)

        except Exception as e:
            logging.error(f"Error searching code examples: {e}")
            return []

    def add_code_examples_to_qdrant(
        self,
        urls: List[str],
        chunk_numbers: List[int],
        code_examples: List[str],
        summaries: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 500,
    ):
        """
        Add code examples to Qdrant with same interface as Supabase version.
        """
        if not urls:
            return

        # Delete existing records for these URLs
        unique_urls = list(set(urls))
        for url in unique_urls:
            try:
                existing_points = self.client.scroll(
                    collection_name="code_examples",
                    scroll_filter=Filter(
                        must=[FieldCondition(key="url", match=MatchValue(value=url))]
                    ),
                    limit=10000,
                    with_payload=False,
                )[0]

                if existing_points:
                    point_ids = [point.id for point in existing_points]
                    self.client.delete(
                        collection_name="code_examples", points_selector=point_ids
                    )

            except Exception as e:
                logging.error(f"Error deleting existing code examples for {url}: {e}")

        # Process in batches and yield for embedding creation
        total_items = len(urls)
        for i in range(0, total_items, batch_size):
            batch_end = min(i + batch_size, total_items)

            points = []
            for j in range(i, batch_end):
                point_id = self.generate_point_id(urls[j], chunk_numbers[j])

                # Extract source_id from URL
                parsed_url = urlparse(urls[j])
                source_id = parsed_url.netloc or parsed_url.path

                payload = {
                    "url": urls[j],
                    "content": code_examples[j],
                    "summary": summaries[j],
                    "chunk_number": chunk_numbers[j],
                    "source_id": source_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **metadatas[j],
                }

                points.append(
                    {
                        "id": point_id,
                        "payload": payload,
                        "combined_text": f"{code_examples[j]}\n\nSummary: {summaries[j]}",
                    }
                )

            yield points

    def scroll_documents(
        self, collection_name: str, scroll_filter: Filter, limit: int = 1000
    ) -> List[Any]:
        """Scroll through documents with filter (for hybrid search)."""
        try:
            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True,
            )[0]  # Get points from scroll result
            return results
        except Exception as e:
            logging.error(f"Error scrolling documents: {e}")
            return []

    def update_source_info(self, source_id: str, summary: str, word_count: int):
        """Update source information in Qdrant sources collection."""
        try:
            # Use source_id as the point ID for easy retrieval
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, source_id))
            now = datetime.now(timezone.utc).isoformat()

            # Create dummy vectors for collections (we're only using this collection for metadata)
            dummy_dense = [0.0] * get_embedding_dimensions()

            payload = {
                "source_id": source_id,
                "summary": summary,
                "total_words": word_count,
                "created_at": now,
                "updated_at": now,
            }

            if self.use_hybrid_search:
                from qdrant_client.models import SparseVector

                dummy_sparse = SparseVector(indices=[0], values=[0.0])
                point = PointStruct(
                    id=point_id,
                    vector={
                        "text-dense": dummy_dense,
                        "text-sparse": dummy_sparse,
                    },
                    payload=payload,
                )
            else:
                # Use single vector for legacy collections
                point = PointStruct(
                    id=point_id,
                    vector=dummy_dense,
                    payload=payload,
                )

            self.client.upsert(collection_name="sources", points=[point], wait=True)
            logging.info(f"Updated source info for: {source_id}")
        except Exception as e:
            logging.error(f"Error updating source info: {e}")
            import traceback

            logging.error(f"Full traceback: {traceback.format_exc()}")

    def get_available_sources(self) -> List[Dict[str, Any]]:
        """Get all available sources from Qdrant sources collection."""
        try:
            # Scroll through all points in sources collection
            result = self.client.scroll(
                collection_name="sources",
                limit=1000,  # Assuming we won't have more than 1000 sources
                with_payload=True,
            )[0]  # Get points from scroll result

            sources = []
            for point in result:
                payload = point.payload
                sources.append(
                    {
                        "source_id": payload["source_id"],
                        "summary": payload["summary"],
                        "total_words": payload["total_words"],
                        "created_at": payload["created_at"],
                        "updated_at": payload["updated_at"],
                    }
                )

            return sources
        except Exception as e:
            logging.error(f"Error getting available sources: {e}")
            return []

    def keyword_search_documents(
        self,
        query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword search on document content using Qdrant scroll functionality.

        Args:
            query: Search query string
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter
            source_filter: Optional source filter

        Returns:
            List of matching documents
        """
        try:
            # Build filter conditions
            filter_conditions = []

            if filter_metadata:
                for key, value in filter_metadata.items():
                    filter_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

            if source_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="source_id", match=MatchValue(value=source_filter)
                    )
                )

            query_filter = Filter(must=filter_conditions) if filter_conditions else None

            # Use scroll to get documents and filter by keyword client-side
            # This is necessary because Qdrant doesn't have built-in text search like PostgreSQL
            results = self.client.scroll(
                collection_name="crawled_pages",
                scroll_filter=query_filter,
                limit=match_count * 10,  # Get more to filter client-side
                with_payload=True,
            )[0]

            # Filter results by keyword match (case-insensitive)
            keyword_matches = []
            query_lower = query.lower()

            for point in results:
                content = point.payload.get("content", "").lower()
                if query_lower in content:
                    # Convert to search result format
                    result = {
                        "id": point.id,
                        "similarity": 0.5,  # Default similarity for keyword matches
                        **point.payload,
                    }
                    keyword_matches.append(result)

                    if len(keyword_matches) >= match_count:
                        break

            return keyword_matches

        except Exception as e:
            logging.error(f"Error in keyword search: {e}")
            return []

    def keyword_search_code_examples(
        self,
        query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword search on code examples using Qdrant scroll functionality.

        Args:
            query: Search query string
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter
            source_filter: Optional source filter

        Returns:
            List of matching code examples
        """
        try:
            # Build filter conditions
            filter_conditions = []

            if filter_metadata:
                for key, value in filter_metadata.items():
                    filter_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

            if source_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="source_id", match=MatchValue(value=source_filter)
                    )
                )

            query_filter = Filter(must=filter_conditions) if filter_conditions else None

            # Use scroll to get code examples and filter by keyword client-side
            results = self.client.scroll(
                collection_name="code_examples",
                scroll_filter=query_filter,
                limit=match_count * 10,  # Get more to filter client-side
                with_payload=True,
            )[0]

            # Filter results by keyword match in both content and summary (case-insensitive)
            keyword_matches = []
            query_lower = query.lower()

            for point in results:
                content = point.payload.get("content", "").lower()
                summary = point.payload.get("summary", "").lower()

                if query_lower in content or query_lower in summary:
                    # Convert to search result format
                    result = {
                        "id": point.id,
                        "similarity": 0.5,  # Default similarity for keyword matches
                        **point.payload,
                    }
                    keyword_matches.append(result)

                    if len(keyword_matches) >= match_count:
                        break

            return keyword_matches

        except Exception as e:
            logging.error(f"Error in keyword search for code examples: {e}")
            return []

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k: int = 60,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Combine dense and sparse search results using Reciprocal Rank Fusion (RRF).

        RRF Score = alpha * (1/(k + dense_rank)) + (1-alpha) * (1/(k + sparse_rank))

        Args:
            dense_results: Results from dense vector search
            sparse_results: Results from sparse vector search
            k: RRF parameter (higher k gives less weight to rank position)
            alpha: Weight for dense results (0.0-1.0, where 0.5 is equal weight)

        Returns:
            List of fused results sorted by RRF score
        """
        # Create ranking maps for efficient lookup
        dense_ranks = {
            result["id"]: idx + 1 for idx, result in enumerate(dense_results)
        }
        sparse_ranks = {
            result["id"]: idx + 1 for idx, result in enumerate(sparse_results)
        }

        # Collect all unique document IDs
        all_doc_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        # Calculate RRF scores
        fused_results = []
        result_map = {}

        # Build result map for metadata lookup
        for result in dense_results + sparse_results:
            if result["id"] not in result_map or len(result.get("content", "")) > len(
                result_map[result["id"]].get("content", "")
            ):
                result_map[result["id"]] = result

        for doc_id in all_doc_ids:
            dense_rank = dense_ranks.get(doc_id, float("inf"))
            sparse_rank = sparse_ranks.get(doc_id, float("inf"))

            # Calculate RRF score
            dense_score = 0.0 if dense_rank == float("inf") else 1.0 / (k + dense_rank)
            sparse_score = (
                0.0 if sparse_rank == float("inf") else 1.0 / (k + sparse_rank)
            )

            rrf_score = alpha * dense_score + (1 - alpha) * sparse_score

            # Get document metadata from result map
            doc_data = result_map.get(doc_id, {"id": doc_id})
            doc_data["similarity"] = rrf_score
            doc_data["dense_rank"] = dense_rank if dense_rank != float("inf") else None
            doc_data["sparse_rank"] = (
                sparse_rank if sparse_rank != float("inf") else None
            )
            doc_data["rrf_score"] = rrf_score

            fused_results.append(doc_data)

        # Sort by RRF score (descending)
        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)

        return fused_results

    def hybrid_search_documents(
        self,
        query: str,
        query_embedding: List[float] = None,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_filter: Optional[str] = None,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Perform native hybrid search combining dense and sparse vectors using Qdrant search_batch.

        Args:
            query: Search query string
            query_embedding: Dense query embedding (if None, will be created)
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter
            source_filter: Optional source filter
            rrf_k: RRF parameter for rank fusion
            dense_weight: Weight for dense results in RRF (0.0-1.0)

        Returns:
            List of search results with RRF scores
        """
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"

        if not use_hybrid_search:
            # Fallback to dense-only search for legacy collections
            logging.info("Hybrid search disabled, using dense-only search")
            if query_embedding is None:
                # Import here to avoid circular imports
                try:
                    from ..services.embedding_service import create_embedding
                except ImportError:
                    from services.embedding_service import create_embedding
                query_embedding = create_embedding(query)

            return self.search_documents(
                query_embedding, match_count, filter_metadata, source_filter
            )

        try:
            # Import sparse vector creation function
            try:
                from ..services.embedding_service import (
                    create_sparse_embedding,
                    create_embedding,
                )
            except ImportError:
                from services.embedding_service import (
                    create_sparse_embedding,
                    create_embedding,
                )

            # Create embeddings if not provided
            if query_embedding is None:
                query_embedding = create_embedding(query)

            # Ensure query_embedding is a flat list (not nested)
            if (
                isinstance(query_embedding, list)
                and len(query_embedding) > 0
                and isinstance(query_embedding[0], list)
            ):
                query_embedding = query_embedding[0]  # Extract from nested list

            query_sparse_vector = create_sparse_embedding(query)

            # Build filter conditions
            filter_conditions = []
            if filter_metadata:
                for key, value in filter_metadata.items():
                    filter_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            if source_filter:
                filter_conditions.append(
                    FieldCondition(key="source", match=MatchValue(value=source_filter))
                )

            query_filter = Filter(must=filter_conditions) if filter_conditions else None

            # Perform batch search with both dense and sparse vectors
            search_requests = [
                # Dense vector search using NamedVector
                SearchRequest(
                    vector=NamedVector(name="text-dense", vector=query_embedding),
                    filter=query_filter,
                    limit=match_count * 2,  # Get more results for better fusion
                    with_payload=True,
                    params=None,
                ),
                # Sparse vector search using NamedSparseVector
                SearchRequest(
                    vector=NamedSparseVector(
                        name="text-sparse",
                        vector=query_sparse_vector.to_qdrant_sparse_vector(),
                    ),
                    filter=query_filter,
                    limit=match_count * 2,  # Get more results for better fusion
                    with_payload=True,
                    params=None,
                ),
            ]

            # Execute batch search
            batch_results = self.client.search_batch(
                collection_name="crawled_pages", requests=search_requests
            )

            # Extract results
            dense_results = self.normalize_search_results(batch_results[0])
            sparse_results = self.normalize_search_results(batch_results[1])

            logging.info(f"Dense search found {len(dense_results)} results")
            logging.info(f"Sparse search found {len(sparse_results)} results")

            # Apply Reciprocal Rank Fusion
            fused_results = self._reciprocal_rank_fusion(
                dense_results, sparse_results, k=rrf_k, alpha=dense_weight
            )

            # Return top results
            final_results = fused_results[:match_count]
            logging.info(f"Hybrid search returning {len(final_results)} fused results")

            return final_results

        except Exception as e:
            logging.error(f"Error in hybrid search: {e}")
            # Fallback to dense-only search on error
            logging.info("Falling back to dense-only search due to error")
            if query_embedding is None:
                try:
                    from ..services.embedding_service import create_embedding
                except ImportError:
                    from services.embedding_service import create_embedding
                query_embedding = create_embedding(query)

            return self.search_documents(
                query_embedding, match_count, filter_metadata, source_filter
            )

    def hybrid_search_code_examples(
        self,
        query: str,
        query_embedding: List[float] = None,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_filter: Optional[str] = None,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Perform native hybrid search on code examples combining dense and sparse vectors.

        Args:
            query: Search query string
            query_embedding: Dense query embedding (if None, will be created)
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter
            source_filter: Optional source filter
            rrf_k: RRF parameter for rank fusion
            dense_weight: Weight for dense results in RRF (0.0-1.0)

        Returns:
            List of search results with RRF scores
        """
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"

        if not use_hybrid_search:
            # Fallback to dense-only search for legacy collections
            logging.info(
                "Hybrid search disabled, using dense-only search for code examples"
            )
            if query_embedding is None:
                try:
                    from ..services.embedding_service import create_embedding
                except ImportError:
                    from services.embedding_service import create_embedding
                query_embedding = create_embedding(query)

            return self.search_code_examples(
                query_embedding, match_count, filter_metadata, source_filter
            )

        try:
            # Import sparse vector creation function
            try:
                from ..services.embedding_service import (
                    create_sparse_embedding,
                    create_embedding,
                )
            except ImportError:
                from services.embedding_service import (
                    create_sparse_embedding,
                    create_embedding,
                )

            # Create embeddings if not provided
            if query_embedding is None:
                query_embedding = create_embedding(query)

            # Ensure query_embedding is a flat list (not nested)
            if (
                isinstance(query_embedding, list)
                and len(query_embedding) > 0
                and isinstance(query_embedding[0], list)
            ):
                query_embedding = query_embedding[0]  # Extract from nested list

            query_sparse_vector = create_sparse_embedding(query)

            # Build filter conditions
            filter_conditions = []
            if filter_metadata:
                for key, value in filter_metadata.items():
                    filter_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            if source_filter:
                filter_conditions.append(
                    FieldCondition(key="source", match=MatchValue(value=source_filter))
                )

            query_filter = Filter(must=filter_conditions) if filter_conditions else None

            # Perform batch search with both dense and sparse vectors
            search_requests = [
                # Dense vector search using NamedVector
                SearchRequest(
                    vector=NamedVector(name="text-dense", vector=query_embedding),
                    filter=query_filter,
                    limit=match_count * 2,  # Get more results for better fusion
                    with_payload=True,
                    params=None,
                ),
                # Sparse vector search using NamedSparseVector
                SearchRequest(
                    vector=NamedSparseVector(
                        name="text-sparse",
                        vector=query_sparse_vector.to_qdrant_sparse_vector(),
                    ),
                    filter=query_filter,
                    limit=match_count * 2,  # Get more results for better fusion
                    with_payload=True,
                    params=None,
                ),
            ]

            # Execute batch search
            batch_results = self.client.search_batch(
                collection_name="code_examples", requests=search_requests
            )

            # Extract results
            dense_results = self.normalize_search_results(batch_results[0])
            sparse_results = self.normalize_search_results(batch_results[1])

            logging.info(
                f"Code examples dense search found {len(dense_results)} results"
            )
            logging.info(
                f"Code examples sparse search found {len(sparse_results)} results"
            )

            # Apply Reciprocal Rank Fusion
            fused_results = self._reciprocal_rank_fusion(
                dense_results, sparse_results, k=rrf_k, alpha=dense_weight
            )

            # Return top results
            final_results = fused_results[:match_count]
            logging.info(
                f"Code examples hybrid search returning {len(final_results)} fused results"
            )

            return final_results

        except Exception as e:
            logging.error(f"Error in code examples hybrid search: {e}")
            # Fallback to dense-only search on error
            logging.info(
                "Falling back to dense-only search for code examples due to error"
            )
            if query_embedding is None:
                try:
                    from ..services.embedding_service import create_embedding
                except ImportError:
                    from services.embedding_service import create_embedding
                query_embedding = create_embedding(query)

            return self.search_code_examples(
                query_embedding, match_count, filter_metadata, source_filter
            )

    def health_check(self) -> Dict[str, Any]:
        """Check Qdrant health and return status."""
        try:
            collections = self.client.get_collections()
            collection_info = {}

            for collection in collections.collections:
                info = self.client.get_collection(collection.name)
                collection_info[collection.name] = {
                    "status": info.status,
                    "points_count": info.points_count,
                    "config": {
                        "distance": info.config.params.vectors.distance.value,
                        "size": info.config.params.vectors.size,
                    },
                }

            return {
                "status": "healthy",
                "collections": collection_info,
                "sources_count": len(self.get_available_sources()),
                "collections_verified": QdrantClientWrapper._collections_verified,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    @classmethod
    def reset_verification_cache(cls):
        """Reset the collections verification cache. Useful for testing or maintenance."""
        cls._collections_verified = False
        logging.info("Qdrant collections verification cache reset")


# Global client instance for singleton pattern
_qdrant_client_instance = None


def get_qdrant_client() -> QdrantClientWrapper:
    """
    Get a Qdrant client wrapper instance using singleton pattern.

    This prevents unnecessary reconnections and collection checks on server restarts.

    Returns:
        QdrantClientWrapper instance
    """
    global _qdrant_client_instance

    # Return existing instance if available and healthy
    if _qdrant_client_instance is not None:
        try:
            # Quick health check to ensure connection is still valid
            _qdrant_client_instance.client.get_collections()
            logging.debug("Reusing existing Qdrant client instance")
            return _qdrant_client_instance
        except Exception as e:
            logging.warning(
                f"Existing Qdrant client unhealthy, creating new instance: {e}"
            )
            _qdrant_client_instance = None

    # Create new instance if none exists or previous one is unhealthy
    try:
        _qdrant_client_instance = QdrantClientWrapper()
        logging.info("Created new Qdrant client instance")
        return _qdrant_client_instance
    except Exception as e:
        logging.error(f"Failed to create Qdrant client: {e}")
        raise
