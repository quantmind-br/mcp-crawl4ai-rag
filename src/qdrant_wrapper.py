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

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from tenacity import retry, stop_after_attempt, wait_exponential

# Import dynamic dimension utilities
try:
    from .embedding_config import get_embedding_dimensions
except ImportError:
    from embedding_config import get_embedding_dimensions


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


# Legacy global collections - replaced by get_collections_config()
# Maintained for backward compatibility during transition
COLLECTIONS = {
    "crawled_pages": {
        "vectors_config": VectorParams(size=1536, distance=Distance.COSINE),
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
        "vectors_config": VectorParams(size=1536, distance=Distance.COSINE),
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
}

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
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))

        # Initialize Qdrant client with retry logic
        self.client = self._create_client()

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

    def _validate_collection_dimensions(
        self, collection_name: str, expected_config: VectorParams
    ) -> Dict[str, Any]:
        """
        Validate collection dimensions against expected configuration.

        Args:
            collection_name: Name of the collection to validate
            expected_config: Expected vector configuration

        Returns:
            dict: Validation results with dimensions and recreation status
        """
        try:
            collection_info = self.client.get_collection(collection_name)
            current_size = collection_info.config.params.vectors.size
            expected_size = expected_config.size

            return {
                "size_match": current_size == expected_size,
                "needs_recreation": current_size != expected_size,
                "current_size": current_size,
                "expected_size": expected_size,
            }
        except Exception as e:
            logging.debug(f"Failed to validate collection {collection_name}: {e}")
            return {"needs_recreation": True, "error": str(e)}

    def _recreate_collection_safely(
        self, collection_name: str, vectors_config: VectorParams
    ):
        """
        Safely recreate a collection with new dimensions.

        Args:
            collection_name: Name of collection to recreate
            vectors_config: New vector configuration
        """
        try:
            # Delete existing collection if it exists
            if self._collection_exists(collection_name):
                logging.warning(
                    f"Deleting collection {collection_name} due to dimension mismatch"
                )
                self.client.delete_collection(collection_name)

            # Create new collection with updated configuration
            self.client.create_collection(
                collection_name=collection_name, vectors_config=vectors_config
            )
            logging.info(
                f"Recreated collection {collection_name} with dimensions: {vectors_config.size}"
            )

        except Exception as e:
            logging.error(f"Failed to recreate collection {collection_name}: {e}")
            raise

    def _ensure_collections_exist(self):
        """Initialize collections with dimension validation and controlled recreation."""
        collections_config = get_collections_config()

        for name, config in collections_config.items():
            vectors_config = config["vectors_config"]

            if self._collection_exists(name):
                # Validate existing collection dimensions
                validation = self._validate_collection_dimensions(name, vectors_config)

                logging.info(f"Validation result for {name}: {validation}")

                if validation.get("needs_recreation", False):
                    current_size = validation.get("current_size", "unknown")
                    expected_size = validation.get("expected_size", vectors_config.size)

                    # CONSERVATIVE APPROACH: Log warning but DO NOT auto-recreate
                    logging.warning(
                        f"Collection {name} has dimension mismatch "
                        f"(current: {current_size}, expected: {expected_size}). "
                        f"AUTO-RECREATION DISABLED. Use reset_verification_cache() to force recreation if needed."
                    )
                    # Continue using existing collection instead of recreating
                else:
                    logging.info(
                        f"Collection {name} dimensions validated successfully - no recreation needed"
                    )
            else:
                # Create new collection only if it doesn't exist
                try:
                    logging.info(
                        f"Collection {name} does not exist, creating with dimensions: {vectors_config.size}"
                    )
                    self.client.create_collection(
                        collection_name=name, vectors_config=vectors_config
                    )
                    logging.info(
                        f"Created collection {name} with dimensions: {vectors_config.size}"
                    )
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
        batch_size: int = 100,
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
                FieldCondition(key="source_id", match=MatchValue(value=source_filter))
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        try:
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
                FieldCondition(key="source_id", match=MatchValue(value=source_filter))
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        try:
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
        batch_size: int = 100,
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

            # Create a dummy vector (we're only using this collection for metadata)
            dummy_vector = [0.0] * get_embedding_dimensions()

            point = PointStruct(
                id=point_id,
                vector=dummy_vector,
                payload={
                    "source_id": source_id,
                    "summary": summary,
                    "total_words": word_count,
                    "created_at": now,
                    "updated_at": now,
                },
            )

            self.client.upsert(collection_name="sources", points=[point], wait=True)
            logging.info(f"Updated source info for: {source_id}")
        except Exception as e:
            logging.error(f"Error updating source info: {e}")

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

    @classmethod
    def force_recreate_collections(cls):
        """
        Force recreation of all collections with current configuration.

        WARNING: This will delete all existing data!
        Use only when you need to update collection dimensions.
        """
        logging.warning(
            "FORCE RECREATION: This will delete all existing collection data!"
        )

        # Reset verification cache first
        cls.reset_verification_cache()

        # Create temporary client for recreation
        temp_client = cls()
        collections_config = get_collections_config()

        for name, config in collections_config.items():
            vectors_config = config["vectors_config"]

            if temp_client._collection_exists(name):
                logging.warning(
                    f"Force recreating collection {name} with dimensions: {vectors_config.size}"
                )
                temp_client._recreate_collection_safely(name, vectors_config)
            else:
                logging.info(
                    f"Creating new collection {name} with dimensions: {vectors_config.size}"
                )
                temp_client.client.create_collection(
                    collection_name=name, vectors_config=vectors_config
                )

        logging.info(
            "Force recreation completed. All collections now match current configuration."
        )


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
