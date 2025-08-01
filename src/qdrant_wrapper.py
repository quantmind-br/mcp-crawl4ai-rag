"""
Qdrant client wrapper for the Crawl4AI MCP server.

This module provides a wrapper around the Qdrant client that maintains compatibility
with the existing Supabase-based interface while using Qdrant as the vector database.

Note: This module is named 'qdrant_client' but provides a wrapper class 
'QdrantClientWrapper' to avoid conflicts with the installed qdrant-client package.
"""
import os
import hashlib
import uuid
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, 
    MatchValue, MatchText
)
from tenacity import retry, stop_after_attempt, wait_exponential

# Collection configurations based on existing Supabase schema
COLLECTIONS = {
    "crawled_pages": {
        "vectors_config": VectorParams(size=1536, distance=Distance.COSINE),
        "payload_schema": {
            "url": str,
            "content": str,
            "chunk_number": int,
            "source_id": str,
            "metadata": dict,
            "created_at": str
        }
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
            "created_at": str
        }
    }
}

# In-memory sources storage (replaces sources table)
sources_storage = {}

class QdrantClientWrapper:
    """
    Wrapper class for Qdrant client that provides Supabase-compatible interface.
    """
    
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
        
        # Ensure collections exist
        self._ensure_collections_exist()
        
        logging.info(f"Qdrant client initialized: {self.host}:{self.port}")
    
    def _create_client(self) -> QdrantClient:
        """Create Qdrant client with error handling."""
        try:
            client = QdrantClient(
                host=self.host,
                port=self.port,
                prefer_grpc=True,  # Better performance
                timeout=30,        # Longer timeout for large operations
            )
            # Test connection
            client.get_collections()
            return client
        except Exception as e:
            logging.error(f"Failed to connect to Qdrant at {self.host}:{self.port}: {e}")
            raise ConnectionError(f"Cannot connect to Qdrant: {e}")
    
    def _collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        try:
            self.client.get_collection(collection_name)
            return True
        except:
            return False
    
    def _ensure_collections_exist(self):
        """Initialize collections if they don't exist."""
        for name, config in COLLECTIONS.items():
            if not self._collection_exists(name):
                try:
                    self.client.create_collection(
                        collection_name=name,
                        vectors_config=config["vectors_config"]
                    )
                    logging.info(f"Created collection: {name}")
                except Exception as e:
                    logging.error(f"Failed to create collection {name}: {e}")
                    raise
    
    def generate_point_id(self, url: str, chunk_number: int) -> str:
        """Generate consistent UUID from URL and chunk number."""
        # Create deterministic UUID from URL and chunk number
        namespace = uuid.uuid5(uuid.NAMESPACE_URL, url)
        return str(uuid.uuid5(namespace, str(chunk_number)))
    
    def normalize_search_results(self, qdrant_results: List[Any]) -> List[Dict[str, Any]]:
        """Convert Qdrant results to Supabase-compatible format."""
        normalized = []
        for hit in qdrant_results:
            result = {
                "id": hit.id,
                "similarity": hit.score,
                **hit.payload  # Include all payload fields at top level
            }
            normalized.append(result)
        return normalized
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
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
        batch_size: int = 100
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
                    with_payload=False  # Only need IDs
                )[0]
                
                if existing_points:
                    point_ids = [point.id for point in existing_points]
                    self.client.delete(
                        collection_name="crawled_pages",
                        points_selector=point_ids
                    )
                    logging.info(f"Deleted {len(point_ids)} existing points for URL: {url}")
                    
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
                    **metadatas[j]  # Include all metadata fields
                }
                
                # Note: embedding will be handled by the calling function
                # This maintains compatibility with existing create_embeddings_batch logic
                points.append({
                    "id": point_id,
                    "payload": payload,
                    "content": contents[j]  # For embedding creation
                })
            
            yield points  # Yield batch for embedding creation by caller
    
    def upsert_points(self, collection_name: str, points: List[PointStruct]):
        """Upsert points to Qdrant collection with retry logic."""
        return self._robust_operation(
            self.client.upsert,
            collection_name=collection_name,
            points=points,
            wait=True
        )
    
    def search_documents(
        self, 
        query_embedding: List[float], 
        match_count: int = 10, 
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_filter: Optional[str] = None
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
                score_threshold=0.0  # Include all results like Supabase
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
        source_filter: Optional[str] = None
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
                score_threshold=0.0
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
        batch_size: int = 100
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
                    with_payload=False
                )[0]
                
                if existing_points:
                    point_ids = [point.id for point in existing_points]
                    self.client.delete(
                        collection_name="code_examples",
                        points_selector=point_ids
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
                    **metadatas[j]
                }
                
                points.append({
                    "id": point_id,
                    "payload": payload,
                    "combined_text": f"{code_examples[j]}\n\nSummary: {summaries[j]}"
                })
            
            yield points
    
    def scroll_documents(
        self,
        collection_name: str,
        scroll_filter: Filter,
        limit: int = 1000
    ) -> List[Any]:
        """Scroll through documents with filter (for hybrid search)."""
        try:
            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True
            )[0]  # Get points from scroll result
            return results
        except Exception as e:
            logging.error(f"Error scrolling documents: {e}")
            return []
    
    def update_source_info(self, source_id: str, summary: str, word_count: int):
        """Update source information in memory storage."""
        sources_storage[source_id] = {
            "source_id": source_id,
            "summary": summary,
            "total_word_count": word_count,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        logging.info(f"Updated source info for: {source_id}")
    
    def get_available_sources(self) -> List[Dict[str, Any]]:
        """Get all available sources from memory storage."""
        return list(sources_storage.values())
    
    def keyword_search_documents(
        self,
        query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_filter: Optional[str] = None
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
                    FieldCondition(key="source_id", match=MatchValue(value=source_filter))
                )
            
            query_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Use scroll to get documents and filter by keyword client-side
            # This is necessary because Qdrant doesn't have built-in text search like PostgreSQL
            results = self.client.scroll(
                collection_name="crawled_pages",
                scroll_filter=query_filter,
                limit=match_count * 10,  # Get more to filter client-side
                with_payload=True
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
                        **point.payload
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
        source_filter: Optional[str] = None
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
                    FieldCondition(key="source_id", match=MatchValue(value=source_filter))
                )
            
            query_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Use scroll to get code examples and filter by keyword client-side
            results = self.client.scroll(
                collection_name="code_examples",
                scroll_filter=query_filter,
                limit=match_count * 10,  # Get more to filter client-side
                with_payload=True
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
                        **point.payload
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
                        "size": info.config.params.vectors.size
                    }
                }
            
            return {
                "status": "healthy",
                "collections": collection_info,
                "sources_count": len(sources_storage)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


def get_qdrant_client() -> QdrantClientWrapper:
    """
    Get a Qdrant client wrapper instance.
    
    Returns:
        QdrantClientWrapper instance
    """
    try:
        return QdrantClientWrapper()
    except Exception as e:
        logging.error(f"Failed to create Qdrant client: {e}")
        raise