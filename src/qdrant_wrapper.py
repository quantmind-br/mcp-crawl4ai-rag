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
    SparseVectorParams,
    SparseIndexParams,  
    NamedVector,
    NamedSparseVector,
    SearchRequest,
    Modifier,
)
from tenacity import retry, stop_after_attempt, wait_exponential

# Import dynamic dimension utilities
try:
    from .embedding_config import get_embedding_dimensions
except ImportError:
    from embedding_config import get_embedding_dimensions

# Import sparse vector configuration (avoiding circular import)
try:
    from .sparse_vector_types import SparseVectorConfig
except ImportError:
    pass


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
                    modifier=Modifier.IDF  # REQUIRED for BM25
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
                    modifier=Modifier.IDF  # REQUIRED for BM25
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
                    modifier=Modifier.IDF  # REQUIRED for BM25  
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
        
        # Set hybrid search mode based on environment variable
        self.use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"

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

    def _validate_collection_schema(
        self, collection_name: str, expected_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive collection schema validation against expected configuration.
        
        Detects changes between:
        - Legacy single vector collections (VectorParams)
        - Hybrid named vector collections (dict with vectors_config + sparse_vectors_config)
        - Dimension mismatches within same schema type

        Args:
            collection_name: Name of the collection to validate
            expected_config: Expected collection configuration from get_active_collections_config()

        Returns:
            dict: Comprehensive validation results with migration recommendations
        """
        try:
            collection_info = self.client.get_collection(collection_name)
            use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"
            
            # Extract current configuration
            current_vectors = collection_info.config.params.vectors
            current_sparse_vectors = getattr(collection_info.config.params, 'sparse_vectors', None)
            
            # Initialize validation result
            validation = {
                "schema_type_match": True,
                "dimensions_match": True,  
                "sparse_config_match": True,
                "needs_migration": False,
                "migration_type": None,
                "data_loss_warning": False,
                "current_schema": "unknown",
                "expected_schema": "unknown"
            }
            
            # Determine current schema type
            if isinstance(current_vectors, dict) or current_sparse_vectors:
                validation["current_schema"] = "hybrid"
                if isinstance(current_vectors, dict):
                    validation["current_dense_size"] = current_vectors.get("text-dense", {}).size if hasattr(current_vectors.get("text-dense", {}), 'size') else "unknown"
                else:
                    # Handle legacy collections misidentified as hybrid
                    validation["current_schema"] = "legacy"
                    validation["current_dense_size"] = current_vectors.size
            else:
                validation["current_schema"] = "legacy"  
                validation["current_dense_size"] = current_vectors.size
                
            # Determine expected schema type
            expected_vectors_config = expected_config["vectors_config"]
            expected_sparse_config = expected_config.get("sparse_vectors_config", None)
            
            if isinstance(expected_vectors_config, dict) and expected_sparse_config:
                validation["expected_schema"] = "hybrid"
                validation["expected_dense_size"] = expected_vectors_config["text-dense"].size
            else:
                validation["expected_schema"] = "legacy"
                validation["expected_dense_size"] = expected_vectors_config.size if hasattr(expected_vectors_config, 'size') else expected_vectors_config
            
            # Schema type comparison
            validation["schema_type_match"] = validation["current_schema"] == validation["expected_schema"]
            
            # Dimension comparison (when schema types match)
            if validation["schema_type_match"]:
                validation["dimensions_match"] = validation["current_dense_size"] == validation["expected_dense_size"]
            else:
                validation["dimensions_match"] = False
                
            # Determine migration requirements
            if not validation["schema_type_match"]:
                validation["needs_migration"] = True
                validation["data_loss_warning"] = True
                if validation["current_schema"] == "legacy" and validation["expected_schema"] == "hybrid":
                    validation["migration_type"] = "legacy_to_hybrid"
                elif validation["current_schema"] == "hybrid" and validation["expected_schema"] == "legacy":
                    validation["migration_type"] = "hybrid_to_legacy"
            elif not validation["dimensions_match"]:
                validation["needs_migration"] = True
                validation["data_loss_warning"] = True  
                validation["migration_type"] = "dimension_change"
                
            return validation
            
        except Exception as e:
            logging.error(f"Failed to validate collection schema for {collection_name}: {e}")
            return {
                "needs_migration": True,
                "data_loss_warning": True, 
                "migration_type": "schema_unknown",
                "error": str(e)
            }

    def _create_collection_backup(self, collection_name: str) -> Dict[str, Any]:
        """
        Create a backup of collection data before migration.
        
        Args:
            collection_name: Name of collection to backup
            
        Returns:
            dict: Backup metadata with point count and backup timestamp
        """
        try:
            backup_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_name = f"{collection_name}_backup_{backup_timestamp}"
            
            # Get all points from source collection
            logging.info(f"Creating backup {backup_name} for collection {collection_name}")
            
            all_points = []
            offset = None
            batch_size = 1000
            
            # Scroll through all points in batches
            while True:
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                points, next_offset = scroll_result
                if not points:
                    break
                    
                all_points.extend(points)
                offset = next_offset
                
                if next_offset is None:
                    break
            
            # Get source collection configuration
            source_info = self.client.get_collection(collection_name)
            backup_config = source_info.config.params
            
            # Create backup collection with same configuration
            self.client.create_collection(
                collection_name=backup_name,
                vectors_config=backup_config.vectors,
                sparse_vectors_config=getattr(backup_config, 'sparse_vectors', None)
            )
            
            # Copy points to backup collection in batches  
            if all_points:
                for i in range(0, len(all_points), batch_size):
                    batch = all_points[i:i + batch_size]
                    backup_points = []
                    
                    for point in batch:
                        backup_points.append(
                            PointStruct(
                                id=point.id,
                                vector=point.vector,
                                payload=point.payload
                            )
                        )
                    
                    self.client.upsert(
                        collection_name=backup_name,
                        points=backup_points,
                        wait=True
                    )
            
            backup_info = {
                "backup_name": backup_name,
                "source_collection": collection_name,
                "point_count": len(all_points),
                "backup_timestamp": backup_timestamp,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            logging.info(f"Successfully created backup {backup_name} with {len(all_points)} points")
            return backup_info
            
        except Exception as e:
            logging.error(f"Failed to create backup for collection {collection_name}: {e}")
            raise Exception(f"Backup creation failed: {e}")

    def _restore_collection_from_backup(self, backup_name: str, target_collection: str) -> bool:
        """
        Restore a collection from backup.
        
        Args:
            backup_name: Name of the backup collection
            target_collection: Name of the target collection to restore to
            
        Returns:
            bool: True if restore successful, False otherwise
        """
        try:
            if not self._collection_exists(backup_name):
                logging.error(f"Backup collection {backup_name} does not exist")
                return False
                
            # Delete target collection if it exists
            if self._collection_exists(target_collection):
                self.client.delete_collection(target_collection)
                
            # Get backup collection configuration
            backup_info = self.client.get_collection(backup_name)
            backup_config = backup_info.config.params
            
            # Create target collection with backup configuration
            self.client.create_collection(
                collection_name=target_collection,
                vectors_config=backup_config.vectors,
                sparse_vectors_config=getattr(backup_config, 'sparse_vectors', None)
            )
            
            # Copy all points from backup to target
            all_points = []
            offset = None
            batch_size = 1000
            
            while True:
                scroll_result = self.client.scroll(
                    collection_name=backup_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                points, next_offset = scroll_result
                if not points:
                    break
                    
                all_points.extend(points)
                offset = next_offset
                
                if next_offset is None:
                    break
                    
            # Restore points in batches
            if all_points:
                for i in range(0, len(all_points), batch_size):
                    batch = all_points[i:i + batch_size]
                    restore_points = []
                    
                    for point in batch:
                        restore_points.append(
                            PointStruct(
                                id=point.id,
                                vector=point.vector,
                                payload=point.payload
                            )
                        )
                    
                    self.client.upsert(
                        collection_name=target_collection,
                        points=restore_points,
                        wait=True
                    )
            
            logging.info(f"Successfully restored {len(all_points)} points from {backup_name} to {target_collection}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to restore collection from backup {backup_name}: {e}")
            return False

    def _migrate_collection_with_confirmation(
        self, collection_name: str, expected_config: Dict[str, Any], validation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Migrate collection with user warnings and confirmation prompts.
        
        Args:
            collection_name: Name of collection to migrate
            expected_config: Expected collection configuration
            validation: Schema validation results
            
        Returns:
            dict: Migration results and status
        """
        migration_result = {
            "success": False,
            "backup_created": False,
            "backup_name": None,
            "migration_performed": False,
            "rollback_available": False,
            "error": None
        }
        
        try:
            # Display migration warnings
            migration_type = validation.get("migration_type", "unknown")
            current_schema = validation.get("current_schema", "unknown")
            expected_schema = validation.get("expected_schema", "unknown")
            
            logging.warning("=" * 80)
            logging.warning("COLLECTION MIGRATION REQUIRED")
            logging.warning("=" * 80)
            logging.warning(f"Collection: {collection_name}")
            logging.warning(f"Migration Type: {migration_type}")
            logging.warning(f"Current Schema: {current_schema}")
            logging.warning(f"Expected Schema: {expected_schema}")
            
            if validation.get("data_loss_warning", False):
                logging.warning("")
                logging.warning("⚠️  DATA LOSS WARNING ⚠️")
                logging.warning("This migration will DELETE all existing data in the collection!")
                logging.warning("A backup will be created automatically before migration.")
                logging.warning("")
            
            # Check if AUTO_MIGRATE environment variable is set
            auto_migrate = os.getenv("AUTO_MIGRATE_COLLECTIONS", "false").lower() == "true"
            
            if not auto_migrate:
                logging.warning("Migration is DISABLED by default for safety.")
                logging.warning("To enable automatic migration, set: AUTO_MIGRATE_COLLECTIONS=true")
                logging.warning("Collection will remain in current state.")
                return migration_result
                
            logging.info("AUTO_MIGRATE_COLLECTIONS=true detected, proceeding with migration...")
            
            # Create backup before migration
            logging.info(f"Creating backup before migrating {collection_name}...")
            backup_info = self._create_collection_backup(collection_name)
            migration_result["backup_created"] = True
            migration_result["backup_name"] = backup_info["backup_name"]
            migration_result["rollback_available"] = True
            
            # Perform migration based on type
            if migration_type in ["legacy_to_hybrid", "hybrid_to_legacy", "dimension_change"]:
                success = self._perform_schema_migration(collection_name, expected_config, migration_type)
                migration_result["migration_performed"] = success
                migration_result["success"] = success
                
                if success:
                    logging.info(f"✅ Successfully migrated collection {collection_name}")
                    logging.info(f"📦 Backup available at: {backup_info['backup_name']}")
                else:
                    logging.error(f"❌ Migration failed for collection {collection_name}")
                    
            else:
                raise Exception(f"Unsupported migration type: {migration_type}")
                
        except Exception as e:
            migration_result["error"] = str(e)
            logging.error(f"Migration failed for {collection_name}: {e}")
            
            # Attempt rollback if backup was created
            if migration_result["backup_created"] and migration_result["backup_name"]:
                logging.info(f"Attempting rollback from backup {migration_result['backup_name']}")
                rollback_success = self._restore_collection_from_backup(
                    migration_result["backup_name"], collection_name
                )
                if rollback_success:
                    logging.info("✅ Rollback successful, collection restored from backup")
                else:
                    logging.error("❌ Rollback failed, manual intervention required")
                    
        return migration_result

    def _perform_schema_migration(
        self, collection_name: str, expected_config: Dict[str, Any], migration_type: str
    ) -> bool:
        """
        Perform the actual schema migration.
        
        Args:
            collection_name: Name of collection to migrate
            expected_config: Expected collection configuration
            migration_type: Type of migration to perform
            
        Returns:
            bool: True if migration successful, False otherwise
        """
        try:
            # Delete existing collection
            if self._collection_exists(collection_name):
                self.client.delete_collection(collection_name)
                logging.info(f"Deleted existing collection {collection_name}")
            
            # Create new collection with expected configuration
            vectors_config = expected_config["vectors_config"]
            sparse_vectors_config = expected_config.get("sparse_vectors_config", None)
            
            if sparse_vectors_config:
                # Create hybrid collection with named vectors
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_vectors_config
                )
                logging.info(f"Created hybrid collection {collection_name} with named vectors")
            else:
                # Create legacy collection with single vector
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vectors_config
                )
                logging.info(f"Created legacy collection {collection_name} with single vector")
                
            return True
            
        except Exception as e:
            logging.error(f"Schema migration failed for {collection_name}: {e}")
            return False

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
        """Initialize collections with comprehensive schema validation and migration support."""
        collections_config = get_active_collections_config()

        for name, config in collections_config.items():
            if self._collection_exists(name):
                # Perform comprehensive schema validation
                validation = self._validate_collection_schema(name, config)
                
                logging.info(f"Schema validation for {name}: {validation}")

                if validation.get("needs_migration", False):
                    # Collection requires migration due to schema or dimension changes
                    migration_result = self._migrate_collection_with_confirmation(name, config, validation)
                    
                    if migration_result["success"]:
                        logging.info(f"✅ Collection {name} successfully migrated")
                    else:
                        if migration_result["error"]:
                            logging.error(f"❌ Migration failed for {name}: {migration_result['error']}")
                        else:
                            logging.warning(f"⚠️ Migration skipped for {name} (AUTO_MIGRATE_COLLECTIONS not enabled)")
                else:
                    logging.info(f"✅ Collection {name} schema validated successfully - no migration needed")
                    
            else:
                # Create new collection with appropriate configuration
                try:
                    vectors_config = config["vectors_config"]
                    sparse_vectors_config = config.get("sparse_vectors_config", None)
                    
                    if sparse_vectors_config:
                        # Create hybrid collection with named vectors
                        logging.info(f"Creating hybrid collection {name} with named vectors")
                        self.client.create_collection(
                            collection_name=name,
                            vectors_config=vectors_config,
                            sparse_vectors_config=sparse_vectors_config
                        )
                        logging.info(f"✅ Created hybrid collection {name}")
                    else:
                        # Create legacy collection with single vector
                        logging.info(f"Creating legacy collection {name} with single vector")
                        self.client.create_collection(
                            collection_name=name, 
                            vectors_config=vectors_config
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
            # Ensure query_embedding is a flat list (not nested)
            if isinstance(query_embedding, list) and len(query_embedding) > 0 and isinstance(query_embedding[0], list):
                query_embedding = query_embedding[0]  # Extract from nested list
            
            # Use named vectors for hybrid collections, regular vector for legacy
            if self.use_hybrid_search:
                results = self.client.search(
                    collection_name="crawled_pages",
                    query_vector=("text-dense", query_embedding),
                    query_filter=query_filter,
                    limit=match_count,
                    with_payload=True
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
                FieldCondition(key="source_id", match=MatchValue(value=source_filter))
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        try:
            # Ensure query_embedding is a flat list (not nested)
            if isinstance(query_embedding, list) and len(query_embedding) > 0 and isinstance(query_embedding[0], list):
                query_embedding = query_embedding[0]  # Extract from nested list
            
            # Use named vectors for hybrid collections, regular vector for legacy
            if self.use_hybrid_search:
                results = self.client.search(
                    collection_name="code_examples",
                    query_vector=("text-dense", query_embedding),
                    query_filter=query_filter,
                    limit=match_count,
                    with_payload=True
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

            # Create dummy vectors for collections (we're only using this collection for metadata)
            dummy_dense = [0.0] * get_embedding_dimensions()
            
            if self.use_hybrid_search:
                # For hybrid collections, we need to handle the sources collection properly
                # Check if sources collection actually uses hybrid format or if it's metadata-only
                try:
                    # Get collection info to determine actual schema
                    collection_info = self.client.get_collection("sources")
                    current_vectors = collection_info.config.params.vectors
                    has_sparse_vectors = hasattr(collection_info.config.params, 'sparse_vectors') and collection_info.config.params.sparse_vectors
                    
                    # Debug logging
                    logging.debug(f"Sources collection schema - vectors: {type(current_vectors)}, has_sparse: {has_sparse_vectors}")
                    if isinstance(current_vectors, dict):
                        logging.debug(f"Vector names: {list(current_vectors.keys())}")
                    
                    if isinstance(current_vectors, dict) and has_sparse_vectors:
                        # True hybrid collection with named vectors
                        from qdrant_client.models import SparseVector
                        
                        # Create minimal dummy sparse vector
                        dummy_sparse = SparseVector(indices=[0], values=[0.0])
                        
                        # For current qdrant-client, use the correct format for named vectors
                        point = PointStruct(
                            id=point_id,
                            vector={
                                "text-dense": dummy_dense,
                                "text-sparse": dummy_sparse
                            },
                            payload={
                                "source_id": source_id,
                                "summary": summary,
                                "total_words": word_count,
                                "created_at": now,
                                "updated_at": now,
                            },
                        )
                        logging.debug("Created hybrid point structure for sources collection")
                    else:
                        # Legacy-style sources collection even in hybrid mode
                        point = PointStruct(
                            id=point_id,
                            vector=dummy_dense,
                            payload={
                                "source_id": source_id,
                                "summary": summary,
                                "total_words": word_count,
                                "created_at": now,
                                "updated_at": now,
                            },
                        )
                        logging.debug("Created legacy point structure for sources collection")
                except Exception as schema_error:
                    logging.warning(f"Could not determine sources collection schema: {schema_error}")
                    import traceback
                    logging.debug(f"Schema error traceback: {traceback.format_exc()}")
                    # Fall back to legacy format
                    point = PointStruct(
                        id=point_id,
                        vector=dummy_dense,
                        payload={
                            "source_id": source_id,
                            "summary": summary,
                            "total_words": word_count,
                            "created_at": now,
                            "updated_at": now,
                        },
                    )
                    logging.debug("Using fallback legacy point structure")
            else:
                # Use single vector for legacy collections
                point = PointStruct(
                    id=point_id,
                    vector=dummy_dense,
                    payload={
                        "source_id": source_id,
                        "summary": summary,
                        "total_words": word_count,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
                logging.debug("Created single vector point structure for legacy mode")

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
        alpha: float = 0.5
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
        dense_ranks = {result['id']: idx + 1 for idx, result in enumerate(dense_results)}
        sparse_ranks = {result['id']: idx + 1 for idx, result in enumerate(sparse_results)}
        
        # Collect all unique document IDs
        all_doc_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        # Calculate RRF scores
        fused_results = []
        result_map = {}
        
        # Build result map for metadata lookup
        for result in dense_results + sparse_results:
            if result['id'] not in result_map or len(result.get('content', '')) > len(result_map[result['id']].get('content', '')):
                result_map[result['id']] = result
        
        for doc_id in all_doc_ids:
            dense_rank = dense_ranks.get(doc_id, float('inf'))
            sparse_rank = sparse_ranks.get(doc_id, float('inf'))
            
            # Calculate RRF score
            dense_score = 0.0 if dense_rank == float('inf') else 1.0 / (k + dense_rank)
            sparse_score = 0.0 if sparse_rank == float('inf') else 1.0 / (k + sparse_rank)
            
            rrf_score = alpha * dense_score + (1 - alpha) * sparse_score
            
            # Get document metadata from result map
            doc_data = result_map.get(doc_id, {'id': doc_id})
            doc_data['similarity'] = rrf_score
            doc_data['dense_rank'] = dense_rank if dense_rank != float('inf') else None
            doc_data['sparse_rank'] = sparse_rank if sparse_rank != float('inf') else None
            doc_data['rrf_score'] = rrf_score
            
            fused_results.append(doc_data)
        
        # Sort by RRF score (descending)
        fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        return fused_results
        
    def hybrid_search_documents(
        self,
        query: str,
        query_embedding: List[float] = None,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_filter: Optional[str] = None,
        rrf_k: int = 60,
        dense_weight: float = 0.5
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
                    from .utils import create_embedding
                except ImportError:
                    from utils import create_embedding
                query_embedding = create_embedding(query)
                
            return self.search_documents(query_embedding, match_count, filter_metadata, source_filter)
        
        try:
            # Import sparse vector creation function
            try:
                from .utils import create_sparse_embedding, create_embedding
            except ImportError:
                from utils import create_sparse_embedding, create_embedding
            
            # Create embeddings if not provided
            if query_embedding is None:
                query_embedding = create_embedding(query)
                
            # Ensure query_embedding is a flat list (not nested)
            if isinstance(query_embedding, list) and len(query_embedding) > 0 and isinstance(query_embedding[0], list):
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
                    FieldCondition(key="source_id", match=MatchValue(value=source_filter))
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
                    params=None
                ),
                # Sparse vector search using NamedSparseVector
                SearchRequest(
                    vector=NamedSparseVector(
                        name="text-sparse", 
                        vector=query_sparse_vector.to_qdrant_sparse_vector()
                    ),
                    filter=query_filter,
                    limit=match_count * 2,  # Get more results for better fusion
                    with_payload=True,
                    params=None
                )
            ]
            
            # Execute batch search
            batch_results = self.client.search_batch(
                collection_name="crawled_pages",
                requests=search_requests
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
                    from .utils import create_embedding
                except ImportError:
                    from utils import create_embedding
                query_embedding = create_embedding(query)
                
            return self.search_documents(query_embedding, match_count, filter_metadata, source_filter)
            
    def hybrid_search_code_examples(
        self,
        query: str,
        query_embedding: List[float] = None,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_filter: Optional[str] = None,
        rrf_k: int = 60,
        dense_weight: float = 0.5
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
            logging.info("Hybrid search disabled, using dense-only search for code examples")
            if query_embedding is None:
                try:
                    from .utils import create_embedding
                except ImportError:
                    from utils import create_embedding
                query_embedding = create_embedding(query)
                
            return self.search_code_examples(query_embedding, match_count, filter_metadata, source_filter)
        
        try:
            # Import sparse vector creation function
            try:
                from .utils import create_sparse_embedding, create_embedding
            except ImportError:
                from utils import create_sparse_embedding, create_embedding
            
            # Create embeddings if not provided
            if query_embedding is None:
                query_embedding = create_embedding(query)
                
            # Ensure query_embedding is a flat list (not nested)
            if isinstance(query_embedding, list) and len(query_embedding) > 0 and isinstance(query_embedding[0], list):
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
                    FieldCondition(key="source_id", match=MatchValue(value=source_filter))
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
                    params=None
                ),
                # Sparse vector search using NamedSparseVector
                SearchRequest(
                    vector=NamedSparseVector(
                        name="text-sparse", 
                        vector=query_sparse_vector.to_qdrant_sparse_vector()
                    ),
                    filter=query_filter,
                    limit=match_count * 2,  # Get more results for better fusion
                    with_payload=True,
                    params=None
                )
            ]
            
            # Execute batch search
            batch_results = self.client.search_batch(
                collection_name="code_examples",
                requests=search_requests
            )
            
            # Extract results
            dense_results = self.normalize_search_results(batch_results[0])
            sparse_results = self.normalize_search_results(batch_results[1])
            
            logging.info(f"Code examples dense search found {len(dense_results)} results")
            logging.info(f"Code examples sparse search found {len(sparse_results)} results")
            
            # Apply Reciprocal Rank Fusion
            fused_results = self._reciprocal_rank_fusion(
                dense_results, sparse_results, k=rrf_k, alpha=dense_weight
            )
            
            # Return top results
            final_results = fused_results[:match_count]
            logging.info(f"Code examples hybrid search returning {len(final_results)} fused results")
            
            return final_results
            
        except Exception as e:
            logging.error(f"Error in code examples hybrid search: {e}")
            # Fallback to dense-only search on error
            logging.info("Falling back to dense-only search for code examples due to error")
            if query_embedding is None:
                try:
                    from .utils import create_embedding
                except ImportError:
                    from utils import create_embedding
                query_embedding = create_embedding(query)
                
            return self.search_code_examples(query_embedding, match_count, filter_metadata, source_filter)

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

    def get_collection_migration_status(self) -> Dict[str, Any]:
        """
        Get migration status for all collections.
        
        Returns:
            dict: Migration status for each collection
        """
        collections_config = get_active_collections_config()
        migration_status = {}
        
        for name, config in collections_config.items():
            if self._collection_exists(name):
                validation = self._validate_collection_schema(name, config)
                migration_status[name] = {
                    "exists": True,
                    "needs_migration": validation.get("needs_migration", False),
                    "migration_type": validation.get("migration_type", None),
                    "current_schema": validation.get("current_schema", "unknown"),
                    "expected_schema": validation.get("expected_schema", "unknown"),
                    "data_loss_warning": validation.get("data_loss_warning", False)
                }
            else:
                migration_status[name] = {
                    "exists": False,
                    "needs_creation": True,
                    "expected_schema": "hybrid" if config.get("sparse_vectors_config") else "legacy"
                }
                
        return migration_status

    def migrate_collection_manually(self, collection_name: str, force: bool = False) -> Dict[str, Any]:
        """
        Manually migrate a specific collection.
        
        Args:
            collection_name: Name of collection to migrate
            force: If True, bypass AUTO_MIGRATE_COLLECTIONS check
            
        Returns:
            dict: Migration result
        """
        collections_config = get_active_collections_config()
        
        if collection_name not in collections_config:
            return {"success": False, "error": f"Collection {collection_name} not in configuration"}
            
        if not self._collection_exists(collection_name):
            return {"success": False, "error": f"Collection {collection_name} does not exist"}
            
        config = collections_config[collection_name]
        validation = self._validate_collection_schema(collection_name, config)
        
        if not validation.get("needs_migration", False):
            return {"success": True, "message": "No migration needed", "validation": validation}
            
        # Temporarily override AUTO_MIGRATE if force is True
        original_auto_migrate = os.getenv("AUTO_MIGRATE_COLLECTIONS", "false")
        if force:
            os.environ["AUTO_MIGRATE_COLLECTIONS"] = "true"
            
        try:
            migration_result = self._migrate_collection_with_confirmation(collection_name, config, validation)
            return migration_result
        finally:
            if force:
                os.environ["AUTO_MIGRATE_COLLECTIONS"] = original_auto_migrate

    def list_collection_backups(self) -> List[Dict[str, Any]]:
        """
        List all collection backups.
        
        Returns:
            list: Available backup collections with metadata
        """
        try:
            collections = self.client.get_collections()
            backups = []
            
            for collection in collections.collections:
                if "_backup_" in collection.name:
                    info = self.client.get_collection(collection.name)
                    # Parse backup name to extract metadata
                    name_parts = collection.name.split("_backup_")
                    if len(name_parts) == 2:
                        source_collection = name_parts[0]
                        timestamp = name_parts[1]
                        
                        backups.append({
                            "backup_name": collection.name,
                            "source_collection": source_collection,
                            "timestamp": timestamp,
                            "points_count": info.points_count,
                            "status": info.status
                        })
                        
            return sorted(backups, key=lambda x: x["timestamp"], reverse=True)
            
        except Exception as e:
            logging.error(f"Failed to list collection backups: {e}")
            return []

    def restore_collection_from_backup_manual(
        self, backup_name: str, target_collection: str = None
    ) -> Dict[str, Any]:
        """
        Manually restore a collection from backup.
        
        Args:
            backup_name: Name of the backup collection
            target_collection: Target collection name (defaults to source collection)
            
        Returns:
            dict: Restoration result
        """
        if not target_collection:
            # Extract source collection name from backup name
            if "_backup_" in backup_name:
                target_collection = backup_name.split("_backup_")[0]
            else:
                return {"success": False, "error": "Cannot determine target collection name"}
                
        try:
            success = self._restore_collection_from_backup(backup_name, target_collection)
            if success:
                return {
                    "success": True,
                    "message": f"Successfully restored {target_collection} from {backup_name}"
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to restore {target_collection} from {backup_name}"
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def cleanup_old_backups(self, days_old: int = 7) -> Dict[str, Any]:
        """
        Clean up old backup collections.
        
        Args:
            days_old: Delete backups older than this many days
            
        Returns:
            dict: Cleanup results
        """
        try:
            from datetime import datetime, timedelta
            
            backups = self.list_collection_backups()
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            deleted_count = 0
            errors = []
            
            for backup in backups:
                try:
                    # Parse timestamp (format: YYYYMMDD_HHMMSS)
                    timestamp_str = backup["timestamp"]
                    backup_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    if backup_date < cutoff_date:
                        self.client.delete_collection(backup["backup_name"])
                        deleted_count += 1
                        logging.info(f"Deleted old backup: {backup['backup_name']}")
                        
                except Exception as e:
                    errors.append(f"Failed to delete {backup['backup_name']}: {e}")
                    
            return {
                "success": True,
                "deleted_count": deleted_count,
                "errors": errors
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

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
        collections_config = get_active_collections_config()

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
