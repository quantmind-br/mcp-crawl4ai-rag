#!/usr/bin/env python3
"""
Database cleanup utility for MCP Crawl4AI RAG system.

This script provides comprehensive database cleanup functionality for both
Qdrant (vector database) and Neo4j (knowledge graph) databases.

Usage:
    python scripts/cleanup_databases.py [options]

Options:
    --qdrant-only    Clean only Qdrant database
    --neo4j-only     Clean only Neo4j database
    --confirm        Skip confirmation prompts
    --dry-run        Show what would be deleted without actually deleting
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logging.getLogger(__name__).info(f"Loaded environment from {env_path}")
    else:
        logging.getLogger(__name__).warning(f"No .env file found at {env_path}")
except ImportError:
    logging.getLogger(__name__).warning("python-dotenv not available, relying on system environment")

try:
    from src.clients.qdrant_client import get_qdrant_client
    from src.k_graph.services.repository_parser import DirectNeo4jExtractor
    import redis
except ImportError as e:
    print(f"ERROR: Import error: {e}")
    print(
        "Make sure you're running from the project root and dependencies are installed"
    )
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseCleaner:
    """Comprehensive database cleanup utility."""

    def __init__(self, dry_run: bool = False, confirm: bool = False):
        self.dry_run = dry_run
        self.confirm = confirm
        self.qdrant_client = None
        self.neo4j_extractor = None
        self.redis_client = None

    async def initialize_clients(self, qdrant_only=False, neo4j_only=False, redis_only=False):
        """Initialize database clients."""
        try:
            # Initialize Qdrant client if needed
            if not neo4j_only and not redis_only:
                self.qdrant_client = get_qdrant_client()
                logger.info("SUCCESS: Qdrant client initialized")

            # Initialize Neo4j client if needed
            if not qdrant_only and not redis_only:
                neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
                neo4j_user = os.getenv("NEO4J_USER", "neo4j")
                neo4j_password = os.getenv("NEO4J_PASSWORD", "password123")
                
                if not neo4j_password:
                    raise ValueError(
                        "Neo4j password not found in environment variables. "
                        "Please create a .env file from .env.example and set NEO4J_PASSWORD"
                    )

                self.neo4j_extractor = DirectNeo4jExtractor(
                    neo4j_uri, neo4j_user, neo4j_password
                )
                await self.neo4j_extractor.initialize()
                logger.info("SUCCESS: Neo4j client initialized")

            # Initialize Redis client if needed
            if not qdrant_only and not neo4j_only or redis_only:
                redis_host = os.getenv("REDIS_HOST", "localhost")
                redis_port = int(os.getenv("REDIS_PORT", "6379"))
                redis_db = int(os.getenv("REDIS_DB", "0"))
                redis_password = os.getenv("REDIS_PASSWORD")
                redis_username = os.getenv("REDIS_USERNAME")
                
                # Create Redis connection parameters
                redis_params = {
                    "host": redis_host,
                    "port": redis_port,
                    "db": redis_db,
                    "decode_responses": True,
                    "socket_timeout": 5,
                    "socket_connect_timeout": 5,
                }
                
                if redis_password:
                    redis_params["password"] = redis_password
                if redis_username:
                    redis_params["username"] = redis_username
                
                self.redis_client = redis.Redis(**redis_params)
                # Test connection
                self.redis_client.ping()
                logger.info("SUCCESS: Redis client initialized")

        except Exception as e:
            logger.error(f"ERROR: Failed to initialize clients: {e}")
            raise

    async def cleanup_qdrant(self) -> Dict[str, int]:
        """Clean up Qdrant vector database."""
        logger.info("PROCESSING: Starting Qdrant cleanup...")

        stats = {"collections_deleted": 0, "documents_deleted": 0, "sources_cleared": 0}

        try:
            # Get all collections - use the internal client from wrapper
            collections = self.qdrant_client.client.get_collections()
            logger.info(f"STATS: Found {len(collections.collections)} collections")

            for collection_info in collections.collections:
                collection_name = collection_info.name

                if self.dry_run:
                    logger.info(
                        f"DRY RUN: [DRY RUN] Would delete collection: {collection_name}"
                    )
                    stats["collections_deleted"] += 1
                    continue

                # Get collection info for document count
                try:
                    info = self.qdrant_client.client.get_collection(collection_name)
                    doc_count = info.points_count
                    logger.info(
                        f"COLLECTION: Collection '{collection_name}' has {doc_count} documents"
                    )

                    # Delete collection
                    self.qdrant_client.client.delete_collection(collection_name)
                    logger.info(f"DELETING: Deleted collection: {collection_name}")

                    stats["collections_deleted"] += 1
                    stats["documents_deleted"] += doc_count

                except Exception as e:
                    logger.warning(
                        f"WARNING: Failed to delete collection {collection_name}: {e}"
                    )

            # Recreate default collections
            if not self.dry_run:
                await self._recreate_default_collections()

            logger.info(f"SUCCESS: Qdrant cleanup completed: {stats}")
            return stats

        except Exception as e:
            logger.error(f"ERROR: Qdrant cleanup failed: {e}")
            raise

    async def _recreate_default_collections(self):
        """Recreate default Qdrant collections."""
        logger.info("PROCESSING: Recreating default collections...")

        # Import the system's collection configuration function
        try:
            from src.clients.qdrant_client import get_active_collections_config

            collections_config = get_active_collections_config()
        except ImportError:
            # Fallback configuration if import fails
            from qdrant_client.models import VectorParams, Distance

            collections_config = {
                "crawled_pages": {
                    "vectors_config": VectorParams(
                        size=1024,  # Current system uses 1024
                        distance=Distance.COSINE,
                    )
                },
                "code_examples": {
                    "vectors_config": VectorParams(size=1024, distance=Distance.COSINE)
                },
                "sources": {
                    "vectors_config": VectorParams(size=1024, distance=Distance.COSINE)
                },
            }

        for collection_name, config in collections_config.items():
            try:
                self.qdrant_client.client.create_collection(
                    collection_name=collection_name, **config
                )
                logger.info(f"SUCCESS: Recreated collection: {collection_name}")
            except Exception as e:
                logger.warning(
                    f"WARNING: Failed to recreate collection {collection_name}: {e}"
                )

    async def cleanup_neo4j(self) -> Dict[str, int]:
        """Clean up Neo4j knowledge graph database."""
        logger.info("PROCESSING: Starting Neo4j cleanup...")

        stats = {
            "nodes_deleted": 0,
            "relationships_deleted": 0,
            "constraints_deleted": 0,
            "indexes_deleted": 0,
        }

        try:
            async with self.neo4j_extractor.driver.session() as session:
                if self.dry_run:
                    # Count nodes and relationships for dry run
                    node_result = await session.run(
                        "MATCH (n) RETURN count(n) as count"
                    )
                    node_record = await node_result.single()
                    stats["nodes_deleted"] = node_record["count"] if node_record else 0

                    rel_result = await session.run(
                        "MATCH ()-[r]-() RETURN count(r) as count"
                    )
                    rel_record = await rel_result.single()
                    stats["relationships_deleted"] = (
                        rel_record["count"] if rel_record else 0
                    )

                    logger.info(
                        f"DRY RUN: [DRY RUN] Would delete {stats['nodes_deleted']} nodes and {stats['relationships_deleted']} relationships"
                    )
                    return stats

                # Delete all relationships first
                logger.info("DELETING: Deleting all relationships...")
                rel_result = await session.run(
                    "MATCH ()-[r]-() DELETE r"
                )
                # Neo4j DELETE doesn't return count, so we need to count separately
                count_result = await session.run("MATCH ()-[r]-() RETURN count(r) as count")
                count_record = await count_result.single()
                initial_rel_count = count_record["count"] if count_record else 0
                
                # Actually delete relationships
                await session.run("MATCH ()-[r]-() DELETE r")
                stats["relationships_deleted"] = initial_rel_count
                logger.info(
                    f"SUCCESS: Deleted {stats['relationships_deleted']} relationships"
                )

                # Delete all nodes
                logger.info("DELETING: Deleting all nodes...")
                # Count nodes first
                count_result = await session.run("MATCH (n) RETURN count(n) as count")
                count_record = await count_result.single()
                initial_node_count = count_record["count"] if count_record else 0
                
                # Actually delete nodes
                await session.run("MATCH (n) DELETE n")
                stats["nodes_deleted"] = initial_node_count
                logger.info(f"SUCCESS: Deleted {stats['nodes_deleted']} nodes")

                # Drop constraints
                logger.info("DELETING: Dropping constraints...")
                try:
                    constraint_result = await session.run("SHOW CONSTRAINTS")
                    constraints = []
                    async for record in constraint_result:
                        constraints.append(dict(record))

                    for constraint in constraints:
                        constraint_name = constraint.get("name")
                        if constraint_name:
                            try:
                                await session.run(f"DROP CONSTRAINT {constraint_name}")
                                stats["constraints_deleted"] += 1
                                logger.info(f"SUCCESS: Dropped constraint: {constraint_name}")
                            except Exception as e:
                                logger.warning(
                                    f"WARNING: Failed to drop constraint {constraint_name}: {e}"
                                )
                except Exception as e:
                    logger.warning(f"WARNING: Failed to query constraints: {e}")

                # Drop indexes
                logger.info("DELETING: Dropping indexes...")
                try:
                    index_result = await session.run("SHOW INDEXES")
                    indexes = []
                    async for record in index_result:
                        indexes.append(dict(record))

                    for index in indexes:
                        index_name = index.get("name")
                        if (
                            index_name and index.get("type") != "LOOKUP"
                        ):  # Don't drop lookup indexes
                            try:
                                await session.run(f"DROP INDEX {index_name}")
                                stats["indexes_deleted"] += 1
                                logger.info(f"SUCCESS: Dropped index: {index_name}")
                            except Exception as e:
                                logger.warning(
                                    f"WARNING: Failed to drop index {index_name}: {e}"
                                )
                except Exception as e:
                    logger.warning(f"WARNING: Failed to query indexes: {e}")

            logger.info(f"SUCCESS: Neo4j cleanup completed: {stats}")
            return stats

        except Exception as e:
            logger.error(f"ERROR: Neo4j cleanup failed: {e}")
            raise

    async def cleanup_redis(self) -> Dict[str, int]:
        """Clean up Redis cache database."""
        logger.info("PROCESSING: Starting Redis cleanup...")

        stats = {
            "keys_deleted": 0,
            "memory_freed": 0,
            "databases_cleared": 0,
        }

        try:
            if self.dry_run:
                # Count keys for dry run
                key_count = self.redis_client.dbsize()
                stats["keys_deleted"] = key_count
                
                # Get memory usage
                info = self.redis_client.info("memory")
                stats["memory_freed"] = info.get("used_memory", 0)
                
                logger.info(
                    f"DRY RUN: [DRY RUN] Would delete {stats['keys_deleted']} keys and free {stats['memory_freed']} bytes"
                )
                return stats

            # Get initial stats
            initial_key_count = self.redis_client.dbsize()
            initial_memory_info = self.redis_client.info("memory")
            initial_memory = initial_memory_info.get("used_memory", 0)

            # Clear all keys in the current database
            logger.info(f"DELETING: Clearing database {self.redis_client.connection_pool.connection_kwargs['db']}...")
            self.redis_client.flushdb()
            
            # Get final stats
            final_memory_info = self.redis_client.info("memory")
            final_memory = final_memory_info.get("used_memory", 0)
            
            stats["keys_deleted"] = initial_key_count
            stats["memory_freed"] = initial_memory - final_memory
            stats["databases_cleared"] = 1

            logger.info(
                f"SUCCESS: Deleted {stats['keys_deleted']} keys, freed {stats['memory_freed']} bytes"
            )

            logger.info(f"SUCCESS: Redis cleanup completed: {stats}")
            return stats

        except Exception as e:
            logger.error(f"ERROR: Redis cleanup failed: {e}")
            raise

    async def get_database_stats(self) -> Dict[str, Dict]:
        """Get current database statistics."""
        stats = {"qdrant": {}, "neo4j": {}, "redis": {}}

        try:
            # Qdrant stats
            collections = self.qdrant_client.client.get_collections()
            stats["qdrant"]["collections"] = len(collections.collections)
            stats["qdrant"]["total_documents"] = 0

            for collection_info in collections.collections:
                try:
                    info = self.qdrant_client.client.get_collection(
                        collection_info.name
                    )
                    stats["qdrant"]["total_documents"] += info.points_count
                    stats["qdrant"][f"{collection_info.name}_documents"] = (
                        info.points_count
                    )
                except Exception:
                    pass

            # Neo4j stats
            async with self.neo4j_extractor.driver.session() as session:
                # Count nodes
                node_result = await session.run("MATCH (n) RETURN count(n) as count")
                node_record = await node_result.single()
                stats["neo4j"]["nodes"] = node_record["count"] if node_record else 0

                # Count relationships
                rel_result = await session.run(
                    "MATCH ()-[r]-() RETURN count(r) as count"
                )
                rel_record = await rel_result.single()
                stats["neo4j"]["relationships"] = (
                    rel_record["count"] if rel_record else 0
                )

                # Count by node labels - use simple approach without APOC
                try:
                    labels_result = await session.run("CALL db.labels() YIELD label RETURN label")
                    labels = []
                    async for record in labels_result:
                        labels.append(record["label"])
                    
                    # Count each label type separately
                    for label in labels:
                        try:
                            count_result = await session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                            count_record = await count_result.single()
                            if count_record:
                                stats["neo4j"][f"{label}_nodes"] = count_record["count"]
                        except Exception as e:
                            logger.warning(f"Failed to count {label} nodes: {e}")
                            
                except Exception as e:
                    # If db.labels() is not available, skip detailed counting
                    logger.warning(f"Could not get node labels: {e}")

            # Redis stats
            if self.redis_client:
                try:
                    # Get basic Redis info
                    redis_info = self.redis_client.info()
                    stats["redis"]["keys"] = self.redis_client.dbsize()
                    stats["redis"]["used_memory"] = redis_info.get("used_memory", 0)
                    stats["redis"]["used_memory_human"] = redis_info.get("used_memory_human", "0B")
                    stats["redis"]["connected_clients"] = redis_info.get("connected_clients", 0)
                    stats["redis"]["uptime_seconds"] = redis_info.get("uptime_in_seconds", 0)
                    stats["redis"]["redis_version"] = redis_info.get("redis_version", "unknown")
                    stats["redis"]["database"] = self.redis_client.connection_pool.connection_kwargs.get('db', 0)
                except Exception as e:
                    logger.warning(f"Could not get Redis stats: {e}")
                    stats["redis"]["error"] = str(e)

        except Exception as e:
            logger.error(f"ERROR: Failed to get database stats: {e}")

        return stats

    async def cleanup_all(self) -> Dict[str, Dict]:
        """Clean up all databases."""
        logger.info("CLEANUP: Starting complete database cleanup...")

        results = {"qdrant": {}, "neo4j": {}, "redis": {}}

        try:
            # Show current stats
            if not self.dry_run:
                logger.info("STATS: Current database statistics:")
                current_stats = await self.get_database_stats()
                for db_name, db_stats in current_stats.items():
                    logger.info(f"  {db_name.upper()}: {db_stats}")

            # Confirm before proceeding (only if not in confirm mode and not dry run)
            if not self.confirm and not self.dry_run:
                try:
                    # Use a more robust input method for Windows compatibility
                    import msvcrt
                    
                    print("\nWARNING: This will DELETE ALL DATA in both databases. Continue? (y/n): ", end="", flush=True)
                    
                    while True:
                        if msvcrt.kbhit():
                            char = msvcrt.getch().decode('utf-8').lower()
                            print(char)  # Echo the character
                            
                            if char == 'y':
                                print("SUCCESS: Proceeding with cleanup...")
                                break
                            elif char == 'n':
                                print("ERROR: Operation cancelled by user")
                                return results
                            else:
                                print("Please enter 'y' for yes or 'n' for no: ", end="", flush=True)
                        
                        await asyncio.sleep(0.1)
                        
                except ImportError:
                    # Fallback for non-Windows or if msvcrt not available
                    try:
                        response = input(
                            "\nWARNING: This will DELETE ALL DATA in both databases. Continue? (yes/no): "
                        )
                        if response.lower() not in ["yes", "y"]:
                            logger.info("ERROR: Operation cancelled by user")
                            return results
                    except EOFError:
                        # Handle cases where input is not available (automated scripts)
                        logger.warning("WARNING: No input available, proceeding with cleanup (use --dry-run to test first)")

            # Clean Qdrant
            if self.qdrant_client:
                results["qdrant"] = await self.cleanup_qdrant()

            # Clean Neo4j
            if self.neo4j_extractor:
                results["neo4j"] = await self.cleanup_neo4j()

            # Clean Redis
            if self.redis_client:
                results["redis"] = await self.cleanup_redis()

            logger.info("COMPLETE: Complete cleanup finished successfully!")
            return results

        except Exception as e:
            logger.error(f"ERROR: Cleanup failed: {e}")
            raise

    async def close(self):
        """Clean up resources."""
        if self.neo4j_extractor:
            await self.neo4j_extractor.close()
        if self.redis_client:
            self.redis_client.close()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean MCP Crawl4AI RAG databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/cleanup_databases.py                    # Clean all databases (with confirmation)
    python scripts/cleanup_databases.py --confirm          # Clean all databases (skip confirmation)
    python scripts/cleanup_databases.py --qdrant-only      # Clean only Qdrant
    python scripts/cleanup_databases.py --neo4j-only       # Clean only Neo4j
    python scripts/cleanup_databases.py --redis-only       # Clean only Redis
    python scripts/cleanup_databases.py --dry-run          # Show what would be deleted
        """,
    )

    parser.add_argument(
        "--qdrant-only", action="store_true", help="Clean only Qdrant database"
    )
    parser.add_argument(
        "--neo4j-only", action="store_true", help="Clean only Neo4j database"
    )
    parser.add_argument(
        "--redis-only", action="store_true", help="Clean only Redis database"
    )
    parser.add_argument(
        "--confirm", action="store_true", help="Skip confirmation prompts"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )

    args = parser.parse_args()

    # Validate arguments
    exclusive_options = [args.qdrant_only, args.neo4j_only, args.redis_only]
    if sum(exclusive_options) > 1:
        logger.error("ERROR: Cannot specify multiple database-only options")
        sys.exit(1)

    cleaner = DatabaseCleaner(dry_run=args.dry_run, confirm=args.confirm)

    try:
        if args.qdrant_only:
            await cleaner.initialize_clients(qdrant_only=True)
            await cleaner.cleanup_qdrant()
        elif args.neo4j_only:
            await cleaner.initialize_clients(neo4j_only=True)
            await cleaner.cleanup_neo4j()
        elif args.redis_only:
            await cleaner.initialize_clients(redis_only=True)
            await cleaner.cleanup_redis()
        else:
            await cleaner.initialize_clients()
            await cleaner.cleanup_all()

    except KeyboardInterrupt:
        logger.info("ERROR: Operation cancelled by user")
    except Exception as e:
        logger.error(f"ERROR: Cleanup failed: {e}")
        sys.exit(1)
    finally:
        await cleaner.close()


if __name__ == "__main__":
    asyncio.run(main())
