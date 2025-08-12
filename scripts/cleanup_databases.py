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

try:
    from src.clients.qdrant_client import get_qdrant_client
    from knowledge_graphs.parse_repo_into_neo4j import DirectNeo4jExtractor
except ImportError as e:
    print(f"❌ Import error: {e}")
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

    async def initialize_clients(self):
        """Initialize database clients."""
        try:
            # Initialize Qdrant client
            self.qdrant_client = get_qdrant_client()
            logger.info("✅ Qdrant client initialized")

            # Initialize Neo4j client
            neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

            self.neo4j_extractor = DirectNeo4jExtractor(
                neo4j_uri, neo4j_user, neo4j_password
            )
            await self.neo4j_extractor.initialize()
            logger.info("✅ Neo4j client initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize clients: {e}")
            raise

    async def cleanup_qdrant(self) -> Dict[str, int]:
        """Clean up Qdrant vector database."""
        logger.info("🔄 Starting Qdrant cleanup...")

        stats = {"collections_deleted": 0, "documents_deleted": 0, "sources_cleared": 0}

        try:
            # Get all collections - use the internal client from wrapper
            collections = self.qdrant_client.client.get_collections()
            logger.info(f"📊 Found {len(collections.collections)} collections")

            for collection_info in collections.collections:
                collection_name = collection_info.name

                if self.dry_run:
                    logger.info(
                        f"🔍 [DRY RUN] Would delete collection: {collection_name}"
                    )
                    stats["collections_deleted"] += 1
                    continue

                # Get collection info for document count
                try:
                    info = self.qdrant_client.client.get_collection(collection_name)
                    doc_count = info.points_count
                    logger.info(
                        f"📁 Collection '{collection_name}' has {doc_count} documents"
                    )

                    # Delete collection
                    self.qdrant_client.client.delete_collection(collection_name)
                    logger.info(f"🗑️ Deleted collection: {collection_name}")

                    stats["collections_deleted"] += 1
                    stats["documents_deleted"] += doc_count

                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to delete collection {collection_name}: {e}"
                    )

            # Recreate default collections
            if not self.dry_run:
                await self._recreate_default_collections()

            logger.info(f"✅ Qdrant cleanup completed: {stats}")
            return stats

        except Exception as e:
            logger.error(f"❌ Qdrant cleanup failed: {e}")
            raise

    async def _recreate_default_collections(self):
        """Recreate default Qdrant collections."""
        logger.info("🔄 Recreating default collections...")

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
                logger.info(f"✅ Recreated collection: {collection_name}")
            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to recreate collection {collection_name}: {e}"
                )

    async def cleanup_neo4j(self) -> Dict[str, int]:
        """Clean up Neo4j knowledge graph database."""
        logger.info("🔄 Starting Neo4j cleanup...")

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
                    node_count = await node_result.single()
                    stats["nodes_deleted"] = node_count["count"] if node_count else 0

                    rel_result = await session.run(
                        "MATCH ()-[r]-() RETURN count(r) as count"
                    )
                    rel_count = await rel_result.single()
                    stats["relationships_deleted"] = (
                        rel_count["count"] if rel_count else 0
                    )

                    logger.info(
                        f"🔍 [DRY RUN] Would delete {stats['nodes_deleted']} nodes and {stats['relationships_deleted']} relationships"
                    )
                    return stats

                # Delete all relationships first
                logger.info("🗑️ Deleting all relationships...")
                rel_result = await session.run(
                    "MATCH ()-[r]-() DELETE r RETURN count(r) as deleted"
                )
                rel_record = await rel_result.single()
                if rel_record:
                    stats["relationships_deleted"] = rel_record["deleted"]
                    logger.info(
                        f"✅ Deleted {stats['relationships_deleted']} relationships"
                    )

                # Delete all nodes
                logger.info("🗑️ Deleting all nodes...")
                node_result = await session.run(
                    "MATCH (n) DELETE n RETURN count(n) as deleted"
                )
                node_record = await node_result.single()
                if node_record:
                    stats["nodes_deleted"] = node_record["deleted"]
                    logger.info(f"✅ Deleted {stats['nodes_deleted']} nodes")

                # Drop constraints
                logger.info("🗑️ Dropping constraints...")
                constraint_result = await session.run("SHOW CONSTRAINTS")
                constraints = await constraint_result.data()

                for constraint in constraints:
                    constraint_name = constraint.get("name")
                    if constraint_name:
                        try:
                            await session.run(f"DROP CONSTRAINT {constraint_name}")
                            stats["constraints_deleted"] += 1
                            logger.info(f"✅ Dropped constraint: {constraint_name}")
                        except Exception as e:
                            logger.warning(
                                f"⚠️ Failed to drop constraint {constraint_name}: {e}"
                            )

                # Drop indexes
                logger.info("🗑️ Dropping indexes...")
                try:
                    index_result = await session.run("SHOW INDEXES")
                    indexes = await index_result.data()

                    for index in indexes:
                        index_name = index.get("name")
                        if (
                            index_name and index.get("type") != "LOOKUP"
                        ):  # Don't drop lookup indexes
                            try:
                                await session.run(f"DROP INDEX {index_name}")
                                stats["indexes_deleted"] += 1
                                logger.info(f"✅ Dropped index: {index_name}")
                            except Exception as e:
                                logger.warning(
                                    f"⚠️ Failed to drop index {index_name}: {e}"
                                )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to query indexes: {e}")

            logger.info(f"✅ Neo4j cleanup completed: {stats}")
            return stats

        except Exception as e:
            logger.error(f"❌ Neo4j cleanup failed: {e}")
            raise

    async def get_database_stats(self) -> Dict[str, Dict]:
        """Get current database statistics."""
        stats = {"qdrant": {}, "neo4j": {}}

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

                # Count by node labels
                labels_result = await session.run("""
                    CALL db.labels() YIELD label
                    CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {}) 
                    YIELD value
                    RETURN label, value.count as count
                """)

                try:
                    async for record in labels_result:
                        label = record["label"]
                        count = record["count"]
                        stats["neo4j"][f"{label}_nodes"] = count
                except Exception:
                    # If APOC is not available, use basic approach
                    logger.warning("APOC not available, using basic node counting")

        except Exception as e:
            logger.error(f"❌ Failed to get database stats: {e}")

        return stats

    async def cleanup_all(self) -> Dict[str, Dict]:
        """Clean up both databases."""
        logger.info("🧹 Starting complete database cleanup...")

        results = {"qdrant": {}, "neo4j": {}}

        try:
            # Show current stats
            if not self.dry_run:
                logger.info("📊 Current database statistics:")
                current_stats = await self.get_database_stats()
                for db_name, db_stats in current_stats.items():
                    logger.info(f"  {db_name.upper()}: {db_stats}")

            # Confirm before proceeding
            if not self.confirm and not self.dry_run:
                response = input(
                    "\n⚠️ This will DELETE ALL DATA in both databases. Continue? (yes/no): "
                )
                if response.lower() not in ["yes", "y"]:
                    logger.info("❌ Operation cancelled by user")
                    return results

            # Clean Qdrant
            results["qdrant"] = await self.cleanup_qdrant()

            # Clean Neo4j
            results["neo4j"] = await self.cleanup_neo4j()

            logger.info("🎉 Complete cleanup finished successfully!")
            return results

        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            raise

    async def close(self):
        """Clean up resources."""
        if self.neo4j_extractor:
            await self.neo4j_extractor.close()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean MCP Crawl4AI RAG databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/cleanup_databases.py                    # Clean both databases (with confirmation)
    python scripts/cleanup_databases.py --confirm          # Clean both databases (skip confirmation)
    python scripts/cleanup_databases.py --qdrant-only      # Clean only Qdrant
    python scripts/cleanup_databases.py --neo4j-only       # Clean only Neo4j
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
        "--confirm", action="store_true", help="Skip confirmation prompts"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.qdrant_only and args.neo4j_only:
        logger.error("❌ Cannot specify both --qdrant-only and --neo4j-only")
        sys.exit(1)

    cleaner = DatabaseCleaner(dry_run=args.dry_run, confirm=args.confirm)

    try:
        await cleaner.initialize_clients()

        if args.qdrant_only:
            await cleaner.cleanup_qdrant()
        elif args.neo4j_only:
            await cleaner.cleanup_neo4j()
        else:
            await cleaner.cleanup_all()

    except KeyboardInterrupt:
        logger.info("❌ Operation cancelled by user")
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        sys.exit(1)
    finally:
        await cleaner.close()


if __name__ == "__main__":
    asyncio.run(main())
