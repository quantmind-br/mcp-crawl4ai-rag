#!/usr/bin/env python3
"""
Debug script to investigate method count duplication
"""

import asyncio
import logging
import os
from dotenv import load_dotenv
from knowledge_graphs.parse_repo_into_neo4j import DirectNeo4jExtractor

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def debug_method_count():
    """Debug method count issue by examining database directly"""

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    extractor = DirectNeo4jExtractor(neo4j_uri, neo4j_user, neo4j_password)

    try:
        await extractor.initialize()

        async with extractor.driver.session() as session:
            # Count methods by class
            result = await session.run("""
                MATCH (c:Class)-[:HAS_METHOD]->(m:Method) 
                RETURN c.full_name, COUNT(m) as method_count, collect(m.name) as method_names
                ORDER BY c.full_name
            """)

            logger.info("Methods by class:")
            total_methods = 0
            async for record in result:
                class_name = record["c.full_name"]
                method_count = record["method_count"]
                method_names = record["method_names"]
                total_methods += method_count
                logger.info(f"  {class_name}: {method_count} methods - {method_names}")

            logger.info(f"\nTotal methods found: {total_methods}")

            # Check for duplicate method IDs
            result2 = await session.run("""
                MATCH (m:Method) 
                RETURN m.method_id, COUNT(*) as count
                HAVING COUNT(*) > 1
                ORDER BY count DESC
            """)

            logger.info("\nDuplicate method IDs:")
            async for record in result2:
                method_id = record["m.method_id"]
                count = record["count"]
                logger.info(f"  {method_id}: {count} duplicates")

            # Show some example methods
            result3 = await session.run("""
                MATCH (m:Method) 
                RETURN m.method_id, m.name, m.full_name
                LIMIT 10
            """)

            logger.info("\nFirst 10 methods:")
            async for record in result3:
                method_id = record["m.method_id"]
                name = record["m.name"]
                full_name = record["m.full_name"]
                logger.info(f"  ID: {method_id} | Name: {name} | Full: {full_name}")

    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")

    finally:
        if extractor.driver:
            await extractor.close()


if __name__ == "__main__":
    asyncio.run(debug_method_count())
