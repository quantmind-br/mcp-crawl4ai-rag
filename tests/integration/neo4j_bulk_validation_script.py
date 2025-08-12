#!/usr/bin/env python3
"""
Test script to validate Neo4j bulk UNWIND optimization functionality.
"""

import asyncio
import logging
import os
import time
from typing import List, Dict

from dotenv import load_dotenv
from knowledge_graphs.parse_repo_into_neo4j import DirectNeo4jExtractor

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_data(
    num_files: int = 3, num_classes_per_file: int = 2, num_methods_per_class: int = 3
) -> List[Dict]:
    """Create test data matching the expected modules_data structure"""

    test_data = []

    for file_idx in range(num_files):
        classes = []
        for class_idx in range(num_classes_per_file):
            methods = []
            for method_idx in range(num_methods_per_class):
                methods.append(
                    {
                        "name": f"method_{method_idx}",
                        "args": ["self", f"param_{method_idx}"],
                        "params_list": ["self:Any", f"param_{method_idx}:str"],
                        "params_detailed": [
                            {"name": "self", "type": "Any"},
                            {"name": f"param_{method_idx}", "type": "str"},
                        ],
                        "return_type": "None",
                    }
                )

            classes.append(
                {
                    "name": f"TestClass_{class_idx}",
                    "full_name": f"test_file_{file_idx}.TestClass_{class_idx}",
                    "methods": methods,
                    "attributes": [{"name": f"attr_{class_idx}", "type": "str"}],
                }
            )

        functions = [
            {
                "name": f"test_function_{file_idx}",
                "full_name": f"test_file_{file_idx}.test_function_{file_idx}",
                "args": ["param1", "param2"],
                "params_list": ["param1:str", "param2:int"],
                "params_detailed": [
                    {"name": "param1", "type": "str"},
                    {"name": "param2", "type": "int"},
                ],
                "return_type": "bool",
            }
        ]

        test_data.append(
            {
                "file_path": f"test_files/test_file_{file_idx}.py",
                "module_name": f"test_file_{file_idx}",
                "classes": classes,
                "functions": functions,
                "imports": [f"import_{file_idx}", "os", "sys"],
                "line_count": 50 + file_idx * 10,
                "language": "python",
            }
        )

    return test_data


async def test_bulk_optimization():
    """Test the bulk UNWIND optimization implementation"""

    # Get Neo4j connection details
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    logger.info(f"Testing Neo4j connection to: {neo4j_uri}")

    extractor = DirectNeo4jExtractor(neo4j_uri, neo4j_user, neo4j_password)

    try:
        # Initialize Neo4j connection
        logger.info("Initializing Neo4j connection...")
        await extractor.initialize()
        logger.info("✅ Neo4j connection initialized successfully")

        # Create test data
        test_data = create_test_data(
            num_files=5, num_classes_per_file=2, num_methods_per_class=4
        )
        repo_name = "bulk-optimization-test"

        logger.info(
            f"Created test data: {len(test_data)} files, "
            f"{sum(len(mod['classes']) for mod in test_data)} classes, "
            f"{sum(len(cls['methods']) for mod in test_data for cls in mod['classes'])} methods"
        )

        # Clear any existing test data
        logger.info("Clearing existing test repository data...")
        await extractor.clear_repository_data(repo_name)

        # Also clear any orphaned nodes for clean test environment
        logger.info("Clearing orphaned test nodes...")
        async with extractor.driver.session() as session:
            # Delete all orphaned methods, attributes, functions, classes
            await session.run(
                "MATCH (m:Method) WHERE NOT (m)<-[:HAS_METHOD]-(:Class) DETACH DELETE m"
            )
            await session.run(
                "MATCH (a:Attribute) WHERE NOT (a)<-[:HAS_ATTRIBUTE]-(:Class) DETACH DELETE a"
            )
            await session.run(
                "MATCH (f:Function) WHERE NOT (f)<-[:DEFINES]-(:File) DETACH DELETE f"
            )
            await session.run(
                "MATCH (c:Class) WHERE NOT (c)<-[:DEFINES]-(:File) DETACH DELETE c"
            )

        # Test bulk insertion
        logger.info("Testing bulk UNWIND optimization...")
        start_time = time.time()

        await extractor._create_graph(repo_name, test_data)

        duration = time.time() - start_time
        logger.info(f"✅ Bulk insertion completed in {duration:.3f} seconds")

        # Validate results by querying the database
        logger.info("Validating inserted data...")

        # Test repository exists
        repos_result = await extractor.search_graph("repos")
        if isinstance(repos_result, list):
            logger.info("✅ Found repositories in database")

        # Test file count
        # We can't easily use the existing search methods, so let's run a direct query
        async with extractor.driver.session() as session:
            # Count repositories
            repo_result = await session.run(
                "MATCH (r:Repository {name: $repo_name}) RETURN count(r) as count",
                repo_name=repo_name,
            )
            repo_count = await repo_result.single()
            logger.info(f"✅ Repository count: {repo_count['count']}")

            # Count files
            files_result = await session.run(
                "MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File) RETURN count(f) as count",
                repo_name=repo_name,
            )
            files_count = await files_result.single()
            logger.info(
                f"✅ Files count: {files_count['count']} (expected: {len(test_data)})"
            )

            # Count classes
            classes_result = await session.run(
                "MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class) RETURN count(c) as count",
                repo_name=repo_name,
            )
            classes_count = await classes_result.single()
            expected_classes = sum(len(mod["classes"]) for mod in test_data)
            logger.info(
                f"✅ Classes count: {classes_count['count']} (expected: {expected_classes})"
            )

            # Count methods (only from our test repository)
            methods_result = await session.run(
                "MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)-[:HAS_METHOD]->(m:Method) RETURN count(m) as count",
                repo_name=repo_name,
            )
            methods_count = await methods_result.single()
            expected_methods = sum(
                len(cls["methods"]) for mod in test_data for cls in mod["classes"]
            )
            logger.info(
                f"✅ Methods count: {methods_count['count']} (expected: {expected_methods})"
            )

            # Count functions (only from our test repository)
            functions_result = await session.run(
                "MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(func:Function) RETURN count(func) as count",
                repo_name=repo_name,
            )
            functions_count = await functions_result.single()
            expected_functions = sum(len(mod["functions"]) for mod in test_data)
            logger.info(
                f"✅ Functions count: {functions_count['count']} (expected: {expected_functions})"
            )

        # Validate all counts match expectations
        assert files_count["count"] == len(test_data), (
            f"File count mismatch: {files_count['count']} != {len(test_data)}"
        )
        assert classes_count["count"] == expected_classes, (
            f"Class count mismatch: {classes_count['count']} != {expected_classes}"
        )
        assert methods_count["count"] == expected_methods, (
            f"Method count mismatch: {methods_count['count']} != {expected_methods}"
        )
        assert functions_count["count"] == expected_functions, (
            f"Function count mismatch: {functions_count['count']} != {expected_functions}"
        )

        logger.info("🎉 ALL VALIDATION TESTS PASSED!")
        logger.info("✅ Bulk UNWIND optimization working correctly")
        logger.info(f"✅ Performance: {duration:.3f}s for {len(test_data)} files")
        logger.info(
            "✅ Data integrity: All node/relationship counts match expected values"
        )

        return True

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

    finally:
        # Cleanup
        try:
            if extractor.driver:
                await extractor.clear_repository_data(repo_name)
                await extractor.close()
        except Exception as cleanup_error:
            logger.warning(f"Cleanup error: {cleanup_error}")


async def test_chunking_functionality():
    """Test chunking functionality for large datasets"""

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    extractor = DirectNeo4jExtractor(neo4j_uri, neo4j_user, neo4j_password)

    try:
        await extractor.initialize()

        # Create large test data (will trigger chunking)
        large_test_data = create_test_data(
            num_files=25, num_classes_per_file=1, num_methods_per_class=2
        )
        repo_name = "chunking-test"

        logger.info(
            f"Testing chunking with {len(large_test_data)} files (>10 batch size)"
        )

        await extractor.clear_repository_data(repo_name)

        # Also clear any orphaned nodes for clean test environment
        async with extractor.driver.session() as session:
            # Delete all orphaned methods, attributes, functions, classes
            await session.run(
                "MATCH (m:Method) WHERE NOT (m)<-[:HAS_METHOD]-(:Class) DETACH DELETE m"
            )
            await session.run(
                "MATCH (a:Attribute) WHERE NOT (a)<-[:HAS_ATTRIBUTE]-(:Class) DETACH DELETE a"
            )
            await session.run(
                "MATCH (f:Function) WHERE NOT (f)<-[:DEFINES]-(:File) DETACH DELETE f"
            )
            await session.run(
                "MATCH (c:Class) WHERE NOT (c)<-[:DEFINES]-(:File) DETACH DELETE c"
            )

        start_time = time.time()
        await extractor._create_graph(repo_name, large_test_data)
        duration = time.time() - start_time

        logger.info(f"✅ Chunking test completed in {duration:.3f} seconds")

        return True

    except Exception as e:
        logger.error(f"❌ Chunking test failed: {e}")
        return False

    finally:
        try:
            if extractor.driver:
                await extractor.clear_repository_data(repo_name)
                await extractor.close()
        except Exception as cleanup_error:
            logger.warning(f"Cleanup error: {cleanup_error}")


async def main():
    """Run all validation tests"""

    logger.info("🚀 Starting Neo4j Bulk UNWIND Optimization Tests")

    # Test 1: Basic functionality
    logger.info("\n" + "=" * 50)
    logger.info("TEST 1: Basic Bulk Optimization Functionality")
    logger.info("=" * 50)

    basic_test_passed = await test_bulk_optimization()

    # Test 2: Chunking functionality
    logger.info("\n" + "=" * 50)
    logger.info("TEST 2: Large Dataset Chunking")
    logger.info("=" * 50)

    chunking_test_passed = await test_chunking_functionality()

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)

    if basic_test_passed and chunking_test_passed:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✅ Bulk UNWIND optimization is working correctly")
        logger.info("✅ Data validation successful")
        logger.info("✅ Chunking functionality working")
        return True
    else:
        logger.error("❌ Some tests failed")
        logger.error(f"Basic test: {'PASS' if basic_test_passed else 'FAIL'}")
        logger.error(f"Chunking test: {'PASS' if chunking_test_passed else 'FAIL'}")
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
