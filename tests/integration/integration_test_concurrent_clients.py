#!/usr/bin/env python3
"""
Integration test to verify that multiple MCP clients can connect while
index_github_repository tool is running, proving the event loop is not blocked.

This test simulates the user's reported issue and validates the fix.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Mock context for testing
def create_mock_context():
    """Create a mock MCP context with ThreadPoolExecutor."""
    context = Mock()
    context.request_context = Mock()
    context.request_context.lifespan_context = Mock()

    # Mock components with actual ThreadPoolExecutor
    context.request_context.lifespan_context.qdrant_client = Mock()
    context.request_context.lifespan_context.cpu_executor = ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="integration_test"
    )

    return context


async def simulate_repository_indexing(
    ctx, repo_url="https://github.com/test/small-repo"
):
    """Simulate repository indexing that previously blocked the event loop."""
    try:
        # Import the actual tool function
        from src.tools.github_tools import index_github_repository

        # Mock GitHub processor to avoid actual git operations
        with patch(
            "src.services.unified_indexing_service.GitHubProcessor"
        ) as mock_github_proc_class:
            mock_github_proc = Mock()

            # Simulate CPU-intensive repository cloning
            def slow_clone_with_cpu_work(*args, **kwargs):
                """Simulate repository cloning with CPU work."""
                logger.info("🔄 Starting simulated repository cloning (CPU-intensive)")

                # Simulate CPU-bound work that would block event loop if not properly handled
                start_time = time.time()
                result = 0
                while time.time() - start_time < 0.5:  # 500ms of CPU work
                    result += sum(range(1000))

                logger.info(
                    f"✅ Completed repository cloning simulation (result: {result})"
                )
                return {"success": True, "temp_directory": "/tmp/test_repo_simulation"}

            mock_github_proc.clone_repository_temp = Mock(
                side_effect=slow_clone_with_cpu_work
            )
            mock_github_proc_class.return_value = mock_github_proc

            # Mock file discovery to return empty (focus on the blocking issue)
            from src.services.unified_indexing_service import UnifiedIndexingService

            async def mock_discover_files(*args, **kwargs):
                """Mock file discovery with some async delay."""
                await asyncio.sleep(0.1)  # Small I/O delay
                return []  # No files to avoid complex processing

            with patch.object(
                UnifiedIndexingService,
                "_discover_repository_files",
                side_effect=mock_discover_files,
            ):
                # Call the actual index_github_repository tool
                logger.info(f"🚀 Starting repository indexing for {repo_url}")
                result = await index_github_repository(
                    ctx,
                    repo_url=repo_url,
                    destination="qdrant",
                    file_types=[".py"],
                    max_files=5,
                )

                # Parse result
                result_data = json.loads(result)
                logger.info(
                    f"📊 Repository indexing result: {result_data.get('success', False)}"
                )
                return result_data

    except Exception as e:
        logger.error(f"❌ Error in repository indexing: {e}")
        return {"success": False, "error": str(e)}


async def simulate_concurrent_client_connection(client_id: int, delay: float = 0.1):
    """Simulate a concurrent MCP client connection during repository processing."""
    await asyncio.sleep(delay)  # Stagger client connections

    logger.info(f"🔌 Client {client_id}: Attempting to connect...")

    # Simulate client connection work (I/O bound)
    start_time = asyncio.get_event_loop().time()
    await asyncio.sleep(0.1)  # Simulate connection establishment
    end_time = asyncio.get_event_loop().time()

    response_time = end_time - start_time

    if response_time < 0.2:  # Should be responsive
        logger.info(
            f"✅ Client {client_id}: Connected successfully (response time: {response_time:.3f}s)"
        )
        return {"client_id": client_id, "success": True, "response_time": response_time}
    else:
        logger.error(
            f"❌ Client {client_id}: Connection slow/blocked (response time: {response_time:.3f}s)"
        )
        return {
            "client_id": client_id,
            "success": False,
            "response_time": response_time,
        }


async def main():
    """Main integration test function."""
    logger.info(
        "🧪 Starting integration test: Multiple clients during repository indexing"
    )
    logger.info("=" * 70)

    # Create mock context with ThreadPoolExecutor
    ctx = create_mock_context()

    try:
        # Start repository indexing task (previously would block the event loop)
        indexing_task = asyncio.create_task(
            simulate_repository_indexing(
                ctx, "https://github.com/test/integration-test"
            )
        )

        # Start multiple concurrent client connections (should remain responsive)
        client_tasks = []
        for i in range(5):
            client_task = asyncio.create_task(
                simulate_concurrent_client_connection(i + 1, delay=i * 0.05)
            )
            client_tasks.append(client_task)

        logger.info(
            "⏱️  Running repository indexing and client connections concurrently..."
        )

        # Execute both repository indexing and client connections concurrently
        start_time = time.time()

        # Wait for both indexing and all client connections
        indexing_result = await indexing_task
        client_results = await asyncio.gather(*client_tasks)

        end_time = time.time()
        total_time = end_time - start_time

        logger.info("=" * 70)
        logger.info("📋 INTEGRATION TEST RESULTS:")
        logger.info(f"   Total execution time: {total_time:.3f}s")

        # Check repository indexing
        indexing_success = indexing_result.get("success", False)
        logger.info(
            f"   Repository indexing: {'✅ SUCCESS' if indexing_success else '❌ FAILED'}"
        )

        # Check client connections
        successful_clients = sum(1 for result in client_results if result["success"])
        total_clients = len(client_results)

        logger.info(
            f"   Client connections: {successful_clients}/{total_clients} successful"
        )

        # Calculate average response time
        avg_response_time = sum(r["response_time"] for r in client_results) / len(
            client_results
        )
        logger.info(f"   Average client response time: {avg_response_time:.3f}s")

        # Test verdict
        test_passed = (
            indexing_success
            and successful_clients == total_clients
            and avg_response_time < 0.15  # Should be very responsive
        )

        logger.info("=" * 70)
        if test_passed:
            logger.info("🎉 INTEGRATION TEST PASSED!")
            logger.info(
                "   ✅ Repository indexing completed without blocking event loop"
            )
            logger.info("   ✅ All client connections remained responsive")
            logger.info("   ✅ ThreadPoolExecutor integration working correctly")
        else:
            logger.error("❌ INTEGRATION TEST FAILED!")
            if not indexing_success:
                logger.error("   ❌ Repository indexing failed")
            if successful_clients != total_clients:
                logger.error(
                    f"   ❌ Only {successful_clients}/{total_clients} clients connected"
                )
            if avg_response_time >= 0.15:
                logger.error(
                    f"   ❌ Client response time too slow: {avg_response_time:.3f}s"
                )

        return test_passed

    except Exception as e:
        logger.error(f"❌ Integration test failed with exception: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

    finally:
        # Cleanup
        if hasattr(ctx.request_context.lifespan_context, "cpu_executor"):
            ctx.request_context.lifespan_context.cpu_executor.shutdown(wait=True)
            logger.info("🧹 Cleaned up ThreadPoolExecutor")


if __name__ == "__main__":
    # Run the integration test
    success = asyncio.run(main())

    print()
    if success:
        print("🎉 Integration test PASSED - Event loop blocking issue has been FIXED!")
        exit(0)
    else:
        print("❌ Integration test FAILED - Event loop blocking issue persists!")
        exit(1)
