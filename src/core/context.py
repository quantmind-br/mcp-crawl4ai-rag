"""Application context for the Crawl4AI MCP server."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from crawl4ai import AsyncWebCrawler
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass
class Crawl4AIContext:
    """Enhanced context for the Crawl4AI MCP server containing all application dependencies.

    This context serves as a dependency injection container that holds references
    to all major application components including web crawler, vector database client,
    machine learning models, services, and executors for performance optimization.

    Attributes:
        crawler: AsyncWebCrawler instance for web scraping
        qdrant_client: Qdrant vector database client wrapper
        embedding_cache: Redis cache for embedding storage
        reranker: CrossEncoder model for result reranking (optional)
        knowledge_validator: Knowledge graph validator for hallucination detection (optional)
        repo_extractor: Neo4j repository extractor for code analysis (optional)
        embedding_service: Service for generating and managing embeddings
        rag_service: Service for retrieval-augmented generation operations
        io_executor: ThreadPoolExecutor for I/O-bound operations
        cpu_executor: ProcessPoolExecutor for CPU-bound operations (parsing)
        performance_config: Performance configuration settings
    """

    # Core infrastructure components
    crawler: AsyncWebCrawler
    qdrant_client: Any  # QdrantClientWrapper - using Any to avoid circular import
    embedding_cache: Any  # EmbeddingCache instance

    # Executors for performance optimization
    io_executor: Optional[ThreadPoolExecutor] = None
    cpu_executor: Optional[ProcessPoolExecutor] = None
    performance_config: Optional[Any] = None  # PerformanceConfig instance

    # Optional ML models
    reranker: Optional[CrossEncoder] = None
    knowledge_validator: Optional[Any] = None  # KnowledgeGraphValidator when available
    repo_extractor: Optional[Any] = None  # DirectNeo4jExtractor when available

    # Application services (to be added in future phases)
    embedding_service: Optional[Any] = None  # EmbeddingService instance
    rag_service: Optional[Any] = None  # RagService instance

    async def __aenter__(self):
        """Async context manager entry."""
        # Initialize executors if not already done
        if not self.io_executor or not self.cpu_executor:
            await self.initialize_executors()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        await self.cleanup_executors()

    async def initialize_executors(self):
        """Initialize ThreadPoolExecutor and ProcessPoolExecutor for performance optimization."""
        try:
            # Import performance config
            try:
                from ..utils.performance_config import get_performance_config
            except ImportError:
                from utils.performance_config import get_performance_config

            # Load configuration if not provided
            if not self.performance_config:
                self.performance_config = get_performance_config()

            # Initialize ThreadPoolExecutor for I/O-bound operations
            if not self.io_executor:
                self.io_executor = ThreadPoolExecutor(
                    max_workers=self.performance_config.io_workers,
                    thread_name_prefix="crawl4ai-io",
                )
                logger.info(
                    f"Initialized ThreadPoolExecutor with {self.performance_config.io_workers} workers"
                )

            # Initialize ProcessPoolExecutor for CPU-bound operations
            if not self.cpu_executor:
                self.cpu_executor = ProcessPoolExecutor(
                    max_workers=self.performance_config.cpu_workers,
                )
                logger.info(
                    f"Initialized ProcessPoolExecutor with {self.performance_config.cpu_workers} workers"
                )

        except Exception as e:
            logger.error(f"Error initializing executors: {e}")
            # Fallback to minimal configuration
            if not self.io_executor:
                self.io_executor = ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="crawl4ai-io-fallback"
                )
            if not self.cpu_executor:
                self.cpu_executor = ProcessPoolExecutor(max_workers=2)

    async def cleanup_executors(self):
        """Clean up executors and release resources."""
        cleanup_errors = []

        # Shutdown ProcessPoolExecutor
        if self.cpu_executor:
            try:
                logger.debug("Shutting down ProcessPoolExecutor")
                self.cpu_executor.shutdown(wait=True, cancel_futures=False)
                self.cpu_executor = None
                logger.info("ProcessPoolExecutor shutdown completed")
            except Exception as e:
                cleanup_errors.append(f"ProcessPoolExecutor cleanup error: {e}")
                logger.error(f"Error shutting down ProcessPoolExecutor: {e}")

        # Shutdown ThreadPoolExecutor
        if self.io_executor:
            try:
                logger.debug("Shutting down ThreadPoolExecutor")
                self.io_executor.shutdown(wait=True, cancel_futures=False)
                self.io_executor = None
                logger.info("ThreadPoolExecutor shutdown completed")
            except Exception as e:
                cleanup_errors.append(f"ThreadPoolExecutor cleanup error: {e}")
                logger.error(f"Error shutting down ThreadPoolExecutor: {e}")

        if cleanup_errors:
            logger.warning(
                f"Executor cleanup completed with {len(cleanup_errors)} errors"
            )
        else:
            logger.info("All executors cleaned up successfully")

    def ensure_executors_initialized(self):
        """Ensure executors are initialized (synchronous version for compatibility)."""
        if not self.io_executor or not self.cpu_executor:
            # Use asyncio.run to initialize if we're not in an async context
            try:
                asyncio.get_running_loop()
                # We're in an async context, create a task
                asyncio.create_task(self.initialize_executors())
            except RuntimeError:
                # No running loop, use asyncio.run
                asyncio.run(self.initialize_executors())
