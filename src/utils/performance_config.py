"""
Performance configuration for the repository indexing pipeline.

This module provides configurable batch sizes and worker counts for optimizing
repository processing performance across CPU-bound and I/O-bound operations.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """
    Configuration class for batch processing performance parameters.

    Provides environment variable-based configuration with sensible defaults
    for different types of operations in the repository indexing pipeline.
    """

    # Worker configuration
    cpu_workers: int = 4  # ProcessPoolExecutor workers for CPU-bound tasks
    io_workers: int = 10  # ThreadPoolExecutor workers for I/O-bound tasks

    # Batch size configuration for different operations
    batch_size_qdrant: int = 500  # Qdrant vector insertion batch size
    batch_size_neo4j: int = 5000  # Neo4j bulk operation batch size
    batch_size_embeddings: int = 1000  # OpenAI embeddings API batch size
    batch_size_file_processing: int = 10  # Concurrent file processing batch size

    # Memory and performance tuning
    max_concurrent_parsing: int = 8  # Maximum concurrent parsing operations
    embedding_chunk_size: int = 100  # Chunk size for embedding generation

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Ensure minimum values for safety
        if self.cpu_workers < 1:
            logger.warning(f"CPU workers must be at least 1, got {self.cpu_workers}")
            self.cpu_workers = 1

        if self.io_workers < 1:
            logger.warning(f"I/O workers must be at least 1, got {self.io_workers}")
            self.io_workers = 1

        # ENTERPRISE HARDWARE OPTIMIZATION: Removed restrictive batch size limits
        # Your Dual Xeon E5-2673 v4 setup with 128GB RAM can handle much larger batches

        # Only warn for extremely high values but don't limit them
        if self.batch_size_embeddings > 10000:
            logger.info(
                f"Large embedding batch size configured: {self.batch_size_embeddings}. "
                f"Ensure your API provider supports this batch size."
            )

        if self.batch_size_qdrant > 5000:
            logger.info(
                f"Large Qdrant batch size configured: {self.batch_size_qdrant}. "
                f"This is optimal for your enterprise hardware configuration."
            )

        logger.debug(
            f"Performance config initialized: CPU workers={self.cpu_workers}, "
            f"I/O workers={self.io_workers}, Qdrant batch={self.batch_size_qdrant}, "
            f"Embeddings batch={self.batch_size_embeddings}"
        )


def load_performance_config() -> PerformanceConfig:
    """
    Load performance configuration from environment variables with fallback defaults.

    Environment Variables:
        CPU_WORKERS: Number of ProcessPoolExecutor workers for CPU-bound tasks (default: 4)
        IO_WORKERS: Number of ThreadPoolExecutor workers for I/O-bound tasks (default: 10)
        BATCH_SIZE_QDRANT: Qdrant vector insertion batch size (default: 500)
        BATCH_SIZE_NEO4J: Neo4j bulk operation batch size (default: 5000)
        BATCH_SIZE_EMBEDDINGS: OpenAI embeddings API batch size (default: 1000)
        BATCH_SIZE_FILE_PROCESSING: Concurrent file processing batch size (default: 10)
        MAX_CONCURRENT_PARSING: Maximum concurrent parsing operations (default: 8)
        EMBEDDING_CHUNK_SIZE: Chunk size for embedding generation (default: 100)

    Returns:
        PerformanceConfig: Configured performance parameters
    """
    # Load from environment with defaults
    cpu_workers = int(os.getenv("CPU_WORKERS", "4"))
    io_workers = int(os.getenv("IO_WORKERS", "10"))

    batch_size_qdrant = int(os.getenv("BATCH_SIZE_QDRANT", "500"))
    batch_size_neo4j = int(os.getenv("BATCH_SIZE_NEO4J", "5000"))
    batch_size_embeddings = int(os.getenv("BATCH_SIZE_EMBEDDINGS", "1000"))
    batch_size_file_processing = int(os.getenv("BATCH_SIZE_FILE_PROCESSING", "10"))

    max_concurrent_parsing = int(os.getenv("MAX_CONCURRENT_PARSING", "8"))
    embedding_chunk_size = int(os.getenv("EMBEDDING_CHUNK_SIZE", "100"))

    # Create configuration
    config = PerformanceConfig(
        cpu_workers=cpu_workers,
        io_workers=io_workers,
        batch_size_qdrant=batch_size_qdrant,
        batch_size_neo4j=batch_size_neo4j,
        batch_size_embeddings=batch_size_embeddings,
        batch_size_file_processing=batch_size_file_processing,
        max_concurrent_parsing=max_concurrent_parsing,
        embedding_chunk_size=embedding_chunk_size,
    )

    logger.info(
        f"Loaded performance configuration: "
        f"CPU={config.cpu_workers}, I/O={config.io_workers}, "
        f"Qdrant batch={config.batch_size_qdrant}, "
        f"Neo4j batch={config.batch_size_neo4j}"
    )

    return config


def get_optimal_worker_count(task_type: str = "cpu") -> int:
    """
    Get optimal worker count based on system capabilities and task type.

    Args:
        task_type: Type of task - "cpu" for CPU-bound, "io" for I/O-bound

    Returns:
        Optimal number of workers for the specified task type
    """
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()

    if task_type == "cpu":
        # For CPU-bound tasks, use all cores but respect environment override
        optimal = int(os.getenv("CPU_WORKERS", str(cpu_count)))
        # Cap at CPU count to avoid oversubscription
        return min(optimal, cpu_count)
    elif task_type == "io":
        # For I/O-bound tasks, can use more workers than CPU count
        optimal = int(os.getenv("IO_WORKERS", str(cpu_count * 2)))
        # ENTERPRISE HARDWARE OPTIMIZATION: Increased cap for high-memory systems
        # Your 128GB RAM setup can handle much higher I/O concurrency
        return min(optimal, 200)  # Increased from 20 to 200 for enterprise hardware
    else:
        logger.warning(f"Unknown task type '{task_type}', defaulting to CPU count")
        return cpu_count


def get_optimal_batch_size(operation_type: str) -> int:
    """
    Get optimal batch size for specific operations.

    Args:
        operation_type: Type of operation - "qdrant", "neo4j", "embeddings", "files"

    Returns:
        Optimal batch size for the specified operation
    """
    config = load_performance_config()

    batch_size_map = {
        "qdrant": config.batch_size_qdrant,
        "neo4j": config.batch_size_neo4j,
        "embeddings": config.batch_size_embeddings,
        "files": config.batch_size_file_processing,
    }

    return batch_size_map.get(operation_type, 100)


# Global configuration instance
_performance_config: Optional[PerformanceConfig] = None


def get_performance_config() -> PerformanceConfig:
    """
    Get global performance configuration instance (singleton pattern).

    Returns:
        PerformanceConfig: Global configuration instance
    """
    global _performance_config
    if _performance_config is None:
        _performance_config = load_performance_config()
    return _performance_config


def reset_performance_config():
    """Reset the global performance configuration. Useful for testing."""
    global _performance_config
    _performance_config = None
    logger.debug("Performance configuration reset")
