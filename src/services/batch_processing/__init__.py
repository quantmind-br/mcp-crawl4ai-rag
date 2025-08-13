"""
Batch processing utilities for optimized repository indexing.

This package provides high-performance batch processing components for the
repository indexing pipeline, including CPU-bound parsing and I/O operations.
"""

from .file_processor import (
    parse_file_for_kg,
    read_file_async,
    process_file_batch,
)
from .pipeline_stages import OptimizedIndexingPipeline

__all__ = [
    "parse_file_for_kg",
    "read_file_async",
    "process_file_batch",
    "OptimizedIndexingPipeline",
]
