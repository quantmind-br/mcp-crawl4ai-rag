"""
Sparse vector types and configurations for hybrid search.

This module contains the data structures and types needed for sparse vector
operations, separated to avoid circular imports.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class SparseVectorConfig:
    """Configuration for sparse vectors with indices and values."""

    indices: List[int]
    values: List[float]

    def to_qdrant_sparse_vector(self):
        """Convert to Qdrant SparseVector format."""
        from qdrant_client.models import SparseVector

        return SparseVector(indices=self.indices, values=self.values)
