"""
File classifier service for intelligent routing in unified indexing.

This service provides fast, extension-based file classification that routes
files to optimal processing destinations (Qdrant RAG vs Neo4j KG) based on
file type and user-defined override patterns.
"""

import re
import time
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Set

# Import models and types
try:
    from ..models.classification_models import (
        IntelligentRoutingConfig,
        ClassificationResult,
        IndexingDestination,
    )
except ImportError:
    from models.classification_models import (
        IntelligentRoutingConfig,
        ClassificationResult,
        IndexingDestination,
    )

# Configure logger
logger = logging.getLogger(__name__)


class FileClassifier:
    """
    Intelligent file classifier for optimal system routing.

    This classifier provides fast, extension-based file classification with
    user override patterns and performance optimization through caching.
    Thread-safe implementation suitable for concurrent usage.
    """

    # PATTERN: Class-level constants for performance (frozen sets for O(1) lookup)
    RAG_EXTENSIONS: Set[str] = frozenset(
        [
            # Documentation files
            ".md",
            ".mdx",
            ".rst",
            ".txt",
            ".adoc",
            ".wiki",
            # Configuration files
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            # Additional documentation formats
            ".markdown",
            ".asciidoc",
        ]
    )

    KG_EXTENSIONS: Set[str] = frozenset(
        [
            # Python
            ".py",
            ".pyi",
            # JavaScript/TypeScript
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".mjs",
            ".cjs",
            # Java ecosystem
            ".java",
            ".scala",
            ".kt",
            ".kts",
            # Systems programming
            ".go",
            ".rs",
            ".c",
            ".cpp",
            ".cc",
            ".cxx",
            ".h",
            ".hpp",
            ".hxx",
            ".hh",
            # Other languages
            ".cs",
            ".php",
            ".rb",
            ".swift",
            ".dart",
            ".m",
            ".mm",
        ]
    )

    def __init__(self):
        """Initialize the file classifier with performance tracking."""
        self.stats: Dict[str, int] = {
            "total_classifications": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "override_applications": 0,
            "rag_classifications": 0,
            "kg_classifications": 0,
            "unknown_extensions": 0,
        }

        logger.debug(
            f"FileClassifier initialized with {len(self.RAG_EXTENSIONS)} RAG extensions and {len(self.KG_EXTENSIONS)} KG extensions"
        )

    @lru_cache(maxsize=512)  # PATTERN: Caching for performance, thread-safe
    def _classify_extension_cached(
        self, extension: str
    ) -> tuple[IndexingDestination, float, str]:
        """
        Cache classification results by extension for performance.

        Args:
            extension: File extension in lowercase (e.g., '.py', '.md')

        Returns:
            Tuple of (destination, confidence, reasoning)
        """
        if extension in self.RAG_EXTENSIONS:
            return (
                IndexingDestination.QDRANT,
                1.0,
                f"Documentation/config file: {extension}",
            )
        elif extension in self.KG_EXTENSIONS:
            return IndexingDestination.NEO4J, 1.0, f"Code file: {extension}"
        else:
            return (
                IndexingDestination.QDRANT,
                0.5,
                f"Unknown extension, defaulting to documentation: {extension}",
            )

    def classify_by_extension(self, file_path: str) -> ClassificationResult:
        """
        Classify file by extension with performance tracking.

        Args:
            file_path: Path to the file to classify

        Returns:
            ClassificationResult with destination, confidence, and timing
        """
        start_time = time.perf_counter()

        # Extract extension and normalize
        extension = Path(file_path).suffix.lower()

        # Use cached classification
        destination, confidence, reasoning = self._classify_extension_cached(extension)

        # Update statistics
        self.stats["total_classifications"] += 1
        if destination == IndexingDestination.QDRANT:
            self.stats["rag_classifications"] += 1
        elif destination == IndexingDestination.NEO4J:
            self.stats["kg_classifications"] += 1

        if confidence < 1.0:
            self.stats["unknown_extensions"] += 1

        # Calculate timing
        duration_ms = (time.perf_counter() - start_time) * 1000

        result = ClassificationResult(
            destination=destination,
            confidence=confidence,
            reasoning=reasoning,
            classification_time_ms=duration_ms,
        )

        logger.debug(
            f"Classified {file_path} -> {destination.value} (confidence: {confidence:.2f})"
        )
        return result

    def apply_user_overrides(
        self,
        file_path: str,
        base_classification: ClassificationResult,
        routing_config: IntelligentRoutingConfig,
    ) -> ClassificationResult:
        """
        Apply user-defined routing overrides to base classification.

        Args:
            file_path: Path to the file being classified
            base_classification: Initial classification result
            routing_config: User routing configuration with patterns

        Returns:
            Updated ClassificationResult with overrides applied
        """
        start_time = time.perf_counter()
        applied_overrides = []

        # Check force RAG patterns
        for pattern in routing_config.force_rag_patterns:
            try:
                if re.search(pattern, file_path, re.IGNORECASE):
                    applied_overrides.append(f"force_rag: {pattern}")
                    logger.debug(
                        f"Applied force_rag override for {file_path}: {pattern}"
                    )

                    duration_ms = (time.perf_counter() - start_time) * 1000
                    self.stats["override_applications"] += 1

                    return ClassificationResult(
                        destination=IndexingDestination.QDRANT,
                        confidence=1.0,  # High confidence for explicit overrides
                        reasoning=f"Force RAG override: {pattern}",
                        applied_overrides=applied_overrides,
                        classification_time_ms=base_classification.classification_time_ms
                        + duration_ms,
                    )
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                continue

        # Check force KG patterns
        for pattern in routing_config.force_kg_patterns:
            try:
                if re.search(pattern, file_path, re.IGNORECASE):
                    applied_overrides.append(f"force_kg: {pattern}")
                    logger.debug(
                        f"Applied force_kg override for {file_path}: {pattern}"
                    )

                    duration_ms = (time.perf_counter() - start_time) * 1000
                    self.stats["override_applications"] += 1

                    return ClassificationResult(
                        destination=IndexingDestination.NEO4J,
                        confidence=1.0,  # High confidence for explicit overrides
                        reasoning=f"Force KG override: {pattern}",
                        applied_overrides=applied_overrides,
                        classification_time_ms=base_classification.classification_time_ms
                        + duration_ms,
                    )
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                continue

        # No overrides applied, return original classification
        duration_ms = (time.perf_counter() - start_time) * 1000
        base_classification.classification_time_ms += duration_ms
        return base_classification

    def classify_file(
        self, file_path: str, routing_config: IntelligentRoutingConfig
    ) -> ClassificationResult:
        """
        Classify file with full pipeline including overrides.

        Args:
            file_path: Path to the file to classify
            routing_config: Routing configuration including override patterns

        Returns:
            Final ClassificationResult with all processing applied
        """
        # Primary classification by extension
        base_result = self.classify_by_extension(file_path)

        # Apply user overrides only if intelligent routing is enabled and patterns are configured
        if routing_config.enable_intelligent_routing and (
            routing_config.force_rag_patterns or routing_config.force_kg_patterns
        ):
            return self.apply_user_overrides(file_path, base_result, routing_config)

        return base_result

    def classify_with_fallback(
        self,
        file_path: str,
        content: str = None,
        routing_config: IntelligentRoutingConfig = None,
    ) -> ClassificationResult:
        """
        Classify with comprehensive fallback logic for edge cases.

        Args:
            file_path: Path to the file to classify
            content: Optional file content for content-based analysis
            routing_config: Optional routing configuration

        Returns:
            ClassificationResult with fallback handling
        """
        try:
            # Use standard classification if routing config provided
            if routing_config:
                return self.classify_file(file_path, routing_config)

            # Fallback to basic extension classification
            result = self.classify_by_extension(file_path)

            # Apply confidence threshold fallback
            if result.confidence < 0.8:
                logger.warning(
                    f"Low confidence classification for {file_path}, using safe default"
                )
                return ClassificationResult(
                    destination=IndexingDestination.QDRANT,  # Safe default
                    confidence=0.5,
                    reasoning="Low confidence, defaulting to documentation processing",
                    classification_time_ms=result.classification_time_ms,
                )

            return result

        except Exception as e:
            logger.error(f"Classification failed for {file_path}: {e}")
            return ClassificationResult(
                destination=IndexingDestination.QDRANT,  # Safe default
                confidence=0.0,
                reasoning=f"Classification error: {e}",
                classification_time_ms=0.0,
            )

    def get_statistics(self) -> Dict[str, int]:
        """Get classification statistics for monitoring and debugging."""
        # Add cache statistics
        stats_with_cache = self.stats.copy()
        stats_with_cache["cache_info"] = dict(
            self._classify_extension_cached.cache_info()._asdict()
        )
        return stats_with_cache

    def clear_cache(self) -> None:
        """Clear the classification cache (useful for testing)."""
        self._classify_extension_cached.cache_clear()
        logger.debug("Classification cache cleared")

    def get_supported_extensions(self) -> Dict[str, List[str]]:
        """Get lists of supported extensions by category."""
        return {
            "rag_extensions": sorted(list(self.RAG_EXTENSIONS)),
            "kg_extensions": sorted(list(self.KG_EXTENSIONS)),
            "total_supported": len(self.RAG_EXTENSIONS) + len(self.KG_EXTENSIONS),
        }
