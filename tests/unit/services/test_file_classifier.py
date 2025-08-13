"""
Unit tests for FileClassifier service.

Tests the intelligent file classification system for optimal routing
between Qdrant (RAG) and Neo4j (KG) processing destinations.
"""

import pytest
from unittest.mock import patch

# Import the classes to test
from src.services.file_classifier import FileClassifier
from src.models.classification_models import (
    IntelligentRoutingConfig,
    ClassificationResult,
    IndexingDestination,
)


class TestFileClassifier:
    """Test suite for FileClassifier core functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.classifier = FileClassifier()

    def test_initialization(self):
        """Test FileClassifier initialization."""
        assert isinstance(self.classifier, FileClassifier)
        assert self.classifier.stats["total_classifications"] == 0
        assert len(self.classifier.RAG_EXTENSIONS) > 0
        assert len(self.classifier.KG_EXTENSIONS) > 0

        # Verify no overlap between extension sets
        overlap = self.classifier.RAG_EXTENSIONS.intersection(
            self.classifier.KG_EXTENSIONS
        )
        assert len(overlap) == 0, f"Found overlapping extensions: {overlap}"

    def test_rag_extension_classification(self):
        """Test classification of RAG-targeted files."""
        rag_test_cases = [
            ("document.md", ".md"),
            ("README.rst", ".rst"),
            ("config.json", ".json"),
            ("settings.yaml", ".yaml"),
            ("data.yml", ".yml"),
            ("config.toml", ".toml"),
            ("setup.cfg", ".cfg"),
            ("notes.txt", ".txt"),
        ]

        for file_path, expected_ext in rag_test_cases:
            result = self.classifier.classify_by_extension(file_path)

            assert result.destination == IndexingDestination.QDRANT
            assert result.confidence == 1.0
            assert expected_ext in result.reasoning
            assert result.classification_time_ms >= 0

    def test_kg_extension_classification(self):
        """Test classification of KG-targeted files."""
        kg_test_cases = [
            ("module.py", ".py"),
            ("script.js", ".js"),
            ("component.tsx", ".tsx"),
            ("app.ts", ".ts"),
            ("Main.java", ".java"),
            ("main.go", ".go"),
            ("lib.rs", ".rs"),
            ("core.cpp", ".cpp"),
            ("header.h", ".h"),
        ]

        for file_path, expected_ext in kg_test_cases:
            result = self.classifier.classify_by_extension(file_path)

            assert result.destination == IndexingDestination.NEO4J
            assert result.confidence == 1.0
            assert expected_ext in result.reasoning
            assert result.classification_time_ms >= 0

    def test_unknown_extension_classification(self):
        """Test classification of unknown file extensions."""
        unknown_test_cases = [
            "file.xyz",
            "document.unknown",
            "data.custom",
            "file",  # No extension
        ]

        for file_path in unknown_test_cases:
            result = self.classifier.classify_by_extension(file_path)

            assert result.destination == IndexingDestination.QDRANT  # Default to RAG
            assert result.confidence == 0.5  # Lower confidence
            assert "Unknown extension" in result.reasoning
            assert result.classification_time_ms >= 0

    def test_case_insensitive_extensions(self):
        """Test that extension classification is case-insensitive."""
        test_cases = [
            ("FILE.PY", IndexingDestination.NEO4J),
            ("Document.MD", IndexingDestination.QDRANT),
            ("Config.JSON", IndexingDestination.QDRANT),
            ("Script.JS", IndexingDestination.NEO4J),
        ]

        for file_path, expected_destination in test_cases:
            result = self.classifier.classify_by_extension(file_path)
            assert result.destination == expected_destination
            assert result.confidence == 1.0

    def test_extension_caching(self):
        """Test that extension classification results are cached."""
        # Clear cache to start fresh
        self.classifier.clear_cache()

        # First classification should be a cache miss
        result1 = self.classifier.classify_by_extension("test.py")
        cache_info_1 = dict(
            self.classifier._classify_extension_cached.cache_info()._asdict()
        )

        # Second classification of same extension should be a cache hit
        result2 = self.classifier.classify_by_extension("another.py")
        cache_info_2 = dict(
            self.classifier._classify_extension_cached.cache_info()._asdict()
        )

        # Results should be identical
        assert result1.destination == result2.destination
        assert result1.confidence == result2.confidence

        # Cache hits should increase
        assert cache_info_2["hits"] > cache_info_1["hits"]

    def test_statistics_tracking(self):
        """Test that classification statistics are properly tracked."""
        # Clear cache and reset stats
        self.classifier.clear_cache()
        initial_stats = self.classifier.stats.copy()

        # Perform various classifications
        self.classifier.classify_by_extension("test.py")  # KG
        self.classifier.classify_by_extension("doc.md")  # RAG
        self.classifier.classify_by_extension("unknown.xyz")  # Unknown

        final_stats = self.classifier.get_statistics()

        # Check statistics updates
        assert (
            final_stats["total_classifications"]
            == initial_stats["total_classifications"] + 3
        )
        assert (
            final_stats["kg_classifications"] == initial_stats["kg_classifications"] + 1
        )
        assert (
            final_stats["rag_classifications"]
            == initial_stats["rag_classifications"]
            + 2  # .md and unknown.xyz both go to RAG
        )
        assert (
            final_stats["unknown_extensions"] == initial_stats["unknown_extensions"] + 1
        )


class TestIntelligentRouting:
    """Test suite for intelligent routing with user overrides."""

    def setup_method(self):
        """Setup for each test method."""
        self.classifier = FileClassifier()

    def test_force_rag_patterns(self):
        """Test force RAG pattern overrides."""
        routing_config = IntelligentRoutingConfig(
            enable_intelligent_routing=True,
            force_rag_patterns=[".*README.*", ".*docs/.*", ".*\\.test\\.py$"],
            force_kg_patterns=[],
        )

        test_cases = [
            ("README.md", IndexingDestination.QDRANT),  # Would be RAG anyway
            ("README.py", IndexingDestination.QDRANT),  # Override from KG to RAG
            ("docs/api.py", IndexingDestination.QDRANT),  # Override from KG to RAG
            ("src/test.test.py", IndexingDestination.QDRANT),  # Override from KG to RAG
            ("src/main.py", IndexingDestination.NEO4J),  # No override, normal KG
        ]

        for file_path, expected_destination in test_cases:
            result = self.classifier.classify_file(file_path, routing_config)
            assert result.destination == expected_destination

            # Check if override was applied
            if "README" in file_path or "docs/" in file_path or ".test.py" in file_path:
                if file_path.endswith(".py"):  # Override applied
                    assert "Force RAG override" in result.reasoning
                    assert result.confidence == 1.0

    def test_force_kg_patterns(self):
        """Test force KG pattern overrides."""
        routing_config = IntelligentRoutingConfig(
            enable_intelligent_routing=True,
            force_rag_patterns=[],
            force_kg_patterns=[".*\\.config\\.json$", ".*spec.*", ".*test.*"],
        )

        test_cases = [
            ("app.config.json", IndexingDestination.NEO4J),  # Override from RAG to KG
            ("spec.md", IndexingDestination.NEO4J),  # Override from RAG to KG
            ("test_results.yaml", IndexingDestination.NEO4J),  # Override from RAG to KG
            ("normal.json", IndexingDestination.QDRANT),  # No override, normal RAG
            ("main.py", IndexingDestination.NEO4J),  # No override, normal KG
        ]

        for file_path, expected_destination in test_cases:
            result = self.classifier.classify_file(file_path, routing_config)
            assert result.destination == expected_destination

            # Check if override was applied
            if any(pattern in file_path for pattern in ["config.json", "spec", "test"]):
                if not file_path.endswith(".py"):  # Override applied to non-KG files
                    assert "Force KG override" in result.reasoning
                    assert result.confidence == 1.0

    def test_conflicting_patterns_priority(self):
        """Test that force RAG patterns take priority over force KG patterns."""
        routing_config = IntelligentRoutingConfig(
            enable_intelligent_routing=True,
            force_rag_patterns=[".*README.*"],
            force_kg_patterns=[".*README.*"],  # Conflicting pattern
        )

        result = self.classifier.classify_file("README.py", routing_config)

        # Force RAG should take priority (checked first)
        assert result.destination == IndexingDestination.QDRANT
        assert "Force RAG override" in result.reasoning

    def test_invalid_regex_patterns(self):
        """Test handling of invalid regex patterns."""
        routing_config = IntelligentRoutingConfig(
            enable_intelligent_routing=True,
            force_rag_patterns=["[invalid_regex"],  # Invalid regex
            force_kg_patterns=["*[another_invalid"],  # Invalid regex
        )

        # Should fall back to normal classification without crashing
        with patch("src.services.file_classifier.logger") as mock_logger:
            result = self.classifier.classify_file("test.py", routing_config)

            # Should still classify normally
            assert result.destination == IndexingDestination.NEO4J
            assert result.confidence == 1.0

            # Should log warnings about invalid patterns
            assert mock_logger.warning.called

    def test_disabled_intelligent_routing(self):
        """Test behavior when intelligent routing is disabled."""
        routing_config = IntelligentRoutingConfig(
            enable_intelligent_routing=False,
            force_rag_patterns=[".*\\.py$"],  # Should be ignored
            force_kg_patterns=[".*\\.md$"],  # Should be ignored
        )

        # When disabled, only basic extension classification should be used
        result_py = self.classifier.classify_file("test.py", routing_config)
        result_md = self.classifier.classify_file("test.md", routing_config)

        # Should follow normal extension-based classification
        assert result_py.destination == IndexingDestination.NEO4J
        assert result_md.destination == IndexingDestination.QDRANT

        # Should not have override reasoning
        assert "override" not in result_py.reasoning.lower()
        assert "override" not in result_md.reasoning.lower()


class TestFallbackAndEdgeCases:
    """Test suite for fallback logic and edge cases."""

    def setup_method(self):
        """Setup for each test method."""
        self.classifier = FileClassifier()

    def test_classify_with_fallback_no_config(self):
        """Test fallback classification without routing config."""
        result = self.classifier.classify_with_fallback("test.py")

        assert result.destination == IndexingDestination.NEO4J
        assert result.confidence == 1.0

    def test_classify_with_fallback_low_confidence(self):
        """Test fallback for low confidence classifications."""
        # Mock the classify_by_extension to return low confidence
        with patch.object(self.classifier, "classify_by_extension") as mock_classify:
            mock_classify.return_value = ClassificationResult(
                destination=IndexingDestination.QDRANT,
                confidence=0.3,  # Low confidence
                reasoning="Low confidence test",
            )

            result = self.classifier.classify_with_fallback("unknown.xyz")

            # Should default to QDRANT with fallback reasoning
            assert result.destination == IndexingDestination.QDRANT
            assert result.confidence == 0.5
            assert (
                "Low confidence, defaulting to documentation processing"
                in result.reasoning
            )

    def test_classify_with_fallback_exception(self):
        """Test fallback when classification raises an exception."""
        # Mock classify_by_extension to raise an exception
        with patch.object(self.classifier, "classify_by_extension") as mock_classify:
            mock_classify.side_effect = Exception("Test exception")

            result = self.classifier.classify_with_fallback("test.py")

            # Should return safe default
            assert result.destination == IndexingDestination.QDRANT
            assert result.confidence == 0.0
            assert "Classification error" in result.reasoning

    def test_get_supported_extensions(self):
        """Test getting supported extensions information."""
        extensions_info = self.classifier.get_supported_extensions()

        assert "rag_extensions" in extensions_info
        assert "kg_extensions" in extensions_info
        assert "total_supported" in extensions_info

        # Verify lists are sorted
        assert extensions_info["rag_extensions"] == sorted(
            extensions_info["rag_extensions"]
        )
        assert extensions_info["kg_extensions"] == sorted(
            extensions_info["kg_extensions"]
        )

        # Verify total count
        expected_total = len(self.classifier.RAG_EXTENSIONS) + len(
            self.classifier.KG_EXTENSIONS
        )
        assert extensions_info["total_supported"] == expected_total

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        # Populate cache
        self.classifier.classify_by_extension("test.py")
        self.classifier.classify_by_extension("test.md")

        cache_info_before = dict(
            self.classifier._classify_extension_cached.cache_info()._asdict()
        )
        assert cache_info_before["currsize"] > 0

        # Clear cache
        self.classifier.clear_cache()

        cache_info_after = dict(
            self.classifier._classify_extension_cached.cache_info()._asdict()
        )
        assert cache_info_after["currsize"] == 0

    def test_path_handling(self):
        """Test various path formats and edge cases."""
        path_test_cases = [
            ("/absolute/path/file.py", IndexingDestination.NEO4J),
            ("relative/path/file.md", IndexingDestination.QDRANT),
            ("./current/dir/file.js", IndexingDestination.NEO4J),
            ("../parent/dir/file.json", IndexingDestination.QDRANT),
            ("file_no_extension", IndexingDestination.QDRANT),  # Default to RAG
            ("file.", IndexingDestination.QDRANT),  # Empty extension
            (".hidden_file", IndexingDestination.QDRANT),  # Hidden file, no extension
            (".gitignore", IndexingDestination.QDRANT),  # Hidden file, no extension
        ]

        for file_path, expected_destination in path_test_cases:
            result = self.classifier.classify_by_extension(file_path)
            assert result.destination == expected_destination

    def test_performance_tracking(self):
        """Test that performance metrics are tracked properly."""
        # Perform multiple classifications
        files_to_classify = [
            "test1.py",
            "test2.py",
            "doc1.md",
            "doc2.md",
            "unknown.xyz",
        ]

        for file_path in files_to_classify:
            result = self.classifier.classify_by_extension(file_path)
            assert result.classification_time_ms >= 0  # Should measure some time

        # Check overall statistics
        stats = self.classifier.get_statistics()
        assert stats["total_classifications"] == len(files_to_classify)
        assert "cache_info" in stats

    def test_routing_config_validation(self):
        """Test IntelligentRoutingConfig validation."""
        # Valid config
        valid_config = IntelligentRoutingConfig(
            enable_intelligent_routing=True,
            force_rag_patterns=[".*test.*"],
            force_kg_patterns=[".*spec.*"],
            classification_confidence_threshold=0.9,
        )
        assert valid_config.classification_confidence_threshold == 0.9

        # Invalid confidence threshold
        with pytest.raises(ValueError):
            IntelligentRoutingConfig(classification_confidence_threshold=1.5)

        with pytest.raises(ValueError):
            IntelligentRoutingConfig(classification_confidence_threshold=-0.1)


class TestClassificationResultModel:
    """Test suite for ClassificationResult model."""

    def test_classification_result_creation(self):
        """Test ClassificationResult creation and validation."""
        result = ClassificationResult(
            destination=IndexingDestination.QDRANT,
            confidence=0.95,
            reasoning="Test classification",
            applied_overrides=["test_override"],
            classification_time_ms=1.5,
        )

        assert result.destination == IndexingDestination.QDRANT
        assert result.confidence == 0.95
        assert result.reasoning == "Test classification"
        assert result.applied_overrides == ["test_override"]
        assert result.classification_time_ms == 1.5

    def test_classification_result_validation(self):
        """Test ClassificationResult validation."""
        # Invalid confidence values
        with pytest.raises(ValueError):
            ClassificationResult(
                destination=IndexingDestination.QDRANT,
                confidence=1.5,  # > 1.0
                reasoning="Test",
            )

        with pytest.raises(ValueError):
            ClassificationResult(
                destination=IndexingDestination.QDRANT,
                confidence=-0.1,  # < 0.0
                reasoning="Test",
            )

    def test_classification_result_properties(self):
        """Test ClassificationResult computed properties."""
        # High confidence RAG result
        rag_result = ClassificationResult(
            destination=IndexingDestination.QDRANT,
            confidence=0.9,
            reasoning="High confidence RAG",
        )

        assert rag_result.is_confident is True
        assert rag_result.should_process_rag is True
        assert rag_result.should_process_kg is False

        # Low confidence KG result
        kg_result = ClassificationResult(
            destination=IndexingDestination.NEO4J,
            confidence=0.7,
            reasoning="Low confidence KG",
        )

        assert kg_result.is_confident is False
        assert kg_result.should_process_rag is False
        assert kg_result.should_process_kg is True

        # BOTH destination result
        both_result = ClassificationResult(
            destination=IndexingDestination.BOTH,
            confidence=1.0,
            reasoning="Both destinations",
        )

        assert both_result.should_process_rag is True
        assert both_result.should_process_kg is True


if __name__ == "__main__":
    pytest.main([__file__])
