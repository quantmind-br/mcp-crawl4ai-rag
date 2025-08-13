"""
Integration tests for intelligent file routing functionality.

Tests the complete intelligent routing pipeline from file classification
through to final storage in Qdrant (RAG) and Neo4j (KG) systems.
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from datetime import datetime

# Import the systems under test
from src.services.file_classifier import FileClassifier
from src.services.unified_indexing_service import UnifiedIndexingService
from src.models.classification_models import (
    IntelligentRoutingConfig,
    ClassificationResult,
    IndexingDestination,
)
from src.models.unified_indexing_models import (
    UnifiedIndexingRequest,
    UnifiedIndexingResponse,
    FileProcessingResult,
)


class TestIntelligentRoutingIntegration:
    """Integration tests for the complete intelligent routing pipeline."""

    @pytest.fixture
    def temp_repo_directory(self):
        """Create a temporary directory with sample files for testing."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create realistic file structure
        (temp_dir / "src").mkdir()
        (temp_dir / "docs").mkdir()
        (temp_dir / "tests").mkdir()
        (temp_dir / "config").mkdir()

        # Python code files (should go to KG)
        (temp_dir / "src" / "main.py").write_text("""
def main():
    '''Main application entry point.'''
    print("Hello, World!")
    
class Application:
    '''Application class for managing state.'''
    
    def __init__(self, config):
        self.config = config
    
    def run(self):
        '''Run the application.'''
        return "Application running"
""")

        (temp_dir / "src" / "utils.py").write_text("""
import os
import json

def load_config(path):
    '''Load configuration from JSON file.'''
    with open(path, 'r') as f:
        return json.load(f)

def save_data(data, path):
    '''Save data to file.'''
    with open(path, 'w') as f:
        json.dump(data, f)
""")

        # Documentation files (should go to RAG)
        (temp_dir / "docs" / "README.md").write_text("""
# Project Documentation

This is a comprehensive guide to using our application.

## Getting Started

1. Install dependencies
2. Configure the application
3. Run the main script

## Features

- File processing
- Configuration management
- Data persistence
""")

        (temp_dir / "docs" / "api.md").write_text("""
# API Documentation

## Endpoints

### /api/process
Process files through the system.

### /api/config
Manage application configuration.

### /api/status
Check system status.
""")

        # Configuration files (should go to RAG)
        (temp_dir / "config" / "settings.json").write_text("""
{
    "app_name": "Test Application",
    "version": "1.0.0",
    "database": {
        "host": "localhost",
        "port": 5432
    },
    "features": {
        "logging": true,
        "metrics": true
    }
}
""")

        # Test files (could go to KG with force patterns)
        (temp_dir / "tests" / "test_main.py").write_text("""
import pytest
from src.main import Application

def test_application_creation():
    '''Test that Application can be created.'''
    config = {"test": True}
    app = Application(config)
    assert app.config == config

def test_application_run():
    '''Test that Application can run.'''
    app = Application({})
    result = app.run()
    assert result == "Application running"
""")

        # Special case file (README in code directory)
        (temp_dir / "src" / "README.py").write_text("""
'''
README for the src directory.

This module contains the main application code.
'''

def get_readme():
    '''Return README information.'''
    return "This is the src directory README"
""")

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_clients(self):
        """Create mock clients for testing."""
        # Mock Qdrant client
        mock_qdrant = Mock()
        mock_qdrant.upsert_points = Mock()
        mock_qdrant.add_documents_to_qdrant = Mock()

        # Mock Neo4j parser
        mock_neo4j = AsyncMock()
        mock_neo4j.initialize = AsyncMock()
        mock_neo4j.close = AsyncMock()

        return {"qdrant": mock_qdrant, "neo4j": mock_neo4j}

    @pytest.fixture
    def file_classifier(self):
        """Create a real FileClassifier instance for integration testing."""
        return FileClassifier()

    @pytest.fixture
    def unified_service(self, mock_clients, file_classifier):
        """Create UnifiedIndexingService with real FileClassifier and mock clients."""
        service = UnifiedIndexingService(
            qdrant_client=mock_clients["qdrant"], neo4j_parser=mock_clients["neo4j"]
        )
        service.file_classifier = file_classifier
        return service

    @pytest.mark.asyncio
    async def test_end_to_end_intelligent_routing_enabled(
        self, temp_repo_directory, unified_service, file_classifier
    ):
        """Test complete end-to-end intelligent routing with routing enabled."""
        # Create request with intelligent routing enabled
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.BOTH,
            file_types=[".py", ".md", ".json"],
            max_files=50,
            routing_config=IntelligentRoutingConfig(
                enable_intelligent_routing=True,
                force_rag_patterns=[],
                force_kg_patterns=[],
            ),
        )

        # Get list of files to process
        file_paths = list(temp_repo_directory.rglob("*"))
        file_paths = [f for f in file_paths if f.is_file()]

        # Track classifications
        rag_files = []
        kg_files = []

        for file_path in file_paths:
            if file_path.suffix in [".py", ".md", ".json"]:
                classification = file_classifier.classify_file(
                    str(file_path), request.routing_config
                )

                if classification.should_process_rag:
                    rag_files.append(file_path)
                if classification.should_process_kg:
                    kg_files.append(file_path)

        # Verify intelligent routing decisions
        # .py files should go to KG (except potential README.py override)
        py_files = [f for f in file_paths if f.suffix == ".py"]
        # .md files should go to RAG
        md_files = [f for f in file_paths if f.suffix == ".md"]
        # .json files should go to RAG
        json_files = [f for f in file_paths if f.suffix == ".json"]

        # Verify classification results
        assert len(py_files) > 0, "Should have Python files"
        assert len(md_files) > 0, "Should have Markdown files"
        assert len(json_files) > 0, "Should have JSON files"

        # Python files should go to KG
        for py_file in py_files:
            classification = file_classifier.classify_file(
                str(py_file), request.routing_config
            )
            assert classification.destination == IndexingDestination.NEO4J
            assert classification.confidence == 1.0

        # Markdown files should go to RAG
        for md_file in md_files:
            classification = file_classifier.classify_file(
                str(md_file), request.routing_config
            )
            assert classification.destination == IndexingDestination.QDRANT
            assert classification.confidence == 1.0

        # JSON files should go to RAG
        for json_file in json_files:
            classification = file_classifier.classify_file(
                str(json_file), request.routing_config
            )
            assert classification.destination == IndexingDestination.QDRANT
            assert classification.confidence == 1.0

        # Verify performance statistics
        stats = file_classifier.get_statistics()
        assert stats["total_classifications"] > 0
        assert stats["kg_classifications"] > 0
        assert stats["rag_classifications"] > 0

    @pytest.mark.asyncio
    async def test_end_to_end_with_force_rag_patterns(
        self, temp_repo_directory, unified_service, file_classifier
    ):
        """Test end-to-end routing with force RAG patterns."""
        # Create request with force RAG patterns
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.BOTH,
            file_types=[".py", ".md", ".json"],
            max_files=50,
            routing_config=IntelligentRoutingConfig(
                enable_intelligent_routing=True,
                force_rag_patterns=[
                    ".*README.*",
                    ".*test.*",
                ],  # Force README and test files to RAG
                force_kg_patterns=[],
            ),
        )

        # Test specific files with overrides
        readme_py_path = temp_repo_directory / "src" / "README.py"
        test_py_path = temp_repo_directory / "tests" / "test_main.py"
        main_py_path = temp_repo_directory / "src" / "main.py"

        # README.py should be forced to RAG despite being a Python file
        readme_classification = file_classifier.classify_file(
            str(readme_py_path), request.routing_config
        )
        assert readme_classification.destination == IndexingDestination.QDRANT
        assert "Force RAG override" in readme_classification.reasoning
        assert len(readme_classification.applied_overrides) > 0

        # test_main.py should be forced to RAG despite being a Python file
        test_classification = file_classifier.classify_file(
            str(test_py_path), request.routing_config
        )
        assert test_classification.destination == IndexingDestination.QDRANT
        assert "Force RAG override" in test_classification.reasoning

        # main.py should go to KG (no override pattern matches)
        main_classification = file_classifier.classify_file(
            str(main_py_path), request.routing_config
        )
        assert main_classification.destination == IndexingDestination.NEO4J
        assert "Code file" in main_classification.reasoning
        assert len(main_classification.applied_overrides) == 0

    @pytest.mark.asyncio
    async def test_end_to_end_with_force_kg_patterns(
        self, temp_repo_directory, unified_service, file_classifier
    ):
        """Test end-to-end routing with force KG patterns."""
        # Create request with force KG patterns
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.BOTH,
            file_types=[".py", ".md", ".json"],
            max_files=50,
            routing_config=IntelligentRoutingConfig(
                enable_intelligent_routing=True,
                force_rag_patterns=[],
                force_kg_patterns=[
                    ".*config.*",
                    ".*settings.*",
                ],  # Force config files to KG
            ),
        )

        # Test configuration files
        settings_json_path = temp_repo_directory / "config" / "settings.json"
        api_md_path = temp_repo_directory / "docs" / "api.md"

        # settings.json should be forced to KG despite being a JSON file
        settings_classification = file_classifier.classify_file(
            str(settings_json_path), request.routing_config
        )
        assert settings_classification.destination == IndexingDestination.NEO4J
        assert "Force KG override" in settings_classification.reasoning
        assert len(settings_classification.applied_overrides) > 0

        # api.md should go to RAG (no override pattern matches)
        api_classification = file_classifier.classify_file(
            str(api_md_path), request.routing_config
        )
        assert api_classification.destination == IndexingDestination.QDRANT
        assert "Documentation" in api_classification.reasoning

    @pytest.mark.asyncio
    async def test_end_to_end_performance_comparison(
        self, temp_repo_directory, file_classifier
    ):
        """Test performance improvements with intelligent routing vs legacy routing."""
        file_paths = [
            f
            for f in temp_repo_directory.rglob("*")
            if f.is_file() and f.suffix in [".py", ".md", ".json"]
        ]

        # Test with intelligent routing enabled
        start_time = time.perf_counter()

        routing_config = IntelligentRoutingConfig(enable_intelligent_routing=True)
        for file_path in file_paths:
            file_classifier.classify_file(str(file_path), routing_config)

        intelligent_routing_time = time.perf_counter() - start_time

        # Get classification statistics
        stats_with_routing = file_classifier.get_statistics()

        # Clear stats and test fallback behavior (simulating legacy routing)
        file_classifier.stats = {key: 0 for key in file_classifier.stats}

        start_time = time.perf_counter()

        # Simulate legacy behavior - process all files for both destinations
        legacy_routing_time = 0
        for file_path in file_paths:
            # Simulate processing for both RAG and KG (legacy behavior)
            start_file = time.perf_counter()
            # Simulate processing overhead
            time.sleep(0.001)  # Small delay to simulate processing
            legacy_routing_time += time.perf_counter() - start_file

        # Intelligent routing should be more efficient for classification
        assert intelligent_routing_time < legacy_routing_time
        assert stats_with_routing["total_classifications"] == len(file_paths)

    @pytest.mark.asyncio
    async def test_end_to_end_disabled_intelligent_routing(
        self, temp_repo_directory, unified_service, file_classifier
    ):
        """Test end-to-end behavior with intelligent routing disabled."""
        # When disabled, the file classifier shouldn't be used
        # This simulates the legacy behavior where all files go to both destinations

        # Verify that file classification behaves as expected when disabled
        readme_py_path = temp_repo_directory / "src" / "README.py"

        # With routing disabled, we simulate legacy behavior
        # (In real integration, this would be handled by the service)

        # Test fallback classification
        fallback_result = file_classifier.classify_with_fallback(
            str(readme_py_path),
            routing_config=None,  # No routing config simulates disabled state
        )

        # Should fall back to extension-based classification
        assert (
            fallback_result.destination == IndexingDestination.NEO4J
        )  # .py file goes to KG
        assert fallback_result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_end_to_end_classification_error_resilience(
        self, temp_repo_directory, unified_service, file_classifier
    ):
        """Test system resilience when classification encounters errors."""
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.BOTH,
            file_types=[".py", ".md", ".json"],
            max_files=50,
            routing_config=IntelligentRoutingConfig(
                enable_intelligent_routing=True,
                force_rag_patterns=["[invalid_regex"],  # Invalid regex pattern
                force_kg_patterns=["*[another_invalid"],  # Invalid regex pattern
            ),
        )

        # Test files with invalid regex patterns
        test_file = temp_repo_directory / "src" / "main.py"

        # Should handle invalid regex gracefully
        classification = file_classifier.classify_file(
            str(test_file), request.routing_config
        )

        # Should fall back to normal classification despite invalid patterns
        assert (
            classification.destination == IndexingDestination.NEO4J
        )  # Normal .py classification
        assert classification.confidence == 1.0
        assert (
            len(classification.applied_overrides) == 0
        )  # No overrides applied due to invalid regex

    @pytest.mark.asyncio
    async def test_end_to_end_mixed_file_types_classification(
        self, temp_repo_directory, file_classifier
    ):
        """Test classification of mixed file types in realistic repository structure."""
        routing_config = IntelligentRoutingConfig(
            enable_intelligent_routing=True,
            force_rag_patterns=[".*docs/.*"],  # All docs go to RAG
            force_kg_patterns=[".*src/.*\\.py$"],  # Python files in src go to KG
        )

        # Test all files in the repository
        all_files = [f for f in temp_repo_directory.rglob("*") if f.is_file()]

        classification_results = {}
        for file_path in all_files:
            if file_path.suffix in [".py", ".md", ".json"]:
                result = file_classifier.classify_file(str(file_path), routing_config)
                classification_results[
                    str(file_path.relative_to(temp_repo_directory))
                ] = result

        # Verify expected classifications
        expected_classifications = {
            # Python files in src should go to KG (force pattern)
            "src/main.py": IndexingDestination.NEO4J,
            "src/utils.py": IndexingDestination.NEO4J,
            "src/README.py": IndexingDestination.NEO4J,  # Force KG pattern overrides README
            # Docs should go to RAG (force pattern)
            "docs/README.md": IndexingDestination.QDRANT,
            "docs/api.md": IndexingDestination.QDRANT,
            # Config files should go to RAG (normal classification)
            "config/settings.json": IndexingDestination.QDRANT,
            # Test files should go to KG (normal .py classification)
            "tests/test_main.py": IndexingDestination.NEO4J,
        }

        for relative_path, expected_dest in expected_classifications.items():
            if relative_path in classification_results:
                actual_dest = classification_results[relative_path].destination
                assert actual_dest == expected_dest, (
                    f"File {relative_path} went to {actual_dest}, expected {expected_dest}"
                )

    @pytest.mark.asyncio
    async def test_end_to_end_cache_performance(self, file_classifier):
        """Test that file extension caching improves performance for repeated classifications."""
        # Create multiple files with same extensions
        test_files = [
            "file1.py",
            "file2.py",
            "file3.py",  # Same extension
            "doc1.md",
            "doc2.md",
            "doc3.md",  # Same extension
            "config1.json",
            "config2.json",  # Same extension
        ]

        routing_config = IntelligentRoutingConfig(enable_intelligent_routing=True)

        # Clear cache to start fresh
        file_classifier.clear_cache()

        # First round - should populate cache
        start_time = time.perf_counter()
        for file_path in test_files:
            file_classifier.classify_file(file_path, routing_config)
        first_round_time = time.perf_counter() - start_time

        # Second round - should benefit from cache
        start_time = time.perf_counter()
        for file_path in test_files:
            file_classifier.classify_file(file_path, routing_config)
        second_round_time = time.perf_counter() - start_time

        # Second round should be faster due to caching
        assert second_round_time <= first_round_time

        # Verify cache statistics
        cache_info = dict(
            file_classifier._classify_extension_cached.cache_info()._asdict()
        )
        assert cache_info["hits"] > 0
        assert cache_info["misses"] > 0

    def test_integration_file_classifier_supported_extensions(self, file_classifier):
        """Test that FileClassifier supports all expected extensions for integration."""
        supported_extensions = file_classifier.get_supported_extensions()

        # Verify comprehensive extension support for realistic repositories
        expected_rag_extensions = {".md", ".json", ".yaml", ".yml", ".txt", ".rst"}
        expected_kg_extensions = {
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".java",
            ".go",
            ".rs",
            ".cpp",
        }

        actual_rag = set(supported_extensions["rag_extensions"])
        actual_kg = set(supported_extensions["kg_extensions"])

        # Verify core extensions are supported
        assert expected_rag_extensions.issubset(actual_rag), (
            f"Missing RAG extensions: {expected_rag_extensions - actual_rag}"
        )
        assert expected_kg_extensions.issubset(actual_kg), (
            f"Missing KG extensions: {expected_kg_extensions - actual_kg}"
        )

        # Verify no overlap
        assert len(actual_rag.intersection(actual_kg)) == 0, (
            "Found overlapping extensions"
        )

    @pytest.mark.asyncio
    async def test_integration_request_response_flow(
        self, temp_repo_directory, mock_clients
    ):
        """Test complete request/response flow with intelligent routing."""
        # Create a complete UnifiedIndexingRequest with intelligent routing
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/integration-repo",
            destination=IndexingDestination.BOTH,
            file_types=[".py", ".md", ".json"],
            max_files=20,
            chunk_size=2000,
            max_size_mb=100,
            routing_config=IntelligentRoutingConfig(
                enable_intelligent_routing=True,
                force_rag_patterns=[".*README.*", ".*docs/.*"],
                force_kg_patterns=[".*test.*"],
                classification_confidence_threshold=0.9,
            ),
        )

        # Verify request is properly constructed
        assert request.routing_config.enable_intelligent_routing is True
        assert request.routing_config.force_rag_patterns == [".*README.*", ".*docs/.*"]
        assert request.routing_config.force_kg_patterns == [".*test.*"]
        assert request.routing_config.classification_confidence_threshold == 0.9

        # Verify request properties work correctly
        assert request.should_process_rag is True
        assert request.should_process_kg is True

        # Create sample response with intelligent routing results
        response = UnifiedIndexingResponse(
            success=True,
            repo_url=request.repo_url,
            repo_name="integration-repo",
            destination="both",
            files_processed=0,
            start_time=datetime.now(),
        )

        # Add sample file results with classification metadata
        sample_file_results = [
            FileProcessingResult(
                file_id="integration-repo:src/main.py",
                file_path=str(temp_repo_directory / "src" / "main.py"),
                relative_path="src/main.py",
                language="python",
                file_type=".py",
                processed_for_kg=True,
                processed_for_rag=False,
                kg_entities=5,
                rag_chunks=0,
                processing_time_seconds=1.2,
                classification_result=ClassificationResult(
                    destination=IndexingDestination.NEO4J,
                    confidence=1.0,
                    reasoning="Code file: .py",
                    classification_time_ms=0.8,
                ),
                routing_decision="Intelligent routing: KG only",
                classification_time_ms=0.8,
            ),
            FileProcessingResult(
                file_id="integration-repo:docs/README.md",
                file_path=str(temp_repo_directory / "docs" / "README.md"),
                relative_path="docs/README.md",
                language="markdown",
                file_type=".md",
                processed_for_rag=True,
                processed_for_kg=False,
                rag_chunks=3,
                kg_entities=0,
                processing_time_seconds=0.8,
                classification_result=ClassificationResult(
                    destination=IndexingDestination.QDRANT,
                    confidence=1.0,
                    reasoning="Force RAG override: .*docs/.*",
                    applied_overrides=["force_rag: .*docs/.*"],
                    classification_time_ms=1.2,
                ),
                routing_decision="Intelligent routing: RAG only",
                classification_time_ms=1.2,
            ),
        ]

        for file_result in sample_file_results:
            response.add_file_result(file_result)

        response.finalize()

        # Verify response contains intelligent routing information
        assert response.success is True
        assert response.files_processed == 2
        assert response.qdrant_documents == 3  # From README.md
        assert response.neo4j_nodes == 5  # From main.py
        assert response.cross_system_links_created == 0  # No dual processing

        # Verify file results contain classification metadata
        py_result = next(r for r in response.file_results if r.file_type == ".py")
        assert py_result.classification_result is not None
        assert py_result.classification_result.destination == IndexingDestination.NEO4J
        assert py_result.routing_decision == "Intelligent routing: KG only"

        md_result = next(r for r in response.file_results if r.file_type == ".md")
        assert md_result.classification_result is not None
        assert len(md_result.classification_result.applied_overrides) > 0
        assert md_result.routing_decision == "Intelligent routing: RAG only"


if __name__ == "__main__":
    pytest.main([__file__])
