"""
Tests for Neo4j batch processing functionality in unified indexing service.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.services.unified_indexing_service import (
    UnifiedIndexingService,
    UnifiedIndexingRequest,
    IndexingDestination,
)


class TestNeo4jBatchProcessing:
    """Test cases for Neo4j batch processing functionality."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        return Mock()

    @pytest.fixture
    def mock_neo4j_parser(self):
        """Create a mock Neo4j parser."""
        parser = Mock()
        parser.initialize = AsyncMock()
        parser._create_graph = AsyncMock()
        parser.driver = Mock()
        return parser

    @pytest.fixture
    def unified_indexing_service(self, mock_qdrant_client, mock_neo4j_parser):
        """Create a UnifiedIndexingService instance."""
        return UnifiedIndexingService(
            qdrant_client=mock_qdrant_client, neo4j_parser=mock_neo4j_parser
        )

    @pytest.mark.asyncio
    async def test_batch_process_neo4j_analyses_success(self, unified_indexing_service):
        """Test successful Neo4j batch processing."""
        # Set up mock analyses data
        unified_indexing_service._neo4j_analyses = [
            {
                "file_id": "repo:file1.py",
                "classes": [
                    {
                        "name": "TestClass",
                        "methods": [
                            {"name": "method1", "params": []},
                            {"name": "method2", "params": ["self", "arg1"]},
                        ],
                    }
                ],
                "functions": [{"name": "test_function", "params": ["arg1", "arg2"]}],
            },
            {
                "file_id": "repo:file2.py",
                "classes": [],
                "functions": [{"name": "another_function", "params": ["x", "y"]}],
            },
        ]
        unified_indexing_service._neo4j_repo_name = "test-repo"

        # Mock session and transaction behavior
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_record = Mock()
        mock_record.__getitem__ = Mock(
            side_effect=lambda key: 1 if key == "count" else None
        )
        mock_result.single = AsyncMock(return_value=mock_record)
        mock_session.run = AsyncMock(return_value=mock_result)

        # Mock session context manager
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)
        unified_indexing_service.neo4j_parser.driver.session.return_value = (
            mock_session_cm
        )

        # Call the method
        result = await unified_indexing_service._batch_process_neo4j_analyses()

        # Verify results
        assert result is not None
        assert result["entities_created"] == 5  # 1 class + 2 methods + 2 functions = 5
        assert result["files_processed"] == 2

        # Verify _create_graph was called with correct parameters
        # Need to capture the call args before cleanup happens
        call_args = unified_indexing_service.neo4j_parser._create_graph.call_args
        assert call_args is not None
        assert call_args[1]["repo_name"] == "test-repo"
        assert len(call_args[1]["modules_data"]) == 2

        # Verify analyses were cleaned up
        assert not hasattr(unified_indexing_service, "_neo4j_analyses")
        assert not hasattr(unified_indexing_service, "_neo4j_repo_name")

    @pytest.mark.asyncio
    async def test_batch_process_neo4j_analyses_no_data(self, unified_indexing_service):
        """Test batch processing with no analyses data."""
        # No analyses data set
        result = await unified_indexing_service._batch_process_neo4j_analyses()

        # Should return None and not call any Neo4j operations
        assert result is None
        unified_indexing_service.neo4j_parser._create_graph.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_process_neo4j_analyses_empty_list(
        self, unified_indexing_service
    ):
        """Test batch processing with empty analyses list."""
        # Set empty analyses list
        unified_indexing_service._neo4j_analyses = []

        result = await unified_indexing_service._batch_process_neo4j_analyses()

        # Should return None and not call any Neo4j operations
        assert result is None
        unified_indexing_service.neo4j_parser._create_graph.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_process_neo4j_analyses_with_exception(
        self, unified_indexing_service
    ):
        """Test batch processing with exception handling."""
        # Set up mock analyses data
        unified_indexing_service._neo4j_analyses = [{"test": "data"}]
        unified_indexing_service._neo4j_repo_name = "test-repo"

        # Make _create_graph raise an exception
        unified_indexing_service.neo4j_parser._create_graph.side_effect = Exception(
            "Neo4j error"
        )

        # Should re-raise the exception (as per corrected implementation)
        with pytest.raises(Exception) as exc_info:
            await unified_indexing_service._batch_process_neo4j_analyses()

        assert "Neo4j error" in str(exc_info.value)

        # Verify cleanup still happened
        assert not hasattr(unified_indexing_service, "_neo4j_analyses")
        assert not hasattr(unified_indexing_service, "_neo4j_repo_name")

    @pytest.mark.asyncio
    async def test_batch_process_neo4j_analyses_verification_queries(
        self, unified_indexing_service
    ):
        """Test that verification queries are executed correctly."""
        # Set up mock analyses data
        unified_indexing_service._neo4j_analyses = [{"test": "data"}]
        unified_indexing_service._neo4j_repo_name = "test-repo"

        # Mock session and queries
        mock_session = AsyncMock()

        # Mock repository count query
        repo_result = Mock()
        repo_record = Mock()
        repo_record.__getitem__ = Mock(return_value=1)  # Return 1 for "count"
        repo_result.single = AsyncMock(return_value=repo_record)

        # Mock total nodes query
        nodes_result = Mock()
        nodes_record = Mock()
        nodes_record.__getitem__ = Mock(return_value=100)  # Return 100 for "count"
        nodes_result.single = AsyncMock(return_value=nodes_record)

        # Mock session.run to return different results for different queries
        def mock_run(query, **kwargs):
            if "Repository" in query:
                return repo_result
            elif "MATCH (n)" in query:
                return nodes_result
            return Mock()

        mock_session.run = AsyncMock(side_effect=mock_run)

        # Mock session context manager
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)
        unified_indexing_service.neo4j_parser.driver.session.return_value = (
            mock_session_cm
        )

        # Call the method
        result = await unified_indexing_service._batch_process_neo4j_analyses()

        # Verify verification queries were executed
        assert mock_session.run.call_count >= 2  # At least repo and nodes count queries

        # Check that queries contained expected patterns
        call_args_list = [call[0][0] for call in mock_session.run.call_args_list]
        assert any("Repository" in query for query in call_args_list)
        assert any("MATCH (n)" in query for query in call_args_list)

    def test_batch_process_entity_counting(self, unified_indexing_service):
        """Test entity counting logic in batch processing."""
        # Test the entity counting logic used in _batch_process_neo4j_analyses
        analyses = [
            {
                "classes": [
                    {
                        "name": "Class1",
                        "methods": [{"name": "method1"}, {"name": "method2"}],
                    },
                    {"name": "Class2", "methods": [{"name": "method3"}]},
                ],
                "functions": [{"name": "func1"}, {"name": "func2"}],
            },
            {"classes": [], "functions": [{"name": "func3"}]},
        ]

        # Calculate expected counts
        total_classes = sum(len(analysis.get("classes", [])) for analysis in analyses)
        total_functions = sum(
            len(analysis.get("functions", [])) for analysis in analyses
        )
        total_methods = sum(
            len(cls.get("methods", []))
            for analysis in analyses
            for cls in analysis.get("classes", [])
        )

        assert total_classes == 2  # Class1, Class2
        assert total_functions == 3  # func1, func2, func3
        assert total_methods == 3  # method1, method2, method3

        total_entities = total_classes + total_functions + total_methods
        assert total_entities == 8

    @pytest.mark.asyncio
    async def test_process_repository_unified_integrates_neo4j_results(
        self, unified_indexing_service
    ):
        """Test that the main processing method properly integrates Neo4j batch results."""
        request = UnifiedIndexingRequest(
            repo_url="https://github.com/test/repo",
            destination=IndexingDestination.BOTH,
            file_types=[".py"],
            max_files=5,
        )

        # Mock internal methods
        with (
            patch.object(unified_indexing_service, "_clone_repository"),
            patch.object(unified_indexing_service, "_discover_repository_files"),
            patch.object(unified_indexing_service, "_process_files_unified"),
            patch.object(
                unified_indexing_service, "_batch_process_neo4j_analyses"
            ) as mock_batch,
        ):
            # Mock all necessary return values
            unified_indexing_service._clone_repository = AsyncMock(return_value=Mock())
            unified_indexing_service._discover_repository_files = AsyncMock(
                return_value=[Mock()]
            )

            # Mock file processing as async generator function
            async def mock_process_files_unified(*args, **kwargs):
                # Simulate analyses being created during file processing
                unified_indexing_service._neo4j_analyses = [{"test": "data"}]
                unified_indexing_service._neo4j_repo_name = "test-repo"

                # Create a proper FileProcessingResult mock with numeric values
                from src.services.unified_indexing_service import FileProcessingResult

                result = FileProcessingResult(
                    file_id="test:file.py",
                    relative_path="test/file.py",
                    processed_for_kg=True,
                    kg_entities=5,
                    processed_for_rag=True,
                    rag_chunks=3,
                )
                yield result

            unified_indexing_service._process_files_unified = mock_process_files_unified

            # Mock batch processing result
            batch_result = {
                "entities_created": 150,
                "files_processed": 1,
                "relationships_created": 1,
            }
            mock_batch.return_value = batch_result

            # Call the method
            response = await unified_indexing_service.process_repository_unified(
                request
            )

            # Verify Neo4j results were integrated into response
            assert response.neo4j_nodes == 150
            mock_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_neo4j_lazy_initialization(self, mock_qdrant_client):
        """Test that Neo4j parser is initialized lazily when needed."""
        import os

        # Test lazy initialization
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "bolt://test:7687",
                "NEO4J_USER": "test_user",
                "NEO4J_PASSWORD": "test_password",
            },
        ):
            with patch(
                "src.services.unified_indexing_service.DirectNeo4jExtractor"
            ) as mock_extractor:
                mock_parser = Mock()
                mock_parser.initialize = AsyncMock()
                mock_extractor.return_value = mock_parser

                # Create service without Neo4j parser
                service = UnifiedIndexingService(qdrant_client=mock_qdrant_client)

                # Should have config but no parser yet
                assert service._neo4j_config is not None
                assert service.neo4j_parser is None

                # Create request that needs Neo4j
                request = UnifiedIndexingRequest(
                    repo_url="https://github.com/test/repo",
                    destination=IndexingDestination.NEO4J,
                )

                # Mock other methods to avoid full processing
                with (
                    patch.object(service, "_clone_repository", new_callable=AsyncMock),
                    patch.object(
                        service, "_discover_repository_files", new_callable=AsyncMock
                    ),
                    patch.object(
                        service, "_process_files_unified", new_callable=AsyncMock
                    ),
                ):
                    service._clone_repository.return_value = Mock()
                    service._discover_repository_files.return_value = []

                    # Process - should initialize Neo4j parser
                    await service.process_repository_unified(request)

                    # Verify parser was created and initialized
                    mock_extractor.assert_called_once_with(
                        "bolt://test:7687", "test_user", "test_password"
                    )
                    mock_parser.initialize.assert_called_once()
                    assert service.neo4j_parser == mock_parser
