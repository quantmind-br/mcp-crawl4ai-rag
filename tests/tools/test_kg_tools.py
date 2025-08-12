"""
Tests for Knowledge Graph tools.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from src.tools.kg_tools import (
    validate_script_path,
    parse_github_repository,
    check_ai_script_hallucinations,
    query_knowledge_graph,
)


class TestKGTools:
    """Test cases for Knowledge Graph tools."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock MCP context."""
        context = Mock()
        context.request_context = Mock()
        context.request_context.lifespan_context = Mock()
        return context

    def test_validate_script_path_valid(self):
        """Test validate_script_path with valid path."""
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open"):
                result = validate_script_path("/valid/path/test.py")
                assert result["valid"] is True

    def test_validate_script_path_invalid_extension(self):
        """Test validate_script_path with invalid file extension."""
        with patch("os.path.exists", return_value=True):
            result = validate_script_path("/path/test.txt")
            assert result["valid"] is False
            assert "Only Python" in result["error"]

    def test_validate_script_path_not_exists(self):
        """Test validate_script_path with non-existent file."""
        with patch("os.path.exists", return_value=False):
            result = validate_script_path("/nonexistent/test.py")
            assert result["valid"] is False
            assert "not found" in result["error"]

    def test_validate_script_path_invalid_permissions(self):
        """Test validate_script_path with file that can't be read."""
        with patch("os.path.exists", return_value=True):
            with patch(
                "builtins.open", side_effect=PermissionError("Permission denied")
            ):
                result = validate_script_path("/restricted/test.py")
                assert result["valid"] is False
                assert "Cannot read" in result["error"]

    @pytest.mark.asyncio
    async def test_parse_github_repository_kg_disabled(self, mock_context):
        """Test parse_github_repository when KG is disabled."""
        with patch.dict("os.environ", {"USE_KNOWLEDGE_GRAPH": "false"}):
            result = await parse_github_repository(
                mock_context, "https://github.com/user/repo"
            )
            result_data = json.loads(result)

            assert result_data["success"] is False
            assert "disabled" in result_data["error"]

    @pytest.mark.asyncio
    async def test_parse_github_repository_no_extractor(self, mock_context):
        """Test parse_github_repository when extractor is not available."""
        with patch.dict("os.environ", {"USE_KNOWLEDGE_GRAPH": "true"}):
            mock_context.request_context.lifespan_context.repo_extractor = None

            result = await parse_github_repository(
                mock_context, "https://github.com/user/repo"
            )
            result_data = json.loads(result)

            assert result_data["success"] is False
            assert "extractor not available" in result_data["error"]

    @pytest.mark.asyncio
    async def test_parse_github_repository_invalid_url(self, mock_context):
        """Test parse_github_repository with invalid URL."""
        with patch.dict("os.environ", {"USE_KNOWLEDGE_GRAPH": "true"}):
            mock_context.request_context.lifespan_context.repo_extractor = Mock()

            result = await parse_github_repository(
                mock_context, "https://invalid-url.com"
            )
            result_data = json.loads(result)

            assert result_data["success"] is False
            assert "github.com" in result_data["error"]

    @pytest.mark.asyncio
    async def test_parse_github_repository_success(self, mock_context):
        """Test successful parse_github_repository execution."""
        with patch.dict("os.environ", {"USE_KNOWLEDGE_GRAPH": "true"}):
            with patch(
                "src.tools.kg_tools.validate_github_url", return_value=(True, None)
            ):
                # Mock extractor
                mock_extractor = AsyncMock()
                mock_extractor.driver = Mock()
                mock_context.request_context.lifespan_context.repo_extractor = (
                    mock_extractor
                )

                # Mock session and query results
                mock_session = AsyncMock()
                mock_async_context_manager = AsyncMock()
                mock_async_context_manager.__aenter__.return_value = mock_session
                mock_extractor.driver.session.return_value = mock_async_context_manager

                mock_record = Mock()
                mock_record.get.side_effect = lambda key, default=None: {
                    "repo_name": "repo",
                    "files_count": 5,
                    "classes_count": 10,
                    "methods_count": 25,
                    "functions_count": 15,
                    "attributes_count": 20,
                    "sample_modules": ["module1", "module2"],
                }.get(key, default)

                mock_session.run.return_value.single.return_value = mock_record

                result = await parse_github_repository(
                    mock_context, "https://github.com/user/repo"
                )
                result_data = json.loads(result)

                assert result_data["success"] is True
                assert result_data["repo_url"] == "https://github.com/user/repo"
                assert "statistics" in result_data

    @pytest.mark.asyncio
    async def test_check_ai_script_hallucinations_kg_disabled(self, mock_context):
        """Test check_ai_script_hallucinations when KG is disabled."""
        with patch.dict("os.environ", {"USE_KNOWLEDGE_GRAPH": "false"}):
            result = await check_ai_script_hallucinations(
                mock_context, "/test/script.py"
            )
            result_data = json.loads(result)

            assert result_data["success"] is False
            assert "disabled" in result_data["error"]

    @pytest.mark.asyncio
    async def test_check_ai_script_hallucinations_no_validator(self, mock_context):
        """Test check_ai_script_hallucinations when validator is not available."""
        with patch.dict("os.environ", {"USE_KNOWLEDGE_GRAPH": "true"}):
            mock_context.request_context.lifespan_context.knowledge_validator = None

            result = await check_ai_script_hallucinations(
                mock_context, "/test/script.py"
            )
            result_data = json.loads(result)

            assert result_data["success"] is False
            assert "validator not available" in result_data["error"]

    @pytest.mark.asyncio
    async def test_check_ai_script_hallucinations_invalid_path(self, mock_context):
        """Test check_ai_script_hallucinations with invalid script path."""
        with patch.dict("os.environ", {"USE_KNOWLEDGE_GRAPH": "true"}):
            mock_context.request_context.lifespan_context.knowledge_validator = Mock()

            with patch(
                "src.tools.kg_tools.validate_script_path",
                return_value={"valid": False, "error": "Invalid path"},
            ):
                result = await check_ai_script_hallucinations(
                    mock_context, "/invalid/script.py"
                )
                result_data = json.loads(result)

                assert result_data["success"] is False
                assert "Invalid path" in result_data["error"]

    @pytest.mark.asyncio
    async def test_check_ai_script_hallucinations_success(self, mock_context):
        """Test successful check_ai_script_hallucinations execution."""
        with patch.dict("os.environ", {"USE_KNOWLEDGE_GRAPH": "true"}):
            with patch(
                "src.tools.kg_tools.validate_script_path", return_value={"valid": True}
            ):
                # Mock validator and analyzer
                mock_validator = AsyncMock()
                mock_context.request_context.lifespan_context.knowledge_validator = (
                    mock_validator
                )

                mock_analyzer = Mock()
                mock_analysis_result = Mock()
                mock_analysis_result.errors = []
                mock_analyzer.analyze_script.return_value = mock_analysis_result

                with patch(
                    "src.tools.kg_tools.AIScriptAnalyzer", return_value=mock_analyzer
                ):
                    # Mock validation result
                    mock_validation_result = Mock()
                    mock_validation_result.overall_confidence = 0.95

                    mock_validator.validate_script.return_value = mock_validation_result

                    # Mock reporter
                    mock_reporter = Mock()
                    mock_reporter.generate_comprehensive_report.return_value = {
                        "validation_summary": {
                            "total_validations": 10,
                            "valid_count": 8,
                            "invalid_count": 1,
                            "uncertain_count": 1,
                            "not_found_count": 0,
                            "hallucination_rate": 0.1,
                        },
                        "hallucinations_detected": True,
                        "recommendations": ["Fix import issues"],
                        "analysis_metadata": {
                            "total_imports": 5,
                            "total_classes": 3,
                            "total_methods": 10,
                            "total_attributes": 5,
                            "total_functions": 2,
                        },
                        "libraries_analyzed": ["library1", "library2"],
                    }

                    with patch(
                        "src.tools.kg_tools.HallucinationReporter",
                        return_value=mock_reporter,
                    ):
                        result = await check_ai_script_hallucinations(
                            mock_context, "/test/script.py"
                        )
                        result_data = json.loads(result)

                        assert result_data["success"] is True
                        assert result_data["script_path"] == "/test/script.py"
                        assert "overall_confidence" in result_data
                        assert "validation_summary" in result_data

    @pytest.mark.asyncio
    async def test_query_knowledge_graph_kg_disabled(self, mock_context):
        """Test query_knowledge_graph when KG is disabled."""
        with patch.dict("os.environ", {"USE_KNOWLEDGE_GRAPH": "false"}):
            result = await query_knowledge_graph(mock_context, "repos")
            result_data = json.loads(result)

            assert result_data["success"] is False
            assert "disabled" in result_data["error"]

    @pytest.mark.asyncio
    async def test_query_knowledge_graph_no_driver(self, mock_context):
        """Test query_knowledge_graph when driver is not available."""
        with patch.dict("os.environ", {"USE_KNOWLEDGE_GRAPH": "true"}):
            mock_context.request_context.lifespan_context.repo_extractor = Mock()
            mock_context.request_context.lifespan_context.repo_extractor.driver = None

            result = await query_knowledge_graph(mock_context, "repos")
            result_data = json.loads(result)

            assert result_data["success"] is False
            assert "connection not available" in result_data["error"]

    @pytest.mark.asyncio
    async def test_query_knowledge_graph_empty_command(self, mock_context):
        """Test query_knowledge_graph with empty command."""
        with patch.dict("os.environ", {"USE_KNOWLEDGE_GRAPH": "true"}):
            mock_context.request_context.lifespan_context.repo_extractor = Mock()
            mock_context.request_context.lifespan_context.repo_extractor.driver = Mock()

            result = await query_knowledge_graph(mock_context, "")
            result_data = json.loads(result)

            assert result_data["success"] is False
            assert "Command cannot be empty" in result_data["error"]

    @pytest.mark.asyncio
    async def test_query_knowledge_graph_repos_command(self, mock_context):
        """Test query_knowledge_graph with repos command."""
        with patch.dict("os.environ", {"USE_KNOWLEDGE_GRAPH": "true"}):
            # Mock extractor and driver
            mock_extractor = Mock()
            mock_extractor.driver = Mock()
            mock_context.request_context.lifespan_context.repo_extractor = (
                mock_extractor
            )

            # Mock session and query results
            mock_session = AsyncMock()
            mock_async_context_manager = AsyncMock()
            mock_async_context_manager.__aenter__.return_value = mock_session
            mock_extractor.driver.session.return_value = mock_async_context_manager

            mock_record = Mock()
            mock_record.get.return_value = "test-repo"
            mock_session.run.return_value.__aiter__.return_value = [mock_record]

            result = await query_knowledge_graph(mock_context, "repos")
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["command"] == "repos"
            assert "data" in result_data
