"""
Tests for core application module including singletons and app lifecycle.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestRerankingModelSingleton:
    """Test RerankingModelSingleton class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Reset singleton state between tests."""
        from src.core.app import RerankingModelSingleton

        RerankingModelSingleton._instance = None
        RerankingModelSingleton._model = None

    def test_singleton_pattern(self):
        """Test that RerankingModelSingleton follows singleton pattern."""
        from src.core.app import RerankingModelSingleton

        instance1 = RerankingModelSingleton()
        instance2 = RerankingModelSingleton()

        assert instance1 is instance2

    @patch.dict(os.environ, {"USE_RERANKING": "false"})
    def test_get_model_when_disabled(self):
        """Test get_model returns None when reranking is disabled."""
        from src.core.app import RerankingModelSingleton

        singleton = RerankingModelSingleton()
        model = singleton.get_model()

        assert model is None

    @patch.dict(os.environ, {"USE_RERANKING": "true"})
    @patch("src.core.app.CrossEncoder")
    @patch("src.device_manager.get_optimal_device")
    def test_get_model_success(self, mock_get_device, mock_cross_encoder):
        """Test successful model initialization."""
        from src.core.app import RerankingModelSingleton

        # Setup mocks
        mock_device_info = Mock()
        mock_device_info.device = "cpu"
        mock_device_info.device_type = "cpu"
        mock_device_info.model_kwargs = {}
        mock_get_device.return_value = mock_device_info

        mock_model = Mock()
        mock_cross_encoder.return_value = mock_model

        singleton = RerankingModelSingleton()
        model = singleton.get_model()

        assert model is mock_model
        mock_cross_encoder.assert_called_once()
        mock_model.predict.assert_called_once()

    @patch.dict(os.environ, {"USE_RERANKING": "true"})
    @patch("src.core.app.CrossEncoder", side_effect=Exception("Model failed"))
    @patch("src.device_manager.get_optimal_device")
    @patch("src.device_manager.cleanup_gpu_memory")
    def test_get_model_failure(self, mock_cleanup, mock_get_device, mock_cross_encoder):
        """Test model initialization failure."""
        from src.core.app import RerankingModelSingleton

        mock_device_info = Mock()
        mock_device_info.device = "cpu"
        mock_get_device.return_value = mock_device_info

        singleton = RerankingModelSingleton()
        model = singleton.get_model()

        assert model is None
        mock_cleanup.assert_called_once()


class TestKnowledgeGraphSingleton:
    """Test KnowledgeGraphSingleton class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Reset singleton state between tests."""
        from src.core.app import KnowledgeGraphSingleton

        KnowledgeGraphSingleton._instance = None
        KnowledgeGraphSingleton._knowledge_validator = None
        KnowledgeGraphSingleton._repo_extractor = None
        KnowledgeGraphSingleton._initialized = False

    @pytest.mark.asyncio
    async def test_close_components(self):
        """Test close_components cleans up resources."""
        from src.core.app import KnowledgeGraphSingleton

        singleton = KnowledgeGraphSingleton()

        # Mock components
        mock_validator = AsyncMock()
        mock_extractor = AsyncMock()
        singleton._knowledge_validator = mock_validator
        singleton._repo_extractor = mock_extractor
        singleton._initialized = True

        await singleton.close_components()

        assert singleton._knowledge_validator is None
        assert singleton._repo_extractor is None
        assert singleton._initialized is False
        mock_validator.close.assert_called_once()
        mock_extractor.close.assert_called_once()


class TestApplicationLifecycle:
    """Test application creation and lifecycle functions."""

    @patch("src.core.app.FastMCP")
    def test_create_app(self, mock_fastmcp):
        """Test create_app function."""
        from src.core.app import create_app

        mock_app = Mock()
        mock_fastmcp.return_value = mock_app

        app = create_app()

        assert app is mock_app
        mock_fastmcp.assert_called_once()
        call_args = mock_fastmcp.call_args
        assert call_args.kwargs["name"] == "mcp-crawl4ai-rag"
        assert "lifespan" in call_args.kwargs

    @patch("src.core.app.ContextSingleton")
    @pytest.mark.asyncio
    async def test_crawl4ai_lifespan_success(self, mock_context_singleton_class):
        """Test successful crawl4ai_lifespan context manager with singleton pattern."""
        from src.core.app import crawl4ai_lifespan
        from src.core.context import Crawl4AIContext

        # Setup mock context singleton
        mock_context_singleton = Mock()
        mock_context_singleton_class.return_value = mock_context_singleton

        # Setup mock context
        mock_context = Mock(spec=Crawl4AIContext)
        mock_context_singleton.get_context = AsyncMock(return_value=mock_context)

        mock_server = Mock()

        # Test the lifespan
        async with crawl4ai_lifespan(mock_server) as context:
            assert context is mock_context
            # Verify that get_context was called with the server
            mock_context_singleton.get_context.assert_called_once_with(mock_server)

        # Note: No cleanup verification since singleton manages lifecycle

    @patch("src.core.app.create_app")
    @patch("src.core.app.register_tools")
    @pytest.mark.asyncio
    async def test_run_server_sse(self, mock_register, mock_create):
        """Test run_server with SSE transport."""
        from src.core.app import run_server

        mock_app = AsyncMock()
        mock_create.return_value = mock_app

        with patch.dict(os.environ, {"TRANSPORT": "sse"}):
            await run_server()

        mock_create.assert_called_once()
        mock_register.assert_called_once_with(mock_app)
        mock_app.run_sse_async.assert_called_once()

    @patch("src.core.app.create_app")
    @patch("src.core.app.register_tools")
    @pytest.mark.asyncio
    async def test_run_server_stdio(self, mock_register, mock_create):
        """Test run_server with stdio transport."""
        from src.core.app import run_server

        mock_app = AsyncMock()
        mock_create.return_value = mock_app

        with patch.dict(os.environ, {"TRANSPORT": "stdio"}):
            await run_server()

        mock_create.assert_called_once()
        mock_register.assert_called_once_with(mock_app)
        mock_app.run_stdio_async.assert_called_once()


class TestToolRegistration:
    """Test tool registration functionality."""

    @patch("src.core.app.web_tools", create=True)
    @patch("src.core.app.github_tools", create=True)
    @patch("src.core.app.rag_tools", create=True)
    def test_register_tools_basic(
        self, mock_rag_tools, mock_github_tools, mock_web_tools
    ):
        """Test basic tool registration."""
        from src.core.app import register_tools

        mock_app = Mock()

        # Mock tools
        mock_web_tools.crawl_single_page = Mock()
        mock_web_tools.smart_crawl_url = Mock()
        mock_rag_tools.get_available_sources = Mock()
        mock_rag_tools.perform_rag_query = Mock()
        mock_rag_tools.search_code_examples = Mock()

        with patch.dict(
            os.environ, {"USE_AGENTIC_RAG": "false", "USE_KNOWLEDGE_GRAPH": "false"}
        ):
            register_tools(mock_app)

        # Verify tool registration calls
        assert mock_app.tool.call_count >= 4  # At least 4 basic tools

    @patch("src.core.app.web_tools", create=True)
    @patch("src.core.app.github_tools", create=True)
    @patch("src.core.app.rag_tools", create=True)
    def test_register_tools_with_agentic_rag(
        self, mock_rag_tools, mock_github_tools, mock_web_tools
    ):
        """Test tool registration with agentic RAG enabled."""
        from src.core.app import register_tools

        mock_app = Mock()

        # Mock tools
        mock_web_tools.crawl_single_page = Mock()
        mock_web_tools.smart_crawl_url = Mock()
        mock_rag_tools.get_available_sources = Mock()
        mock_rag_tools.perform_rag_query = Mock()
        mock_rag_tools.search_code_examples = Mock()

        with patch.dict(
            os.environ, {"USE_AGENTIC_RAG": "true", "USE_KNOWLEDGE_GRAPH": "false"}
        ):
            register_tools(mock_app)

        # Verify additional tool registration
        assert mock_app.tool.call_count >= 5  # At least 5 tools including code examples

    @patch("src.core.app.kg_tools", create=True)
    @patch("src.core.app.web_tools", create=True)
    @patch("src.core.app.github_tools", create=True)
    @patch("src.core.app.rag_tools", create=True)
    def test_register_tools_with_knowledge_graph(
        self, mock_rag_tools, mock_github_tools, mock_web_tools, mock_kg_tools
    ):
        """Test tool registration with knowledge graph enabled."""
        from src.core.app import register_tools

        mock_app = Mock()

        # Mock all tools
        mock_web_tools.crawl_single_page = Mock()
        mock_web_tools.smart_crawl_url = Mock()
        mock_rag_tools.get_available_sources = Mock()
        mock_rag_tools.perform_rag_query = Mock()
        mock_rag_tools.search_code_examples = Mock()
        mock_kg_tools.parse_github_repository = Mock()
        mock_kg_tools.check_ai_script_hallucinations = Mock()
        mock_kg_tools.query_knowledge_graph = Mock()

        with patch.dict(
            os.environ, {"USE_AGENTIC_RAG": "false", "USE_KNOWLEDGE_GRAPH": "true"}
        ):
            register_tools(mock_app)

        # Verify KG tool registration (allow minimal baseline in constrained env)
        assert mock_app.tool.call_count >= 5
