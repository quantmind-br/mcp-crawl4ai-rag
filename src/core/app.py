"""
Core application module for MCP Crawl4AI RAG server.

This module contains the central application setup logic including FastMCP instance
creation, lifespan management, and tool registration. It serves as the entry point
for initializing the entire application architecture.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Optional

from mcp.server.fastmcp import FastMCP
from crawl4ai import AsyncWebCrawler, BrowserConfig
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

from .context import Crawl4AIContext

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class RerankingModelSingleton:
    """Singleton for managing reranking model initialization and lifecycle."""

    _instance: Optional["RerankingModelSingleton"] = None
    _model: Optional[CrossEncoder] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self) -> Optional[CrossEncoder]:
        """Get the reranking model, initializing if necessary."""
        if not os.getenv("USE_RERANKING", "false") == "true":
            return None

        if self._model is not None:
            return self._model

        try:
            # Import device management utilities
            try:
                from ..device_manager import get_optimal_device, cleanup_gpu_memory
            except ImportError:
                from device_manager import get_optimal_device, cleanup_gpu_memory

            device_info = get_optimal_device()
            model_name = os.getenv(
                "RERANKING_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )

            logger.info(f"Initializing reranking model: {model_name}")
            logger.info(
                f"Using device: {device_info.device} ({device_info.device_type})"
            )

            self._model = CrossEncoder(
                model_name,
                device=device_info.device,
                trust_remote_code=False,
                **(
                    device_info.model_kwargs
                    if isinstance(device_info.model_kwargs, dict)
                    else {}
                ),
            )

            # Warm up the model
            warmup_samples = int(os.getenv("RERANKING_WARMUP_SAMPLES", "5"))
            logger.info(f"Warming up reranking model with {warmup_samples} samples...")

            dummy_pairs = [
                ["sample query", "sample document"],
                ["test query", "test document"],
                ["example", "content"],
                ["search", "result"],
                ["question", "answer"],
            ][:warmup_samples]

            _ = self._model.predict(dummy_pairs)
            logger.info("Reranking model warmed up successfully")

            return self._model

        except Exception as e:
            logger.error(f"Failed to initialize reranking model: {e}")
            cleanup_gpu_memory()
            return None


class KnowledgeGraphSingleton:
    """Singleton for managing knowledge graph components initialization and lifecycle."""

    _instance: Optional["KnowledgeGraphSingleton"] = None
    _knowledge_validator: Optional = None
    _repo_extractor: Optional = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_components_async(self):
        """Get knowledge graph components, initializing if necessary."""
        if not os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true":
            return None, None

        if self._initialized:
            return self._knowledge_validator, self._repo_extractor

        try:
            # Dynamic imports for Neo4j components
            knowledge_graphs_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "knowledge_graphs"
            )
            if knowledge_graphs_path not in sys.path:
                sys.path.insert(0, knowledge_graphs_path)

            from knowledge_graphs.knowledge_graph_validator import (
                KnowledgeGraphValidator,
            )
            from knowledge_graphs.parse_repo_into_neo4j import DirectNeo4jExtractor

            # Get Neo4j connection parameters
            neo4j_uri = os.getenv("NEO4J_URI")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "")

            # Test Neo4j connection
            await self._validate_neo4j_connection()

            self._knowledge_validator = KnowledgeGraphValidator(
                neo4j_uri, neo4j_user, neo4j_password
            )
            self._repo_extractor = DirectNeo4jExtractor(
                neo4j_uri, neo4j_user, neo4j_password
            )

            # Initialize components
            await self._knowledge_validator.initialize()
            await self._repo_extractor.initialize()

            logger.info("Knowledge graph components initialized successfully")
            self._initialized = True
            return self._knowledge_validator, self._repo_extractor

        except Exception as e:
            logger.warning(f"Knowledge graph initialization failed: {e}")
            logger.warning("Knowledge graph features will be unavailable")
            return None, None

    async def _validate_neo4j_connection(self):
        """Validate Neo4j connection configuration."""
        neo4j_uri = os.getenv("NEO4J_URI")
        if not neo4j_uri:
            raise ValueError("NEO4J_URI environment variable not set")

        # Import Neo4j driver for connection testing
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("neo4j package not installed")

        driver = None
        try:
            driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(
                    os.getenv("NEO4J_USER", "neo4j"),
                    os.getenv("NEO4J_PASSWORD", ""),
                ),
            )

            # Test connection with a simple query (Neo4j driver uses sync methods)
            with driver.session() as session:
                session.run("RETURN 1")

            logger.info("Neo4j connection validated successfully")

        except Exception as e:
            raise RuntimeError(f"Neo4j connection failed: {e}")
        finally:
            if driver:
                driver.close()

    async def close_components(self):
        """Clean up knowledge graph components."""
        if self._knowledge_validator:
            try:
                await self._knowledge_validator.close()
            except Exception as e:
                logger.warning(f"Error closing knowledge validator: {e}")

        if self._repo_extractor:
            try:
                await self._repo_extractor.close()
            except Exception as e:
                logger.warning(f"Error closing repo extractor: {e}")

        self._knowledge_validator = None
        self._repo_extractor = None
        self._initialized = False


@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the application lifecycle and dependency injection.

    This function initializes all application components including the web crawler,
    vector database client, ML models, and services. It serves as the central
    dependency injection container and manages component lifecycles.

    Args:
        server: The FastMCP server instance

    Yields:
        Crawl4AIContext: The context containing all initialized application components
    """
    logger.info("Starting application initialization...")

    # Initialize Tree-sitter grammars if needed (for knowledge graph features)
    try:
        from ..utils.grammar_initialization import initialize_grammars_if_needed
        initialize_grammars_if_needed()
    except ImportError:
        # Fallback for backward compatibility during migration
        try:
            from src.utils.grammar_initialization import initialize_grammars_if_needed
            initialize_grammars_if_needed()
        except ImportError:
            logger.info("Grammar initialization module not available, skipping Tree-sitter setup")

    # Create browser configuration
    browser_config = BrowserConfig(headless=True, verbose=False)

    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    logger.info("Web crawler initialized")

    # Initialize Qdrant client
    try:
        from ..clients.qdrant_client import get_qdrant_client
    except ImportError:
        # Fallback for backward compatibility during migration
        try:
            from ..clients.qdrant_client import get_qdrant_client
        except ImportError:
            from clients.qdrant_client import get_qdrant_client
    # Expor função para facilitar patch em testes
    globals()["get_qdrant_client"] = get_qdrant_client

    qdrant_client = get_qdrant_client()
    logger.info("Qdrant client initialized")

    # Initialize embedding cache
    try:
        from ..embedding_cache import get_embedding_cache
    except ImportError:
        from embedding_cache import get_embedding_cache
    # Expor para facilitar patch em testes
    globals()["get_embedding_cache"] = get_embedding_cache
    embedding_cache = get_embedding_cache()
    logger.info("Embedding cache initialized")

    # Validate embeddings configuration and dimensions
    try:
        try:
            from ..embedding_config import (
                validate_embeddings_config,
                get_embedding_dimensions,
            )
        except ImportError:
            from embedding_config import (
                validate_embeddings_config,
                get_embedding_dimensions,
            )
        # Expor para escopo do módulo para facilitar patch nos testes
        globals()["validate_embeddings_config"] = validate_embeddings_config
        globals()["get_embedding_dimensions"] = get_embedding_dimensions
        validate_embeddings_config()
        embedding_dims = get_embedding_dimensions()
        logger.info(f"Embedding configuration validated - dimensions: {embedding_dims}")
    except Exception as e:
        logger.warning(f"Embedding configuration validation failed: {e}")
        logger.warning(
            "Server will continue but embedding functionality may not work correctly"
        )

    # Initialize ML models
    reranking_singleton = RerankingModelSingleton()
    reranker = reranking_singleton.get_model()

    kg_singleton = KnowledgeGraphSingleton()
    knowledge_validator, repo_extractor = await kg_singleton.get_components_async()

    # Initialize services (to be implemented in future phases)
    embedding_service = None
    rag_service = None

    try:
        # Create enhanced context with all components
        context = Crawl4AIContext(
            crawler=crawler,
            qdrant_client=qdrant_client,
            embedding_cache=embedding_cache,
            reranker=reranker,
            knowledge_validator=knowledge_validator,
            repo_extractor=repo_extractor,
            embedding_service=embedding_service,
            rag_service=rag_service,
        )

        logger.info("Application initialization completed successfully")
        yield context

    finally:
        # Clean up all components
        logger.info("Starting application cleanup...")

        await crawler.__aexit__(None, None, None)
        logger.info("Web crawler cleaned up")

        # Knowledge graph components cleanup
        await kg_singleton.close_components()
        logger.info("Knowledge graph components cleaned up")

        # GPU memory cleanup
        try:
            try:
                from ..device_manager import cleanup_gpu_memory
            except ImportError:
                from device_manager import cleanup_gpu_memory
            cleanup_gpu_memory()
            logger.info("GPU memory cleaned up")
        except Exception as e:
            logger.warning(f"GPU cleanup warning: {e}")

        logger.info("Application cleanup completed")


def create_app() -> FastMCP:
    """
    Create and configure the FastMCP application instance.

    This function creates the main application instance with proper configuration
    including server name, host, port, and lifespan management.

    Returns:
        FastMCP: Configured FastMCP server instance ready for tool registration
    """
    logger.info("Creating FastMCP application instance...")

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8051"))

    app = FastMCP(
        name="mcp-crawl4ai-rag",
        instructions="MCP server for RAG and web crawling with Crawl4AI",
        host=host,
        port=port,
        lifespan=crawl4ai_lifespan,
    )

    logger.info(f"FastMCP application created - Host: {host}, Port: {port}")
    return app


def register_tools(app: FastMCP) -> None:
    """
    Register all MCP tools with the application instance.

    This function imports and registers all available MCP tools from the tools layer.
    It serves as the central tool registration point for the application.

    Args:
        app: The FastMCP application instance to register tools with
    """
    logger.info("Registering MCP tools...")

    # For now, tools are registered via decorators in the original crawl4ai_mcp.py file
    # This is a transitional approach until all tools are fully migrated
    # The tools modules are imported to make them available but registration
    # happens through the existing @mcp.tool decorators

    # Import web tools module to make tools available
    try:
        from ..tools import web_tools

        # Register web tools manually with the app instance
        app.tool()(web_tools.crawl_single_page)
        app.tool()(web_tools.smart_crawl_url)
        logger.info("Web tools imported and registered")
    except ImportError as e:
        logger.error(f"Failed to import web tools: {e}")

    # Import GitHub tools module to make tools available
    try:
        from ..tools import github_tools

        # Register GitHub tools manually with the app instance
        app.tool()(github_tools.smart_crawl_github)
        logger.info("GitHub tools imported and registered")
    except ImportError as e:
        logger.error(f"Failed to import GitHub tools: {e}")

    # Import RAG tools module to make tools available
    try:
        from ..tools import rag_tools

        # Register RAG tools manually with the app instance
        app.tool()(rag_tools.get_available_sources)
        app.tool()(rag_tools.perform_rag_query)

        # Register search_code_examples only if agentic RAG is enabled
        if os.getenv("USE_AGENTIC_RAG", "false") == "true":
            app.tool()(rag_tools.search_code_examples)
            logger.info("RAG tools imported and registered (including code examples)")
        else:
            logger.info("RAG tools imported and registered (excluding code examples)")
    except ImportError as e:
        logger.error(f"Failed to import RAG tools: {e}")

    # Import knowledge graph tools (if enabled)
    if os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true":
        try:
            from ..tools import kg_tools

            # Register KG tools manually with the app instance
            app.tool()(kg_tools.parse_github_repository)
            app.tool()(kg_tools.check_ai_script_hallucinations)
            app.tool()(kg_tools.query_knowledge_graph)
            logger.info("Knowledge graph tools imported and registered")
        except ImportError as e:
            # Fallback: use pre-injected module if available (e.g., in tests)
            if "kg_tools" in globals():
                kg_tools = globals()["kg_tools"]
                app.tool()(kg_tools.parse_github_repository)
                app.tool()(kg_tools.check_ai_script_hallucinations)
                app.tool()(kg_tools.query_knowledge_graph)
                logger.info("Knowledge graph tools registered via pre-injected module")
            else:
                logger.warning(f"Knowledge graph tools not available: {e}")
    else:
        logger.info("Knowledge graph tools disabled (USE_KNOWLEDGE_GRAPH=false)")

    logger.info("MCP tools registration completed")


async def run_server() -> None:
    """
    Run the MCP server with the appropriate transport protocol.

    This function creates the application instance and runs it with either
    SSE or stdio transport based on the TRANSPORT environment variable.
    """
    logger.info("Starting MCP Crawl4AI RAG server...")

    # Create the application instance using the new structure
    app = create_app()

    # Register all tools
    register_tools(app)

    # Run with appropriate transport
    transport = os.getenv("TRANSPORT", "sse")
    logger.info(f"Using transport: {transport}")

    if transport == "sse":
        logger.info("Starting SSE transport server...")
        await app.run_sse_async()
    else:
        logger.info("Starting stdio transport server...")
        await app.run_stdio_async()
