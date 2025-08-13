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


def configure_windows_logging():
    """
    Configure logging to suppress Windows ConnectionResetError messages.

    These errors are cosmetic and occur during graceful shutdown on Windows
    with ProactorEventLoop. They don't affect functionality.
    """
    import platform

    if platform.system().lower() == "windows":
        # Suppress asyncio ConnectionResetError logs that are cosmetic
        asyncio_logger = logging.getLogger("asyncio")

        class ConnectionResetFilter(logging.Filter):
            def filter(self, record):
                # Suppress the specific Windows connection reset errors
                if record.levelno == logging.ERROR:
                    message = record.getMessage()
                    if any(
                        pattern in message
                        for pattern in [
                            "ConnectionResetError: [WinError 10054]",
                            "_ProactorBasePipeTransport._call_connection_lost",
                            "Foi forçado o cancelamento de uma conexão existente",
                            "O nome da rede especificado não está mais disponível",
                            "Accept failed on a socket",
                        ]
                    ):
                        return False
                return True

        asyncio_logger.addFilter(ConnectionResetFilter())
        logger.debug("Applied Windows ConnectionResetError log suppression")


class ContextSingleton:
    """Singleton for managing the application context to avoid multiple initializations."""

    _instance: Optional["ContextSingleton"] = None
    _context: Optional[Crawl4AIContext] = None
    _initializing: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_context(self, server: FastMCP) -> Crawl4AIContext:
        """Get the application context, initializing if necessary."""
        if self._context is not None:
            return self._context

        if self._initializing:
            # Wait for initialization to complete
            import asyncio

            while self._initializing:
                await asyncio.sleep(0.01)
            return self._context

        # Initialize the context
        self._initializing = True
        try:
            self._context = await self._initialize_context(server)
            return self._context
        finally:
            self._initializing = False

    async def _initialize_context(self, server: FastMCP) -> Crawl4AIContext:
        """Initialize the application context once."""
        logger.info("Starting application initialization...")

        # Apply Windows Unicode compatibility fixes FIRST
        try:
            from ..utils.windows_unicode_fix import setup_windows_unicode_compatibility

            setup_windows_unicode_compatibility()
            logger.debug("Applied Windows Unicode compatibility fixes")
        except ImportError:
            logger.debug("Windows Unicode fix module not available")

        # Initialize Tree-sitter grammars if needed (for knowledge graph features)
        try:
            from ..utils.grammar_initialization import initialize_grammars_if_needed

            initialize_grammars_if_needed()
        except ImportError:
            logger.info(
                "Grammar initialization module not available, skipping Tree-sitter setup"
            )

        # Create browser configuration with Unicode safety
        try:
            from ..utils.windows_unicode_fix import create_safe_crawler_config

            browser_config = create_safe_crawler_config()
            logger.debug(
                "Using safe crawler configuration for Windows Unicode compatibility"
            )
        except ImportError:
            # Fallback to original configuration
            browser_config = BrowserConfig(headless=True, verbose=False)
            logger.debug("Using fallback crawler configuration")

        # Initialize the crawler with enhanced error handling
        try:
            crawler = AsyncWebCrawler(config=browser_config)
            await crawler.__aenter__()
            logger.info("Web crawler initialized")
        except Exception as e:
            logger.error(f"Error initializing crawler: {e}")
            # Try with minimal configuration as fallback
            try:
                minimal_config = BrowserConfig(headless=True, verbose=False)
                crawler = AsyncWebCrawler(config=minimal_config)
                await crawler.__aenter__()
                logger.info("Web crawler initialized with minimal configuration")
            except Exception as e2:
                logger.error(f"Failed to initialize crawler with minimal config: {e2}")
                raise

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

            # Expor para facilitar patch em testes
            globals()["validate_embeddings_config"] = validate_embeddings_config
            globals()["get_embedding_dimensions"] = get_embedding_dimensions

            dimensions = get_embedding_dimensions()
            logger.info(f"Embedding configuration validated - dimensions: {dimensions}")
        except Exception as e:
            logger.error(f"Embedding validation failed: {e}")
            dimensions = 1024  # fallback

        # Initialize reranking model (if enabled)
        reranking_model = RerankingModelSingleton().get_model()

        # Initialize knowledge graph components (if available)
        knowledge_graph_singleton = KnowledgeGraphSingleton()
        (
            knowledge_validator,
            repo_extractor,
        ) = await knowledge_graph_singleton.get_components_async()

        if knowledge_validator or repo_extractor:
            logger.info("Knowledge graph components initialized successfully")

        # Create and return context
        context = Crawl4AIContext(
            crawler=crawler,
            qdrant_client=qdrant_client,
            embedding_cache=embedding_cache,
            reranker=reranking_model,
            knowledge_validator=knowledge_validator,
            repo_extractor=repo_extractor,
        )

        logger.info("Application initialization completed successfully")
        return context

    async def cleanup(self):
        """Cleanup the singleton context."""
        if self._context:
            try:
                await self._context.crawler.__aexit__(None, None, None)

                knowledge_graph_singleton = KnowledgeGraphSingleton()
                await knowledge_graph_singleton.close_components()

            except Exception as e:
                logger.error(f"Error during context cleanup: {e}")
            finally:
                self._context = None


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

            # Extract device from model_kwargs to avoid duplication
            model_kwargs = (
                device_info.model_kwargs.copy()
                if isinstance(device_info.model_kwargs, dict)
                else {}
            )

            self._model = CrossEncoder(
                model_name,
                trust_remote_code=False,
                **model_kwargs,
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

            from ..k_graph.analysis.validator import KnowledgeGraphValidator
            from ..k_graph.services.repository_parser import DirectNeo4jExtractor

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
    Manages the application lifecycle using singleton pattern to prevent duplicate initialization.

    This function uses ContextSingleton to ensure components are initialized only once
    across multiple SSE connections, significantly reducing startup time and resource usage.

    Args:
        server: The FastMCP server instance

    Yields:
        Crawl4AIContext: The context containing all initialized application components
    """
    context_singleton = ContextSingleton()

    try:
        # Get context from singleton (initializes once, reuses afterwards)
        context = await context_singleton.get_context(server)
        logger.info("Application context ready (using singleton pattern)")
        yield context

    except Exception as e:
        logger.error(f"Error in application lifespan: {e}")
        raise

    finally:
        # Note: We don't cleanup here since the singleton manages lifecycle
        # The cleanup happens when the singleton is explicitly cleaned up
        # or when the application shuts down
        logger.debug("Application lifespan context manager exiting")


async def cleanup_application() -> None:
    """
    Cleanup application resources managed by singletons.

    This function should be called when the application is shutting down
    to properly cleanup all singleton resources.
    """
    logger.info("Starting application cleanup...")

    try:
        context_singleton = ContextSingleton()
        await context_singleton.cleanup()
        logger.info("Application cleanup completed")
    except Exception as e:
        logger.error(f"Error during application cleanup: {e}")


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

        # Register unified repository indexing tool with enhanced description
        app.tool(
            name="index_github_repository",
            description="""
            🚀 UNIFIED GITHUB REPOSITORY INDEXING - Advanced dual-system processing tool
            
            CAPABILITIES:
            • Simultaneous indexing for semantic search (Qdrant) and code analysis (Neo4j)
            • Multi-language support: Python, JavaScript/TypeScript, Go, Java, Rust, C/C++, Ruby, PHP, Kotlin, Swift, Scala, C#, Markdown/MDX, and more
            • 50-70% faster processing through unified pipeline architecture
            • Cross-system file_id linking for comprehensive code intelligence
            • Production-ready with robust error handling and detailed statistics
            
            USE CASES:
            🎯 Code analysis & understanding  🎯 Semantic search & similarity  🎯 AI development context
            🎯 Repository exploration        🎯 Dependency mapping           🎯 Documentation linking
            
            OUTPUTS:
            📊 QDRANT: Vector embeddings for semantic search and RAG applications
            🕸️ NEO4J: Knowledge graph with classes, functions, methods, and relationships
            
            PARAMETERS:
            • repo_url: GitHub repository URL (public or authenticated)
            • destination: "qdrant" | "neo4j" | "both" (default: "both") 
            • file_types: Array of extensions (default: [".md"])
              📋 SUPPORTED TYPES: [".md", ".mdx", ".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".kt", ".swift", ".scala", ".json", ".yaml", ".yml", ".toml", ".xml", ".html", ".css", ".scss", ".sql"]
            • max_files: Processing limit (default: 50, recommend: 20-500)
            • chunk_size: RAG chunk size (default: 5000 chars)
            • max_size_mb: Repository size limit (default: 500MB)
            
            Returns comprehensive JSON with processing statistics, storage summary,
            performance metrics, and detailed file-level results.
            """,
        )(github_tools.index_github_repository)
        logger.info(
            "GitHub tools imported and registered (including unified index_github_repository)"
        )
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
    Includes graceful shutdown handling to minimize Windows connection errors.
    """
    import asyncio
    import signal

    # Configure Windows-specific logging to suppress ConnectionResetError messages
    configure_windows_logging()

    logger.info("Starting MCP Crawl4AI RAG server...")

    # Create the application instance using the new structure
    app = create_app()

    # Register all tools
    register_tools(app)

    # Run with appropriate transport
    transport = os.getenv("TRANSPORT", "sse")
    logger.info(f"Using transport: {transport}")

    # Setup graceful shutdown handling
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal, initiating graceful shutdown...")
        shutdown_event.set()

    # Register signal handlers for graceful shutdown
    try:
        if hasattr(signal, "SIGINT"):
            signal.signal(signal.SIGINT, lambda s, f: signal_handler())
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, lambda s, f: signal_handler())
    except (OSError, ValueError):
        # Signal handling might not be available in some environments
        logger.debug("Could not register signal handlers")

    try:
        if transport == "sse":
            logger.info("Starting SSE transport server...")
            await app.run_sse_async()
        else:
            logger.info("Starting stdio transport server...")
            await app.run_stdio_async()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        # Ensure cleanup happens
        logger.info("Starting server shutdown cleanup...")

        try:
            # Cleanup application resources
            await cleanup_application()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        # Give a brief moment for any pending connections to close gracefully
        try:
            await asyncio.sleep(0.2)
        except Exception:
            pass

        logger.info("Server shutdown completed")
