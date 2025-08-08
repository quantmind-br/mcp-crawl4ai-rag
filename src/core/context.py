"""Application context for the Crawl4AI MCP server."""

from dataclasses import dataclass
from typing import Any, Optional

from crawl4ai import AsyncWebCrawler
from sentence_transformers import CrossEncoder


@dataclass
class Crawl4AIContext:
    """Enhanced context for the Crawl4AI MCP server containing all application dependencies.

    This context serves as a dependency injection container that holds references
    to all major application components including web crawler, vector database client,
    machine learning models, and services.

    Attributes:
        crawler: AsyncWebCrawler instance for web scraping
        qdrant_client: Qdrant vector database client wrapper
        embedding_cache: Redis cache for embedding storage
        reranker: CrossEncoder model for result reranking (optional)
        knowledge_validator: Knowledge graph validator for hallucination detection (optional)
        repo_extractor: Neo4j repository extractor for code analysis (optional)
        embedding_service: Service for generating and managing embeddings
        rag_service: Service for retrieval-augmented generation operations
    """

    # Core infrastructure components
    crawler: AsyncWebCrawler
    qdrant_client: Any  # QdrantClientWrapper - using Any to avoid circular import
    embedding_cache: Any  # EmbeddingCache instance

    # Optional ML models
    reranker: Optional[CrossEncoder] = None
    knowledge_validator: Optional[Any] = None  # KnowledgeGraphValidator when available
    repo_extractor: Optional[Any] = None  # DirectNeo4jExtractor when available

    # Application services (to be added in future phases)
    embedding_service: Optional[Any] = None  # EmbeddingService instance
    rag_service: Optional[Any] = None  # RagService instance
