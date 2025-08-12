# Technology Stack

## Core Framework
- **Python 3.12+**: Main language with modern async/await patterns
- **FastMCP**: MCP server framework for tool registration
- **uv**: Package manager for dependency management

## External Dependencies
### Web Crawling
- **Crawl4AI 0.6.2**: Web crawling and content extraction
- **requests**: HTTP operations

### Vector Database & Search
- **Qdrant**: Vector database for semantic search
- **OpenAI**: LLM API and embeddings
- **sentence-transformers**: Text embeddings and reranking
- **fastembed**: Embedding optimization

### Knowledge Graph
- **Neo4j**: Graph database for code structure analysis
- **tree-sitter**: Multi-language code parsing (10+ languages)

### Infrastructure
- **Docker**: Required for Qdrant, Neo4j, and Redis services
- **Redis**: Caching layer for embeddings
- **PyTorch**: GPU acceleration for reranking models

## Development Tools
- **pytest**: Testing framework with hierarchical test organization
- **ruff**: Code linting and formatting
- **pytest-asyncio**: Async testing support

## Key Integrations
- **MCP Protocol**: Tool registration and client communication
- **OpenAI API**: Chat models and embeddings with fallback support
- **Multi-provider support**: Azure OpenAI, DeepInfra for cost optimization