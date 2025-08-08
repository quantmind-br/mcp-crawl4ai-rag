# Technology Stack

## Core Technologies
- **Python 3.12+**: Primary programming language
- **MCP 1.7.1**: Model Context Protocol for AI agent integration
- **FastMCP**: Async MCP server framework
- **Crawl4AI 0.6.2**: Advanced web crawling engine
- **Qdrant Client 1.12.0+**: Vector database client
- **Neo4j 5.28.1+**: Graph database for knowledge representation

## AI/ML Stack
- **OpenAI 1.71.0**: LLM and embeddings API client
- **FastEmbed 0.4.0+**: Fast embedding generation
- **Sentence Transformers 5.0.0+**: Local embedding models
- **PyTorch 2.5.1**: Deep learning framework with CUDA support
- **Tree-sitter**: Multi-language parsing for code analysis

## Data & Storage
- **Qdrant**: Vector database for embeddings and semantic search
- **Neo4j**: Knowledge graph database for code relationships
- **Redis 5.0.0+**: Caching layer for embeddings and API responses

## Development Tools
- **uv**: Fast Python package manager (preferred over pip)
- **pytest 8.4.1+**: Testing framework with async support
- **Ruff 0.12.7+**: Fast Python linter and formatter
- **Docker**: Containerization for databases (Qdrant, Neo4j, Redis)

## Multi-Language Parsing
Tree-sitter parsers for:
- Python, JavaScript, TypeScript
- Java, Go, Rust, C/C++, C#
- PHP, Ruby, Kotlin

## API Providers (Flexible)
- **OpenAI**: Default provider for chat and embeddings
- **DeepInfra**: Cost-effective alternative
- **Azure OpenAI**: Enterprise integration
- **Custom Providers**: Any OpenAI-compatible API

## Infrastructure
- **Docker Compose**: Service orchestration
- **Windows Support**: Optimized for Windows development
- **Event Loop Fixes**: Windows-specific async handling
- **GPU Acceleration**: CUDA support for reranking models