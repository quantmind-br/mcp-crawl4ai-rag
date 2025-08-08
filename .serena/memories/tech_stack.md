# Technology Stack

## Core Technologies
- **Python**: 3.12+ (required)
- **MCP**: Model Context Protocol v1.7.1 for AI agent integration
- **FastMCP**: Async MCP server framework
- **Crawl4AI**: v0.6.2 for web crawling capabilities
- **Qdrant**: Vector database for semantic search and storage
- **Neo4j**: Graph database for knowledge graph and code analysis

## AI/ML Libraries
- **OpenAI**: v1.71.0 for LLM integration
- **FastEmbed**: v0.4.0+ for embeddings
- **Sentence-Transformers**: v5.0.0+ for additional embedding models
- **PyTorch**: CUDA-enabled for GPU acceleration
- **Tree-sitter**: Multi-language code parsing

## Development Tools
- **uv**: Modern Python package manager (preferred)
- **pytest**: v8.4.1+ for testing
- **ruff**: v0.12.7+ for linting and formatting
- **Docker**: Required for Qdrant and Neo4j services

## Supported Tree-sitter Languages
- Python, JavaScript, Java, Go, Rust, C/C++, C#, PHP, Ruby, Kotlin

## Infrastructure
- **Redis**: Optional caching layer
- **Docker Compose**: Service orchestration
- **Windows/Linux/Mac**: Cross-platform support

## API Providers
- **OpenAI**: Primary LLM/embeddings provider
- **DeepInfra**: Cost-effective alternative
- **Azure OpenAI**: Enterprise option