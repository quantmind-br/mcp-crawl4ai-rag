# Tech Stack - Crawl4AI MCP RAG

## Core Technologies
- **Python**: 3.12+ (primary language)
- **uv**: Package manager for dependency management
- **FastMCP**: MCP server framework for async tool implementation
- **Docker**: Container orchestration for databases

## AI/ML Stack
- **Crawl4AI**: Web crawling and content extraction (v0.6.2)
- **OpenAI**: Chat models and embeddings API integration
- **sentence-transformers**: Local embedding models (v5.0.0+)
- **PyTorch**: GPU acceleration support with CUDA 12.1
- **fastembed**: Fast embedding computation (v0.4.0+)

## Database Technologies
- **Qdrant**: Vector database for semantic search (v1.12.0+)
- **Neo4j**: Knowledge graph database for code analysis (v5.28.1+)
- **Redis**: Caching layer for embeddings (v5.0.0+)

## Development Tools
- **pytest**: Testing framework (v8.4.1+)
- **ruff**: Linting and formatting (v0.12.7+)
- **tenacity**: Retry logic and resilience (v8.0.0+)

## Multi-Language Parsing
- **tree-sitter**: Syntax parsing for multiple languages
  - Python, JavaScript, TypeScript, Java, Go, Rust
  - C, C++, C#, PHP, Ruby, Kotlin

## Transport Protocols
- **SSE (Server-Sent Events)**: Default transport for web clients
- **stdio**: Standard input/output for CLI integration

## Environment
- **Windows**: Primary development platform (with .bat scripts)
- **Linux/Mac**: Cross-platform support with shell scripts
- **Docker**: Cross-platform database services