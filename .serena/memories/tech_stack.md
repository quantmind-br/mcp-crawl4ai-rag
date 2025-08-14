# Technology Stack

## Core Framework
- **FastMCP 1.7.1** - Model Context Protocol server framework
- **Python 3.12+** - Programming language
- **asyncio** - Asynchronous programming foundation

## Web Crawling & Data Processing
- **Crawl4AI 0.6.2** - Advanced web crawling engine
- **Tree-sitter 0.23.0** - Multi-language code parsing
  - Supports: Python, JavaScript/TypeScript, Java, Go, Rust, C/C++, C#, PHP, Ruby, Kotlin

## Vector Database & Search
- **Qdrant 1.12.0+** - Vector database for embeddings
- **sentence-transformers 5.0.0+** - Text embeddings and reranking
- **fastembed 0.4.0+** - Fast embedding generation
- **FastBM25** - Sparse vector search for hybrid retrieval

## Knowledge Graph & Validation
- **Neo4j 5.28.1+** - Graph database for code structure
- **Cypher** - Graph query language

## AI & Machine Learning
- **OpenAI 1.71.0** - LLM and embedding APIs
- **PyTorch 2.5.1+CUDA** - Deep learning framework (GPU acceleration)
- **Cross-encoder models** - Result reranking

## Caching & Performance
- **Redis 5.0.0** - Caching layer for embeddings
- **Concurrent.futures** - Parallel processing
- **HTTPX** - High-performance HTTP client with HTTP/2 support

## Development Tools
- **uv** - Fast Python package manager
- **pytest 8.4.1+** - Testing framework with async support
- **ruff 0.12.7+** - Fast Python linter and formatter
- **Docker & Docker Compose** - Containerization for Qdrant, Neo4j, Redis

## Supported Platforms
- **Windows** - Primary development platform with specialized scripts
- **Linux/Mac** - Full compatibility
- **GPU Support** - CUDA acceleration for reranking models

## API Integrations
- **OpenAI API** - Chat completions and embeddings
- **DeepInfra API** - Cost-effective embedding alternatives
- **GitHub API** - Repository metadata and cloning

## Package Management
- **pyproject.toml** - Modern Python packaging
- **uv sync** - Dependency synchronization
- **Direct PyTorch CUDA URLs** - Windows-optimized GPU packages