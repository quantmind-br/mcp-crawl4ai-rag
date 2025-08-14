# Technology Stack

## Core Framework
- **Python 3.12+**: Primary programming language
- **FastMCP (MCP 1.7.1)**: Model Context Protocol server framework
- **uv**: Package manager for dependency management

## Web Crawling & Content Processing
- **Crawl4AI 0.6.2**: Advanced web crawling engine
- **Requests 2.32.3+**: HTTP library for web requests

## Vector Database & Search
- **Qdrant 1.12.0+**: Vector database for semantic search
- **FastEmbed 0.4.0+**: Embedding generation
- **Sentence Transformers 5.0.0+**: Text embeddings and reranking models

## Knowledge Graph & Code Analysis
- **Neo4j 5.28.1+**: Graph database for code structure analysis
- **Tree-sitter 0.23.0+**: Multi-language code parsing
  - Support for Python, JavaScript, Java, Go, Rust, C/C++, TypeScript, C#, PHP, Ruby, Kotlin

## AI/LLM Integration
- **OpenAI 1.71.0**: LLM API client (supports multiple providers)
- **PyTorch**: GPU acceleration for embedding models
- **CUDA 12.1**: GPU acceleration (Windows-specific wheels)

## Development & Testing
- **pytest 8.4.1+**: Testing framework with async support
- **pytest-asyncio**: Async testing support
- **ruff 0.12.7+**: Code linting and formatting

## Infrastructure
- **Docker & Docker Compose**: Container orchestration for Qdrant and Neo4j
- **Redis 5.0.0**: Caching layer
- **Tenacity 8.0.0+**: Retry logic and fault tolerance

## Environment Management
- **python-dotenv**: Environment variable management
- **Windows-specific optimizations**: Unicode handling, event loop fixes

## Architecture Patterns
- **Async/Await**: Asynchronous programming throughout
- **Singleton Pattern**: Context and resource management
- **Factory Pattern**: Parser and processor selection
- **Modular Design**: Organized into features, services, tools, and utilities