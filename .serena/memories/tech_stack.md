# Tech Stack

## Core Technologies

### Backend Framework
- **Python 3.12+** - Main programming language
- **FastMCP (MCP 1.7.1)** - Model Context Protocol server framework
- **AsyncIO** - Asynchronous programming for performance

### Web Crawling
- **Crawl4AI 0.6.2** - Advanced web crawling engine
- **requests** - HTTP client for API calls

### Databases
- **Qdrant** - Vector database for semantic search (localhost:6333)
- **Neo4j** - Knowledge graph database (localhost:7474/7687)
- **Redis** - Caching layer (localhost:6379)

### AI/ML Stack
- **OpenAI API** - LLM and embeddings (configurable providers)
- **sentence-transformers** - Local embedding models and reranking
- **PyTorch** - ML framework with CUDA support for GPU acceleration
- **fastembed** - Fast embedding generation

### Code Analysis
- **tree-sitter** - Multi-language code parsing
  - Supports Python, JavaScript/TypeScript, Java, Go, Rust, C/C++, C#, PHP, Ruby, Kotlin

### Development Tools
- **uv** - Python package manager (modern pip/poetry replacement)
- **pytest** - Testing framework with asyncio support
- **ruff** - Fast Python linter and formatter
- **Docker & Docker Compose** - Containerization for databases

### Supported Platforms
- **Windows** - Primary development platform with specialized Windows handling
- **Linux/Mac** - Cross-platform support

## Architecture Patterns
- **Singleton Context Management** - Shared resources across MCP tools
- **Async/Await** - Non-blocking I/O operations
- **Factory Pattern** - Dynamic parser and processor selection
- **Modular Design** - Organized by features and functionality