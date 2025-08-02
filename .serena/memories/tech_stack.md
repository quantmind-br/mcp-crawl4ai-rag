# Technology Stack

## Core Framework & Protocol
- **MCP (Model Context Protocol)**: v1.7.1 - Server framework for AI agent integration
- **FastMCP**: Context management and tool orchestration
- **Python**: 3.12+ required

## Web Crawling & Content Processing
- **Crawl4AI**: v0.6.2 - Intelligent web crawling with markdown extraction
- **requests**: v2.32.3+ - HTTP requests for direct API calls
- **sentence-transformers**: v5.0.0+ - ML models for embeddings and reranking

## Vector Database & Search
- **Qdrant**: v1.12.0+ - High-performance vector database
- **QdrantClientWrapper**: Custom wrapper for optimized client management
- **CrossEncoder**: For result reranking when enabled

## AI & Machine Learning
- **OpenAI**: v1.71.0 - Embeddings and chat completion API
- **PyTorch**: v2.5.1+cu121 - GPU acceleration for ML models
- **torchvision**: v0.20.1+cu121 - Computer vision utilities
- **torchaudio**: v2.5.1+cu121 - Audio processing utilities

## Knowledge Graph (Optional)
- **Neo4j**: v5.28.1+ - Graph database for code analysis
- **Custom parsers**: AST-based Python code analysis tools

## Development & Testing
- **uv**: Modern Python dependency management
- **pytest**: v8.4.1+ - Testing framework with async support
- **ruff**: Linting and code formatting
- **mypy**: Static type checking
- **tenacity**: v8.0.0+ - Retry logic for resilient operations

## Configuration & Environment
- **python-dotenv**: v0.9.9 - Environment variable management
- **Docker**: Containerization for Qdrant and Neo4j services

## Platform Support
- **Primary**: Windows (with .bat scripts)
- **Compatible**: Linux/macOS with equivalent shell commands
- **GPU**: CUDA 12.1 support for accelerated ML operations