# Technology Stack

## Core Technologies
- **Python**: 3.12+ (required)
- **MCP Framework**: FastMCP 1.7.1 for Model Context Protocol server
- **Package Manager**: uv (UV package manager)
- **Containerization**: Docker & Docker Compose

## AI/ML Stack
- **Web Crawler**: Crawl4AI 0.6.2 for intelligent web scraping
- **Vector Database**: Qdrant (qdrant-client >=1.12.0)
- **Embeddings**: FastEmbed >=0.4.0 for efficient vector embeddings
- **Reranking**: sentence-transformers >=5.0.0 with CrossEncoder models
- **GPU Support**: PyTorch 2.5.1 with CUDA 12.1 support

## API Providers
- **OpenAI**: GPT models and text-embedding models
- **DeepInfra**: Cost-effective alternative (Qwen, Llama models)
- **Azure OpenAI**: Enterprise-grade deployment option
- **Multi-Provider**: Flexible configuration with fallback support

## Databases
- **Qdrant**: Vector database for semantic search (localhost:6333)
- **Neo4j**: Knowledge graph for code analysis (localhost:7474)
- **Redis**: Caching layer for embeddings (localhost:6379)

## Development Tools
- **Testing**: pytest >=8.4.1 with comprehensive test suites
- **Linting**: ruff >=0.12.7 for code quality
- **Environment**: python-dotenv for configuration management
- **HTTP Client**: requests >=2.32.3 with enhanced Windows compatibility

## Windows-Specific
- **Event Loop Fix**: Custom Windows compatibility layer for ConnectionResetError
- **PyTorch CUDA**: Pre-compiled wheels for Windows CUDA 12.1
- **Batch Scripts**: setup.bat and start.bat for Windows automation

## Docker Services
- **Qdrant**: Vector database container
- **Neo4j**: Knowledge graph container
- **Redis**: Caching layer container