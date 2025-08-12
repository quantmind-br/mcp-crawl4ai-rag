# Technology Stack

## Core Framework & Protocol
- **MCP (Model Context Protocol)** v1.7.1 - AI agent integration framework
- **FastMCP** - Async MCP server implementation 
- **Python 3.12+** - Primary programming language

## Web Crawling & Processing
- **Crawl4AI** v0.6.2 - Advanced web crawling engine
- **AsyncWebCrawler** - Asynchronous crawling with context management

## Databases & Storage
- **Qdrant** >=1.12.0 - Vector database for semantic search
- **Neo4j** >=5.28.1 - Graph database for code structure analysis  
- **Redis** >=5.0.0 - Caching layer for embeddings and API responses

## AI & Machine Learning
- **OpenAI** v1.71.0 - LLM and embedding API client
- **Sentence Transformers** >=5.0.0 - Text embeddings and cross-encoder reranking
- **FastEmbed** >=0.4.0 - Efficient embedding computations
- **PyTorch** (CUDA 12.1) - GPU acceleration for ML models

## Multi-Language Code Analysis
- **Tree-sitter** >=0.23.0 - Multi-language parsing framework
- Language parsers: Python, JavaScript/TypeScript, Java, Go, Rust, C/C++, C#, PHP, Ruby, Kotlin

## Testing & Quality
- **pytest** >=8.4.1 - Testing framework with async support
- **pytest-asyncio** >=1.1.0 - Async test support
- **ruff** >=0.12.7 - Linting and code formatting

## Package Management & Environment
- **uv** - Fast Python package manager (preferred over pip/poetry)
- **Docker & Docker Compose** - Service containerization
- **dotenv** - Environment variable management

## API Providers (Multi-provider support)
- **OpenAI** - Default for chat and embeddings
- **Azure OpenAI** - Enterprise alternative  
- **DeepInfra** - Cost-effective provider
- **Fallback configuration** - High availability setup

## Development Tools
- **Windows** - Primary development platform
- **Git** - Version control
- **setup.bat** - Windows service initialization script