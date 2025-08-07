# Technology Stack

## Core Technologies
- **Python 3.12+**: Primary development language
- **MCP (Model Context Protocol)**: AI agent integration framework
- **FastMCP**: Async MCP server implementation with SSE and stdio transports
- **UV Package Manager**: Fast Python package management and virtual environments

## Web Crawling & Data Processing
- **Crawl4AI 0.6.2**: Advanced web crawling with content extraction
- **Requests 2.32.3+**: HTTP client for API interactions
- **BeautifulSoup** (via Crawl4AI): HTML parsing and content extraction

## Vector Database & Search
- **Qdrant 1.12.0+**: Vector database for semantic search
- **FastEmbed 0.4.0+**: Embedding generation with multiple model support
- **Sentence-Transformers 5.0.0+**: Cross-encoder models for reranking

## AI/ML APIs
- **OpenAI 1.71.0**: Primary API for chat completions and embeddings
- **Multi-Provider Support**: DeepInfra, Azure OpenAI, custom OpenAI-compatible APIs
- **PyTorch 2.5.1+**: GPU acceleration for reranking models (CUDA 12.1)
- **torchvision & torchaudio**: Complete PyTorch ecosystem

## Knowledge Graph & Code Analysis
- **Neo4j 5.28.1+**: Graph database for code structure analysis
- **AST (Python built-in)**: Abstract syntax tree analysis for code validation

## Infrastructure & DevOps
- **Docker & Docker Compose**: Containerized services (Qdrant, Neo4j, Redis)
- **Redis 5.0.0-6.0.0**: Embedding cache and session storage
- **Tenacity 8.0.0+**: Retry logic for API calls and database operations

## Development & Testing
- **Pytest 8.4.1+**: Testing framework with async support
- **Ruff 0.12.7+**: Fast Python linter and formatter
- **Mock/unittest**: Mocking framework for unit tests
- **Concurrent.futures**: Parallel processing for code example extraction

## Windows-Specific
- **Batch Scripts**: `setup.bat`, `start.bat` for Windows automation
- **CUDA 12.1**: GPU acceleration support for Windows
- **Windows Event Loop**: Custom event loop configuration for asyncio compatibility