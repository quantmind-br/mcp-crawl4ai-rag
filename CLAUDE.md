# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a **Crawl4AI RAG MCP Server** that provides AI agents and coding assistants with advanced web crawling and RAG capabilities through the Model Context Protocol (MCP). The system integrates Crawl4AI for web scraping, Qdrant for vector storage, and optionally Neo4j for knowledge graph-based hallucination detection.

### Core Components
- **MCP Server**: `src/crawl4ai_mcp.py` - Main FastMCP server with tools for crawling and RAG
- **Vector Database**: Qdrant client wrapper (`src/qdrant_wrapper.py`) for document and code storage
- **Web Crawler**: Crawl4AI integration with smart URL detection and parallel processing
- **Knowledge Graph**: Neo4j integration (`knowledge_graphs/`) for AI hallucination detection
- **Utils**: Helper functions (`src/utils.py`) for embeddings, chunking, and API clients

### Key Technologies
- **MCP**: Model Context Protocol for AI agent integration
- **Crawl4AI**: Web crawling with automatic content extraction
- **Qdrant**: Vector database for semantic search
- **Neo4j**: Graph database for code analysis (optional)
- **OpenAI/DeepInfra**: APIs for embeddings and chat completions
- **PyTorch**: GPU acceleration for reranking (optional)

## Development Commands

### Setup and Installation
```bash
# Install dependencies
uv sync

# Setup Docker services (Qdrant + Neo4j)
setup.bat  # Windows
# or: docker-compose up -d

# Start MCP server
start.bat  # Windows
# or: uv run -m src
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/test_mcp_basic.py          # Basic MCP functionality
uv run pytest tests/test_qdrant_wrapper.py     # Vector database tests
uv run pytest tests/test_deepinfra_config.py   # API configuration tests
uv run pytest tests/test_github_processor.py   # Knowledge graph tests

# Performance benchmarks
uv run pytest tests/performance_benchmark.py

# Integration tests (requires Docker services)
uv run pytest tests/integration_test.py
```

### Development Scripts
```bash
# Clean Qdrant database
uv run python scripts/clean_qdrant.py

# Fix Qdrant dimensions for existing collections
uv run python scripts/fix_qdrant_dimensions.py

# Knowledge graph tools
uv run python knowledge_graphs/parse_repo_into_neo4j.py <repo_url>
uv run python knowledge_graphs/ai_hallucination_detector.py <script_path>
```

## Configuration Architecture

The system uses a flexible multi-provider API configuration supporting:

### Modern Configuration (Recommended)
- `CHAT_MODEL` + `CHAT_API_KEY` + `CHAT_API_BASE` - For chat/completion operations
- `EMBEDDINGS_MODEL` + `EMBEDDINGS_API_KEY` + `EMBEDDINGS_API_BASE` - For vector embeddings
- `EMBEDDINGS_DIMENSIONS` - Explicit dimension override

### Legacy Configuration (Still Supported)
- `OPENAI_API_KEY` - Fallback for both chat and embeddings
- `MODEL_CHOICE` - Deprecated, use `CHAT_MODEL` instead

### RAG Strategy Flags
All default to `false`, enable as needed:
- `USE_CONTEXTUAL_EMBEDDINGS` - Enhanced context for chunks
- `USE_HYBRID_SEARCH` - Keyword + semantic search
- `USE_AGENTIC_RAG` - Specialized code example extraction
- `USE_RERANKING` - Cross-encoder result reordering
- `USE_KNOWLEDGE_GRAPH` - AI hallucination detection

## MCP Tools

### Core Tools (Always Available)
- `crawl_single_page` - Crawl individual webpage
- `smart_crawl_url` - Intelligent crawling (sitemaps, recursive)
- `get_available_sources` - List indexed sources
- `perform_rag_query` - Semantic search with source filtering

### Conditional Tools
- `search_code_examples` - Code-specific search (requires `USE_AGENTIC_RAG=true`)
- `parse_github_repository` - Index repo structure (requires `USE_KNOWLEDGE_GRAPH=true`)
- `check_ai_script_hallucinations` - Validate AI code (requires `USE_KNOWLEDGE_GRAPH=true`)
- `query_knowledge_graph` - Explore knowledge graph (requires `USE_KNOWLEDGE_GRAPH=true`)

## Model Configuration
- O modelo padrão que estamos usando na aplicação é o gpt-4o-mini. Não altere o modelo em qualquer arquivo sem a expressa instrução do usuário.

## Code Architecture Patterns

### Error Handling
- Use `tenacity` for retrying failed operations (especially API calls)
- Implement graceful fallbacks (GPU → CPU, contextual → basic embeddings)
- Log errors with context using the configured logger

### Async Patterns
- All MCP tools are async functions using `@mcp.tool()` decorator
- Web crawling uses `AsyncWebCrawler` with parallel processing
- Database operations are async with connection pooling

### Device Management
- `src/device_manager.py` handles GPU detection and memory management
- Automatic fallback from CUDA → MPS → CPU
- Memory cleanup after GPU operations

### Configuration Loading
- Environment variables loaded via `dotenv`
- Backward compatibility for legacy configurations
- Dynamic embedding dimensions based on model choice

## Important Implementation Notes

### Vector Dimensions
- Embedding dimensions are auto-detected from model choice
- Override with `EMBEDDINGS_DIMENSIONS` if needed
- Collections recreated if dimensions mismatch

### Windows Compatibility
- `src/event_loop_fix.py` handles Windows ConnectionResetError issues
- Batch scripts (`.bat`) for setup and startup
- PyTorch CUDA wheels specified for Windows in `pyproject.toml`

### Testing Strategy
- Mock external services (OpenAI, Qdrant) in unit tests
- Integration tests require actual Docker services
- Performance benchmarks measure crawling and search speed

## Coding Guidelines
- Não utilize emojis no código
- Use type hints for all function parameters and return values
- Follow async/await patterns for I/O operations
- Implement proper error handling with retries for external services