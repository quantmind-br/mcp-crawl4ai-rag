# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Python/UV Development
```bash
# Install dependencies
uv pip install -e .
crawl4ai-setup

# Run the MCP server directly
uv run src/crawl4ai_mcp.py

# Activate virtual environment
uv venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

### Docker Development
```bash
# Build Docker image
docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .

# Run with Docker
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

### Testing
```bash
# Install test dependencies
uv pip install -e ".[test]"

# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only

# Run specific test file
pytest tests/test_working.py
```

### Knowledge Graph Tools (requires Neo4j setup)
```bash
# Check for AI hallucinations in a Python script
python knowledge_graphs/ai_hallucination_detector.py [script_path]

# Query knowledge graph interactively
python knowledge_graphs/query_knowledge_graph.py
```

## Architecture Overview

This is an MCP (Model Context Protocol) server that provides web crawling and RAG capabilities with optional AI hallucination detection.

### Core Components

- **`src/crawl4ai_mcp.py`**: Main MCP server implementation using FastMCP framework
- **`src/utils.py`**: Utility functions for Supabase operations, embeddings, and content processing
- **`knowledge_graphs/`**: Neo4j-based AI hallucination detection system
- **`crawled_pages.sql`**: Supabase database schema with pgvector extension

### Key Architecture Patterns

**MCP Tools Structure**: The server exposes tools via `@mcp.tool()` decorators. Core tools include:
- `crawl_single_page` - Individual page crawling
- `smart_crawl_url` - Intelligent full-site crawling (handles sitemaps, llms-full.txt, recursive crawling)
- `perform_rag_query` - Vector search with optional source filtering
- `search_code_examples` - Specialized code example retrieval (when `USE_AGENTIC_RAG=true`)
- Knowledge graph tools for hallucination detection (when `USE_KNOWLEDGE_GRAPH=true`)

**RAG Strategy Configuration**: Five configurable RAG strategies via environment variables:
- `USE_CONTEXTUAL_EMBEDDINGS` - LLM-enhanced chunk embeddings with document context
- `USE_HYBRID_SEARCH` - Combines vector and keyword search
- `USE_AGENTIC_RAG` - Extracts and indexes code examples separately
- `USE_RERANKING` - Cross-encoder reranking of search results
- `USE_KNOWLEDGE_GRAPH` - Neo4j-based code validation and hallucination detection

**Data Storage**: Uses Supabase with three main tables:
- `crawled_pages` - Document chunks with vector embeddings
- `code_examples` - Extracted code blocks with summaries (agentic RAG)
- `sources` - Source metadata and summaries

**Knowledge Graph System**: Neo4j-based code structure analysis:
- Parses GitHub repositories into nodes (Repository, File, Class, Method, Function)
- Validates AI-generated code against real implementations
- Detects hallucinations like non-existent methods or incorrect usage

### Environment Configuration

The server requires extensive environment configuration. Key variables:
- Supabase: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`
- OpenAI: `OPENAI_API_KEY`, `EMBEDDING_MODEL`, `MODEL_CHOICE`
- Neo4j (optional): `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- RAG strategies: `USE_CONTEXTUAL_EMBEDDINGS`, `USE_HYBRID_SEARCH`, etc.
- Performance: `MAX_WORKERS`, `SUPABASE_BATCH_SIZE`

### Crawling Intelligence

The `smart_crawl_url` function automatically detects URL types:
- **Sitemaps** (.xml): Extracts URLs and crawls each page
- **LLM text files** (llms-full.txt, llms.txt): Processes structured documentation
- **Regular webpages**: Performs recursive crawling following internal links

Content is intelligently chunked by headers and processed with configurable embedding strategies.

## Important Notes

- The knowledge graph functionality is not fully Docker-compatible yet - use `uv` for local development if using hallucination detection
- The server supports both SSE and stdio transports for MCP clients
- Crawling performance is optimized with concurrent processing and batch operations
- The codebase uses defensive security practices and validates all inputs