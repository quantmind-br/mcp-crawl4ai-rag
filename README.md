# Crawl4AI RAG MCP Server

*Web Crawling and RAG Capabilities for AI Agents and AI Coding Assistants*

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that integrates [Crawl4AI](https://crawl4ai.com) with vector databases for advanced web crawling and RAG capabilities.

**Key Features:**
- **Smart crawling** with automatic URL type detection
- **Advanced RAG strategies** (contextual embeddings, hybrid search, reranking)
- **Local privacy** with Ollama support  
- **Modular architecture** for easy maintenance
- **Knowledge graphs** for AI hallucination detection

## Tools

**Core Tools:**
- `crawl_single_page` - Crawl and store single web page
- `smart_crawl_url` - Auto-detect and crawl websites (sitemaps, txt files, recursive)
- `get_available_sources` - List available data sources
- `perform_rag_query` - Semantic search with source filtering

**Optional Tools:**
- `search_code_examples` - Code-specific search (requires `USE_AGENTIC_RAG=true`)
- `parse_github_repository` - Index GitHub repos (requires `USE_KNOWLEDGE_GRAPH=true`)
- `check_ai_script_hallucinations` - Validate AI-generated code
- `query_knowledge_graph` - Explore code knowledge graphs

## Prerequisites

- **Supabase account** (vector database)
- **API keys** for chat/embedding models (OpenAI or local Ollama)
- **Optional**: Neo4j for knowledge graphs

## Quick Start

### Docker (Recommended)
```bash
git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
cd mcp-crawl4ai-rag
cp .env.example .env
# Edit .env with your credentials
docker-compose up -d
```

### Python Development
```bash
git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
cd mcp-crawl4ai-rag
uv venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e . && crawl4ai-setup
cp .env.example .env
# Edit .env with your credentials
uv run src/crawl4ai_mcp.py
```

### Database Setup
Execute `crawled_pages_1024d.sql` in your Supabase SQL Editor.

## Configuration

Essential `.env` variables:

```bash
# Server
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# Database (required)
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key

# Models (required)
CHAT_MODEL=gpt-4o-mini
CHAT_MODEL_API_KEY=your_api_key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_MODEL_API_KEY=your_api_key
EMBEDDING_DIMENSIONS=1536

# RAG Features (optional)
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false
USE_KNOWLEDGE_GRAPH=false

# Neo4j (optional, for knowledge graphs)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

**Local Ollama Setup:**
```bash
EMBEDDING_MODEL=dengcao/Qwen3-Embedding-0.6B:Q8_0
EMBEDDING_MODEL_API_BASE=http://localhost:11434/v1
EMBEDDING_DIMENSIONS=1024
```

## MCP Client Integration

### SSE (Recommended)
```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "sse",
      "url": "http://localhost:8051/sse"
    }
  }
}
```

### Claude Code
```bash
claude mcp add-json crawl4ai-rag '{"type":"http","url":"http://localhost:8051/sse"}' --scope user
```

### Stdio
```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "uv",
      "args": ["run", "src/crawl4ai_mcp.py"],
      "env": {
        "TRANSPORT": "stdio",
        "SUPABASE_URL": "your_url",
        "SUPABASE_SERVICE_KEY": "your_key"
      }
    }
  }
}
```

## Health Check

```bash
curl http://localhost:8051/health
```

## Troubleshooting

**503 Errors:** Reduce concurrency with `MAX_WORKERS_SUMMARY=1`

**Memory Issues:** Use `EMBEDDING_DIMENSIONS=1024` with Qwen3-0.6B model

**Docker Networks:** Use `http://ollama:11434/v1` in containers, `localhost` otherwise

## Architecture

This refactored server uses a modular architecture:
- **Core**: Server setup and lifespan management
- **Clients**: API client abstractions with fallback support  
- **Services**: Business logic (crawling, RAG, content processing)
- **Tools**: MCP tools organized by functionality
- **Utils**: Shared utilities and rate limiting
