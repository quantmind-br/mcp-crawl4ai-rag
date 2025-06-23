# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this refactored MCP Crawl4AI RAG server.

## Development Commands

### Running the Server

**Development (uv):**
```bash
uv run src/crawl4ai_mcp.py
```

**Docker:**
```bash
docker-compose up -d
```

### Environment Setup

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
crawl4ai-setup
```

**Configuration:**
Copy `.env.example` to `.env` and set:
- `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` (required)
- `CHAT_MODEL_API_KEY` / `EMBEDDING_MODEL_API_KEY` (required)
- Feature flags: `USE_CONTEXTUAL_EMBEDDINGS`, `USE_HYBRID_SEARCH`, etc.

### Database Setup

Execute `crawled_pages_1024d.sql` in Supabase SQL Editor.

## Refactored Architecture

**Modular Structure:**
- **`src/crawl4ai_mcp.py`** - 55-line main entry point (was 2000+ lines)
- **`src/core/`** - Server and context management
- **`src/clients/`** - API client abstractions with fallback support
- **`src/services/`** - Business logic (crawling, RAG, content processing)
- **`src/tools/`** - MCP tools split into logical groups
- **`src/utils/`** - Utility functions and rate limiting

**Tools Available:**
1. **Core**: `crawl_single_page`, `smart_crawl_url`, `get_available_sources`, `perform_rag_query`
2. **Optional**: `search_code_examples` (if `USE_AGENTIC_RAG=true`)
3. **Knowledge Graph**: `parse_github_repository`, `check_ai_script_hallucinations` (if `USE_KNOWLEDGE_GRAPH=true`)

**RAG Strategies:**
Enable via environment flags: `USE_CONTEXTUAL_EMBEDDINGS`, `USE_HYBRID_SEARCH`, `USE_AGENTIC_RAG`, `USE_RERANKING`, `USE_KNOWLEDGE_GRAPH`

## Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Health check
curl http://localhost:8051/health
```

## Troubleshooting

**503 Errors:** Set `MAX_WORKERS_SUMMARY=1`, `MAX_WORKERS_CONTEXT=1`, `RATE_LIMIT_DELAY=0.5`

**Memory Issues:** Use `EMBEDDING_DIMENSIONS=1024` with Qwen3-Embedding-0.6B

**Docker Network:** Use `http://ollama:11434/v1` inside containers, `http://localhost:11434/v1` locally