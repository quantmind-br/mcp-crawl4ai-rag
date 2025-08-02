# CRUSH.md

Build/lint/test
- Env: Python 3.12+ with uv. Create venv: uv venv; activate; install dev: uv pip install -e .; run Crawl4AI setup: crawl4ai-setup
- Run server (SSE): uv run src/crawl4ai_mcp.py; module entry: uv run -m src
- Docker build/run: docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .; docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
- Tests (quiet): pytest -q; single file: pytest tests/test_qdrant_wrapper.py -q; single test: pytest tests/test_qdrant_wrapper.py::TestQdrantClientWrapper::test_init_default_config -q
- Lint/format: ruff check .; ruff format .; Typecheck: mypy .
- Knowledge graph tools: python knowledge_graphs/ai_hallucination_detector.py [script_path]

Code style
- Imports: group stdlib, third-party, local; avoid deep relatives beyond package root; keep __all__ minimal; one import per line
- Formatting: 120-char soft limit; prefer ruff format (or black defaults); trailing commas in multi-line literals; blank line between groups
- Types: annotate public functions; use Optional, Dict, List, Tuple; prefer dataclasses for simple structs; return JSON strings from MCP tools
- Naming: snake_case for funcs/vars, PascalCase for classes, UPPER_CASE for constants/env; private helpers prefixed with _
- Errors: never print secrets; catch broad exceptions only at tool boundaries; return structured JSON {"success": false, "error": "..."}; use logging not prints (except CLI)
- Env/config: load .env via dotenv on startup; feature flags supported: USE_CONTEXTUAL_EMBEDDINGS, USE_HYBRID_SEARCH, USE_AGENTIC_RAG, USE_RERANKING, USE_KNOWLEDGE_GRAPH
- RAG defaults: embeddings via OpenAI text-embedding-3-small; vector DB via QdrantClientWrapper; prefer hybrid search when enabled; rerank with CrossEncoder if USE_RERANKING=true
- Testing: isolate network by mocking QdrantClient/OpenAI; keep deterministic; unit tests live in tests/
- CLI/entry: run single scripts via uv run path/to/script.py; module entry uv run -m src; health at /health when SSE served
- Concurrency: prefer async/await and @mcp.tool patterns; avoid blocking calls in tool handlers
- Security: do not log API keys; validate and sanitize URLs/paths; guard file/network access behind flags
- OpenAI-compatible providers: set OPENAI_API_BASE and OPENAI_MODEL for custom endpoints (ex. http://localhost:11434/v1)

Cursor/Copilot
- No Cursor/Copilot rules found in repo (.cursor/rules, .cursorrules, .github/copilot-instructions.md). If added later, mirror key constraints here.
