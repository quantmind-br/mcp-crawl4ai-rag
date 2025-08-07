# CRUSH.md

Build/lint/test
- Env: Python 3.12+ with uv. Create venv: uv venv; activate; install deps: uv sync; editable install: uv pip install -e .; crawler setup: uv run crawl4ai-setup
- Run server: uv run -m src; stdio: TRANSPORT=stdio uv run -m src; SSE: uv run src/crawl4ai_mcp.py; health at /health
- Docker: docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .; docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag; compose: docker-compose up -d
- Tests: uv run pytest -q; single file: uv run pytest tests/test_qdrant_wrapper.py -q; single test: uv run pytest tests/test_qdrant_wrapper.py::TestQdrantClientWrapper::test_init_default_config -q
- Lint/format/typecheck: uv run ruff check .; uv run ruff format .; uv run mypy .
- Utilities: uv run python scripts/clean_qdrant.py; uv run python scripts/define_qdrant_dimensions.py; KG tools: uv run python knowledge_graphs/ai_hallucination_detector.py <script.py>

Code style
- Imports: group stdlib/third-party/local; one per line; avoid deep relatives; minimal __all__; keep MCP tool imports local to tools
- Formatting: 120-char soft limit; use ruff format (black-compatible); trailing commas on multiline; one blank line between groups
- Types: annotate public funcs; use Optional/Dict/List/Tuple; prefer dataclasses for simple structs; MCP tools return JSON-serializable dicts/strings
- Naming: snake_case for funcs/vars; PascalCase for classes; UPPER_CASE for constants/env; private helpers prefixed with _
- Errors: never log secrets; catch broad exceptions only at tool boundaries; return {"success": false, "error": "..."}; use logging not prints (except CLI)
- Env/config: load .env via dotenv; support CHAT_/EMBEDDINGS_ with fallbacks; legacy OPENAI_* still supported; allow EMBEDDINGS_DIMENSIONS override
- RAG defaults: embeddings text-embedding-3-small; vector DB QdrantClientWrapper; optional hybrid search; rerank via CrossEncoder when USE_RERANKING=true
- Testing: mock Qdrant/OpenAI; deterministic tests; unit tests avoid network/GPU; run via uv for consistent env
- Concurrency: async/await everywhere; @mcp.tool for MCP; avoid blocking I/O in handlers; validate/sanitize URLs/paths; device_manager handles GPU fallback
- Security: do not commit/print API keys; guard file/network access behind flags; sanitize inputs; avoid writing to arbitrary paths

Cursor/Copilot
- No Cursor (.cursor/rules, .cursorrules) or Copilot (.github/copilot-instructions.md) rules found; if added, mirror constraints here and follow strictly

Notes
- Default chat model: gpt-4o-mini (do not change without explicit instruction)