# CRUSH.md

Build/lint/test
- Env: Python 3.12+ with uv. Create venv: uv venv; activate; install dev: uv sync; editable install: uv pip install -e .; run Crawl4AI setup: crawl4ai-setup
- Run server: uv run -m src; SSE entry: uv run src/crawl4ai_mcp.py; health at /health; stdio: TRANSPORT=stdio uv run -m src
- Docker: docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .; docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag; compose: docker-compose up -d
- Tests: uv run pytest -q; one file: uv run pytest tests/test_qdrant_wrapper.py -q; single test: uv run pytest tests/test_qdrant_wrapper.py::TestQdrantClientWrapper::test_init_default_config -q
- Lint/format/typecheck: uv run ruff check .; uv run ruff format .; uv run mypy .
- Utilities: uv run python scripts/clean_qdrant.py; uv run python scripts/fix_qdrant_dimensions.py; KG tools: uv run python knowledge_graphs/ai_hallucination_detector.py <script.py>

Code style
- Imports: group stdlib/third-party/local; one per line; avoid deep relatives; keep __all__ minimal
- Formatting: 120-char soft limit; use ruff format (black-compatible); trailing commas in multi-line; blank line between groups
- Types: annotate public funcs; use Optional, Dict, List, Tuple; prefer dataclasses for simple structs; MCP tools return JSON-friendly dicts/strings
- Naming: snake_case for funcs/vars, PascalCase for classes, UPPER_CASE for constants/env; private helpers prefixed with _
- Errors: never log secrets; catch broad exceptions only at tool boundaries; return {"success": false, "error": "..."}; use logging not prints (except CLI)
- Env/config: load .env via dotenv; support CHAT_/EMBEDDINGS_ primary + fallback; legacy OPENAI_* still supported; explicit EMBEDDINGS_DIMENSIONS when needed
- RAG defaults: embeddings text-embedding-3-small; vector DB QdrantClientWrapper; enable hybrid search when flag set; rerank with CrossEncoder if USE_RERANKING=true
- Testing: mock Qdrant/OpenAI; deterministic; tests in tests/; no network/GPU in unit tests; prefer uv run pytest for consistency
- Concurrency: prefer async/await and @mcp.tool; avoid blocking I/O in handlers; validate/sanitize URLs/paths; use device_manager for GPU fallback
- Security: do not commit/print API keys; guard file/network access behind flags; sanitize user inputs; avoid writing to arbitrary paths

Cursor/Copilot
- No Cursor (.cursor/rules, .cursorrules) or Copilot (.github/copilot-instructions.md) rules present; if added, mirror constraints here and follow them strictly
