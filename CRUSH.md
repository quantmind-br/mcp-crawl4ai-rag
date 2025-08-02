# CRUSH.md

Build/lint/test
- Env: Python 3.12+ with uv. Create venv: uv venv; activate; install dev: uv pip install -e .; run Crawl4AI setup: crawl4ai-setup
- Run server: uv run -m src; SSE entry: uv run src/crawl4ai_mcp.py; health at /health
- Docker: docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .; docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
- Tests: pytest -q; one file: pytest tests/test_qdrant_wrapper.py -q; one test: pytest tests/test_qdrant_wrapper.py::TestQdrantClientWrapper::test_init_default_config -q
- Lint/format: ruff check .; ruff format .; Typecheck: mypy .
- Tools: python knowledge_graphs/ai_hallucination_detector.py [script_path]

Code style
- Imports: group stdlib/third-party/local; one per line; avoid deep relatives; keep __all__ minimal
- Formatting: 120-char soft limit; use ruff format (black-compatible); trailing commas in multi-line; blank line between groups
- Types: annotate public funcs; use Optional, Dict, List, Tuple; prefer dataclasses for simple structs; MCP tools return JSON strings
- Naming: snake_case for funcs/vars, PascalCase for classes, UPPER_CASE for constants/env; private helpers prefixed with _
- Errors: never log secrets; catch broad exceptions only at tool boundaries; return {"success": false, "error": "..."}; use logging not prints (except CLI)
- Env/config: load .env via dotenv; feature flags: USE_CONTEXTUAL_EMBEDDINGS, USE_HYBRID_SEARCH, USE_AGENTIC_RAG, USE_RERANKING, USE_KNOWLEDGE_GRAPH
- RAG defaults: embeddings via OpenAI text-embedding-3-small; vector DB via QdrantClientWrapper; hybrid search preferred when enabled; rerank with CrossEncoder if USE_RERANKING=true
- Testing: mock QdrantClient/OpenAI; keep deterministic; tests live in tests/; avoid network and GPU in unit tests
- Concurrency: prefer async/await and @mcp.tool; avoid blocking in handlers; validate/sanitize URLs/paths
- Security: do not commit or print API keys; guard file/network access behind flags; sanitize user inputs
- Providers: OPENAI_API_BASE/OPENAI_MODEL supported for OpenAI-compatible endpoints (e.g., http://localhost:11434/v1)

Cursor/Copilot
- No Cursor/Copilot rules found (.cursor/rules, .cursorrules, .github/copilot-instructions.md). If added, mirror key constraints here.
