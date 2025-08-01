# CRUSH.md

Build/lint/test
- Env: Python 3.12+, uv recommended. Create venv: uv venv; activate; install: uv pip install -e .; run crawl4ai setup: crawl4ai-setup
- Run server (SSE): uv run src/crawl4ai_mcp.py; module entry: python -m src
- Docker build/run: docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .; docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
- Tests: pytest -q; single test file: pytest tests/test_qdrant_wrapper.py -q; single test: pytest tests/test_qdrant_wrapper.py::TestQdrantClientWrapper::test_init_default_config -q
- Lint/format: ruff check .; ruff format . (if ruff installed). Typecheck: mypy . (if configured)

Project conventions
- Imports: stdlib, third-party, local; avoid relative beyond package root; keep __all__ minimal
- Types: use typing for all public funcs; Optional/Dict/List; dataclasses for simple structs; return JSON strings in MCP tools
- Naming: snake_case for funcs/vars, PascalCase for classes, UPPER_CASE for constants/env; private helpers with leading underscore
- Formatting: 120-char soft limit; ruff format or black default; one import per line; trailing commas in multi-line literals
- Errors: never leak secrets; catch broad exceptions only at tool boundaries and return structured JSON {success,error}; log with logging not prints (except CLI)
- Env/config: load .env via dotenv at startup; feature flags USE_CONTEXTUAL_EMBEDDINGS, USE_HYBRID_SEARCH, USE_AGENTIC_RAG, USE_RERANKING, USE_KNOWLEDGE_GRAPH
- RAG: embeddings via OpenAI text-embedding-3-small; vector DB via QdrantClientWrapper; prefer hybrid search when enabled; rerank with CrossEncoder if USE_RERANKING=true
- Code examples: extract large fenced blocks; store in code_examples with summary; search via search_code_examples tool
- Knowledge graph: optional Neo4j; validate paths and URLs; never run if USE_KNOWLEDGE_GRAPH=false
- Testing: isolate network by mocking QdrantClient and OpenAI; keep deterministic; prefer unit tests in tests/

Notes for agents
- Single-file run: uv run path/to/script.py; health: curl http://localhost:8051/health (if SSE served)
- Respect .serena/memories guidelines for style; follow FastMCP patterns with @mcp.tool and async/await
- No Cursor/Copilot rules found; none to import
