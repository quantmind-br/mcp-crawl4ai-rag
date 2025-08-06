# Technology Stack

## Core Technologies
- **Language**: Python 3.12+ (async-first architecture)
- **Framework**: FastMCP (Model Context Protocol server)
- **Package Manager**: uv (Astral UV)
- **Web Scraping**: Crawl4AI 0.6.2 with async web crawler

## Vector Storage
- **Primary**: Qdrant vector database
- **Collections**: docs (1536-3072 dims), code_examples (dynamic)
- **Indexing**: Semantic embeddings, hybrid search with dense+sparse vectors

## AI Providers
- **Primary**: OpenAI (gpt-4o-mini, text-embedding-3-small)
- **Alternatives**: DeepInfra (Qwen embeddings), Azure OpenAI
- **Configuration**: Multi-provider with automatic fallback support

## Databases
- **Vector**: Qdrant (localhost:6333)
- **Graph**: Neo4j (localhost:7687) - optional for knowledge graphs
- **Cache**: Redis (localhost:6379) - optional for embeddings

## GPU Acceleration
- **Torch**: CUDA 12.1 with Windows-specific wheels
- **Models**: Cross-encoder reranking (transformers)
- **Device Manager**: Automatic GPU detection and memory management

## Development Tools
- **Testing**: pytest with async support, comprehensive test suite
- **Linting**: ruff (installed in dev dependencies)
- **Environment Management**: dotenv with comprehensive .env examples