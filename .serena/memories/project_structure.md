# Project Structure

## Root Directory
```
├── .env.example          # Environment configuration template
├── docker-compose.yaml   # Docker services (Qdrant, Neo4j, Redis)
├── pyproject.toml        # Python dependencies and project metadata
├── pytest.ini           # Test configuration
├── setup.bat            # Windows service initialization script
├── start.bat            # Windows server startup script
└── run_server.py        # Alternative server entry point
```

## Source Code (`src/`)
```
src/
├── core/                 # Application core
│   ├── app.py           # FastMCP server setup and tool registration
│   └── context.py       # Singleton context management
├── tools/               # MCP tools (exposed to AI agents)
│   ├── web_tools.py     # Web crawling tools
│   ├── github_tools.py  # GitHub repository indexing
│   ├── rag_tools.py     # Vector search and RAG queries
│   └── kg_tools.py      # Knowledge graph and hallucination detection
├── services/            # Business logic layer
│   ├── rag_service.py   # RAG search with reranking
│   ├── embedding_service.py # Embedding generation and caching
│   └── unified_indexing_service.py # Multi-destination indexing
├── clients/             # External API integrations
│   ├── qdrant_client.py # Vector database client
│   └── llm_api_client.py # LLM and embedding API client
├── k_graph/             # Knowledge graph system (modularized)
│   ├── core/            # Core interfaces and models
│   ├── parsing/         # Tree-sitter multi-language parsers
│   ├── analysis/        # Hallucination detection and validation
│   └── services/        # Repository parsing services
├── features/            # Feature modules
│   └── github/          # GitHub processing pipeline
│       ├── processors/  # Language-specific processors
│       ├── discovery/   # File discovery and filtering
│       └── repository/  # Git operations and metadata
├── utils/               # Utility functions
└── __main__.py         # Main entry point (`uv run -m src`)
```

## Testing (`tests/`)
```
tests/
├── conftest.py          # Shared pytest fixtures
├── integration/         # Full workflow tests
├── tools/              # MCP tool tests
├── rag/                # RAG service tests
├── knowledge_graphs/   # Tree-sitter parser tests
├── clients/            # Database client tests
├── performance/        # Performance benchmarks
└── fixtures/           # Test data and mocks
```

## Scripts (`scripts/`)
```
scripts/
├── clean_qdrant.py              # Database cleanup utility
├── define_qdrant_dimensions.py  # Fix dimension mismatches
├── query_knowledge_graph.py     # Neo4j query interface
└── build_grammars.py           # Tree-sitter grammar compilation
```

## Configuration Directories
```
.serena/                 # Serena memory files and project configuration
PRPs/                    # Project Requirement Papers (development docs)
SPEC_PRP/               # Specification PRPs
.claude/                # Claude Code integration commands
```

## Key Entry Points
1. **`src/__main__.py`** - Primary entry via `uv run -m src`
2. **`run_server.py`** - Alternative entry via `uv run python run_server.py`
3. **`src/core/app.py`** - FastMCP application factory and tool registration

## Tool Organization Pattern
Each tool category has dedicated files:
- **Web tools**: Single page crawling, smart URL detection
- **GitHub tools**: Repository cloning, indexing, multi-language processing
- **RAG tools**: Vector search, source filtering, code example search
- **Knowledge graph tools**: Code parsing, hallucination detection, graph queries

## Configuration Flow
```
.env → Environment Variables → Context Singletons → Service Instances → MCP Tools
```