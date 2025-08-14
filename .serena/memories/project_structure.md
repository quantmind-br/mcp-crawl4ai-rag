# Project Structure

## Top-level Organization
```
E:\mcp-crawl4ai-rag\
├── src/                          # Main source code
├── tests/                        # Test suite (hierarchical organization)
├── scripts/                      # Development and maintenance scripts
├── docker-compose.yaml           # Docker services (Qdrant, Neo4j)
├── pyproject.toml               # Dependencies and project config
├── CLAUDE.md                    # Project instructions for Claude Code
├── README.md                    # Main documentation
├── .env.example                 # Environment template
└── run_server.py               # Alternative entry point
```

## Source Code Architecture (src/)
```
src/
├── core/                        # Core application framework
│   ├── app.py                   # FastMCP server setup and tool registration
│   └── context.py              # Singleton context management
├── tools/                       # MCP tools by functionality
│   ├── web_tools.py            # Web crawling tools
│   ├── github_tools.py         # GitHub repository tools
│   ├── rag_tools.py            # Vector search and RAG tools
│   └── kg_tools.py             # Knowledge graph tools
├── services/                    # Business logic services
│   ├── unified_indexing_service.py  # Cross-system indexing orchestration
│   ├── rag_service.py          # RAG query processing
│   ├── embedding_service.py     # Embedding generation and caching
│   └── batch_processing/       # Batch processing pipeline
├── clients/                     # External API and database clients
│   ├── qdrant_client.py        # Qdrant vector database client
│   └── llm_api_client.py       # Multi-provider LLM client
├── features/                    # Feature-specific modules
│   └── github/                 # GitHub integration components
│       ├── processors/         # File type processors (Markdown, MDX, Python, etc.)
│       ├── discovery/          # File discovery and filtering
│       ├── repository/         # Git operations and metadata
│       └── services/           # GitHub-specific services
├── k_graph/                     # Knowledge graph and code analysis
│   ├── core/                   # Interfaces and models
│   ├── parsing/                # Multi-language parsing engine
│   │   └── query_patterns/     # Language-specific AST queries
│   ├── analysis/               # Code analysis and validation
│   └── services/               # High-level parsing services
└── utils/                       # Shared utilities
    ├── grammar_initialization.py  # Tree-sitter grammar setup
    ├── file_id_generator.py      # Cross-system file linking
    └── windows_unicode_fix.py    # Windows-specific fixes
```

## Test Organization (tests/)
```
tests/
├── unit/                        # Unit tests by module
│   ├── tools/                  # MCP tools tests
│   ├── services/               # Service layer tests
│   ├── clients/                # Client tests
│   └── core/                   # Core functionality tests
├── specialized/                 # Domain-specific tests
│   ├── embedding/              # Embedding system tests
│   ├── knowledge_graphs/       # Knowledge graph tests
│   └── device_management/      # GPU/CPU device tests
├── infrastructure/              # Infrastructure tests
│   ├── storage/                # Database tests (Qdrant, Redis)
│   └── validation/             # Data validation tests
├── integration/                 # End-to-end tests
├── fixtures/                    # Test data and samples
└── conftest.py                 # Shared pytest fixtures
```

## Entry Points
- **Primary**: `uv run -m src` (uses src/__main__.py)
- **Alternative**: `uv run python run_server.py`
- **Windows Batch**: `start.bat` (wrapper for Windows)

## Configuration Files
- **pyproject.toml**: Dependencies, Python version, dev tools
- **pytest.ini**: Test configuration and markers
- **docker-compose.yaml**: Qdrant and Neo4j services
- **.env**: Environment variables (copy from .env.example)