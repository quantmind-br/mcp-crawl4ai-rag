# Project Structure

## Root Directory
```
mcp-crawl4ai-rag/
├── src/                    # Source code
├── tests/                  # Test suite (hierarchical structure)
├── scripts/                # Utility and setup scripts
├── PRPs/                   # Project Requirements and Plans
├── docker-compose.yaml     # Database services
├── pyproject.toml         # Project dependencies and metadata
├── CLAUDE.md              # Claude Code specific documentation
└── README.md              # Main project documentation
```

## Source Code Structure (src/)
```
src/
├── core/                   # Core application
│   ├── app.py             # Main MCP server and tool registration
│   └── context.py         # Singleton context management
├── tools/                  # MCP tools by functionality
│   ├── web_tools.py       # Web crawling tools
│   ├── github_tools.py    # GitHub integration tools
│   ├── rag_tools.py       # Vector search and RAG tools
│   └── kg_tools.py        # Knowledge graph tools
├── services/              # Business logic services
│   ├── unified_indexing_service.py  # Main indexing orchestrator
│   ├── rag_service.py     # RAG operations
│   ├── embedding_service.py  # Embedding generation
│   └── batch_processing/   # Batch processing pipeline
├── clients/               # Database and API integrations
│   ├── qdrant_client.py   # Qdrant vector database client
│   └── llm_api_client.py  # LLM API client
├── features/              # Feature-specific modules
│   └── github/            # GitHub integration feature
│       ├── core/          # Models and interfaces
│       ├── discovery/     # File discovery logic
│       ├── processors/    # Content processors
│       ├── repository/    # Git operations
│       └── services/      # GitHub-specific services
├── k_graph/               # Knowledge graph system
│   ├── core/              # Interfaces and models
│   ├── parsing/           # Multi-language parsers
│   │   └── query_patterns/  # Language-specific AST patterns
│   ├── analysis/          # Code analysis and validation
│   └── services/          # High-level KG services
└── utils/                 # Utility modules
    ├── grammar_initialization.py  # Tree-sitter grammars
    ├── windows_unicode_fix.py     # Windows encoding fixes
    └── performance_config.py      # Performance optimization
```

## Test Structure (tests/)
Hierarchical organization for better maintainability:
```
tests/
├── unit/                  # Unit tests by module
│   ├── clients/           # Database and API client tests
│   ├── core/              # Core application tests
│   ├── services/          # Business logic tests
│   └── tools/             # MCP tools tests
├── specialized/           # Domain-specific tests
│   ├── embedding/         # Embedding system tests
│   ├── knowledge_graphs/  # Knowledge graph tests
│   └── device_management/ # GPU/CPU device tests
├── infrastructure/        # Infrastructure tests
│   ├── storage/           # Database storage tests
│   └── validation/        # Data validation tests
├── integration/           # End-to-end tests
├── fixtures/              # Test data and samples
└── conftest.py           # Shared pytest fixtures
```

## Key Entry Points
- `src/__main__.py` - Main module entry point
- `run_server.py` - Alternative server launcher
- `src/core/app.py` - MCP server and tool registration