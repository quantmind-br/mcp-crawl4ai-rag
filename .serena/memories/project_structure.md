# Project Structure

## Root Directory
```
mcp-crawl4ai-rag/
├── .env.example              # Environment configuration template
├── .env.test                 # Test environment variables
├── .gitattributes           # Git file handling rules
├── .gitignore               # Git ignore patterns
├── CLAUDE.md                # Project instructions for Claude Code
├── docker-compose.yaml      # Docker services (Qdrant, Neo4j, Redis)
├── Dockerfile               # Container definition
├── LICENSE                  # Project license
├── pyproject.toml           # Python package configuration (use uv for edits)
├── pytest.ini              # Pytest configuration
├── README.md                # Project documentation
├── run_server.py            # Alternative server entry point
├── setup.bat                # Windows Docker setup script
├── start.bat                # Windows server startup script
├── PRPs/                    # Product Requirements and Planning
└── scripts/                 # Utility scripts
```

## Source Code (src/)
```
src/
├── __init__.py              # Package initialization
├── __main__.py              # Main entry point (uv run -m src)
├── device_manager.py        # GPU/CPU device management
├── embedding_cache.py       # Embedding caching logic
├── embedding_config.py      # Embedding model configuration
├── event_loop_fix.py        # Async event loop utilities
├── sparse_vector_types.py   # FastBM25 sparse vector types
├── core/                    # Core application functionality
│   ├── __init__.py
│   ├── app.py               # FastMCP server creation and lifecycle
│   └── context.py           # Singleton context management
├── tools/                   # MCP tools by functionality
│   ├── __init__.py
│   ├── web_tools.py         # Web crawling tools
│   ├── github_tools.py      # GitHub repository tools
│   ├── rag_tools.py         # Vector search and retrieval tools
│   └── kg_tools.py          # Knowledge graph and hallucination detection
├── services/                # Business logic services
│   ├── __init__.py
│   ├── unified_indexing_service.py # Cross-system indexing orchestrator
│   ├── rag_service.py       # RAG operations and search
│   └── ...                  # Other domain services
├── clients/                 # External API and database clients
│   ├── __init__.py
│   ├── qdrant_client.py     # Qdrant vector database client
│   ├── neo4j_client.py      # Neo4j graph database client
│   ├── llm_api_client.py    # OpenAI/LLM API client
│   └── ...                  # Other API clients
├── utils/                   # Utility functions and helpers
│   ├── __init__.py
│   ├── grammar_initialization.py # Tree-sitter grammar setup
│   ├── performance_config.py     # Performance optimization settings
│   └── ...                       # Other utilities
├── features/                # Feature modules (if any)
└── k_graph/                 # Knowledge graph and code parsing
    ├── core/                # Core interfaces and models
    │   ├── interfaces.py    # Language parser interfaces
    │   └── models.py        # Data models for parsing
    ├── parsing/             # Multi-language parsing engine
    │   ├── tree_sitter_parser.py     # Tree-sitter integration
    │   ├── simple_fallback_parser.py # Fallback parser
    │   ├── parser_factory.py         # Parser selection logic
    │   └── query_patterns/           # Language-specific AST queries
    │       ├── python_queries.py     # Python AST patterns
    │       ├── javascript_queries.py # JavaScript/TypeScript patterns
    │       ├── java_queries.py       # Java AST patterns
    │       └── ...                   # Other language patterns
    ├── analysis/            # Code analysis and validation
    │   ├── hallucination_detector.py # AI code validation
    │   ├── validator.py             # Knowledge graph validation
    │   ├── script_analyzer.py       # Script analysis engine
    │   └── reporter.py              # Analysis reporting
    └── services/            # High-level services
        └── repository_parser.py     # Repository processing orchestrator
```

## Tests (tests/)
```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared pytest fixtures
├── README.md                # Test organization guide
├── fixtures/                # Test data and samples
├── unit/                    # Unit tests by module
│   ├── clients/             # Database and API clients tests
│   ├── core/                # Core application tests
│   ├── services/            # Business logic tests
│   ├── tools/               # MCP tools tests
│   └── utils/               # Utility function tests
├── specialized/             # Domain-specific functionality tests
│   ├── embedding/           # Embedding system tests
│   ├── knowledge_graphs/    # Knowledge graph tests
│   └── device_management/   # GPU/CPU device tests
├── infrastructure/          # Infrastructure component tests
│   ├── storage/             # Qdrant, Redis storage tests
│   └── validation/          # Data validation tests
├── integration/             # End-to-end workflow tests
└── performance/             # Performance and benchmark tests
```

## Scripts (scripts/)
```
scripts/
├── README.md                # Scripts documentation
├── build_grammars.py        # Tree-sitter grammar compilation
├── cleanup.bat              # Windows database cleanup script
├── cleanup.sh               # Linux/Mac database cleanup script
├── cleanup_databases.py     # Python database cleanup tool
├── clean_qdrant.py         # Qdrant-specific cleanup
├── define_qdrant_dimensions.py # Fix dimension mismatches
├── query_knowledge_graph.py     # Neo4j query utility
├── setup.bat               # Docker setup (copy of root)
└── start.bat               # Server start (copy of root)
```

## Key Design Principles

### Modular Architecture
- **Separation of concerns** between tools, services, and clients
- **Clear boundaries** between web crawling, RAG, and knowledge graph features
- **Pluggable components** for different language parsers and AI models

### MCP Tool Organization
- **web_tools.py** - Web crawling and URL processing
- **github_tools.py** - Repository indexing and analysis
- **rag_tools.py** - Vector search and document retrieval
- **kg_tools.py** - Knowledge graph queries and hallucination detection

### Cross-System Integration
- **Unified indexing service** coordinates Qdrant and Neo4j operations
- **Consistent file_id linking** between vector and graph databases
- **Context singleton** manages shared resources across MCP tools

### Testing Strategy
- **Hierarchical test organization** by module and functionality
- **Infrastructure tests** for database and external services
- **Integration tests** for end-to-end workflows
- **Performance tests** for optimization validation