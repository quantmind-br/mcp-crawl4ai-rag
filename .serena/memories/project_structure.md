# Project Structure - Crawl4AI MCP RAG

## Root Directory Structure
```
mcp-crawl4ai-rag/
├── src/                    # Main source code
├── tests/                  # Test suites
├── scripts/                # Utility scripts
├── knowledge_graphs/       # Knowledge graph components
├── PRPs/                   # Project Requirements and Plans
├── .env.example           # Environment configuration template
├── pyproject.toml         # Python project configuration
├── docker-compose.yaml    # Docker services definition
├── setup.bat              # Windows Docker setup script
├── start.bat              # Windows server startup script
└── README.md              # Project documentation
```

## Source Code Organization (src/)
```
src/
├── core/                  # Application core
│   ├── app.py            # Main application and server setup
│   └── context.py        # Context management and singletons
├── tools/                # MCP tool implementations
│   ├── rag_tools.py      # RAG query tools
│   ├── web_tools.py      # Web crawling tools
│   ├── github_tools.py   # GitHub integration tools
│   └── kg_tools.py       # Knowledge graph tools
├── services/             # Business logic services
│   └── rag_service.py    # RAG service implementation
├── clients/              # External service clients
│   └── qdrant_wrapper.py # Qdrant database client
├── utils/                # Utility functions
├── features/             # Feature-specific modules
├── device_manager.py     # GPU/CPU device management
├── embedding_cache.py    # Embedding caching logic
├── embedding_config.py   # Embedding configuration
├── __main__.py          # Module entry point
└── __init__.py          # Package initialization
```

## Test Structure (tests/)
```
tests/
├── fixtures/             # Test data and fixtures
├── conftest.py          # Pytest configuration
├── test_*.py            # Individual test modules
├── integration_test.py  # Integration tests
├── performance_*.py     # Performance benchmarks
└── test_summary.md      # Test documentation
```

## Key Test Categories
- **MCP Tests**: Core MCP functionality (`test_mcp_basic.py`)
- **Database Tests**: Qdrant integration (`test_qdrant_wrapper.py`)  
- **Service Tests**: RAG service logic (`test_rag_service.py`)
- **Integration Tests**: End-to-end functionality (`integration_test.py`)
- **Performance Tests**: Benchmarks (`performance_benchmark.py`)

## Scripts Directory
```
scripts/
├── clean_qdrant.py           # Vector database cleanup
├── define_qdrant_dimensions.py # Fix dimension mismatches
├── cleanup_databases.py      # Clean all databases
└── README.md                 # Script documentation
```

## Configuration Directories
- **`.claude/`**: Claude Code integration settings
- **`.crush/`**: Project-specific caching
- **`.cursor/`**: Cursor IDE settings  
- **`.gemini/`**: Gemini integration commands
- **`.kiro/`**: Additional tooling configuration
- **`.serena/`**: Serena semantic analysis cache

## Docker & Deployment
- **`docker-compose.yaml`**: Defines Qdrant, Neo4j, and Redis services
- **`Dockerfile`**: Application containerization (if needed)
- **`setup.bat`**: Windows Docker initialization
- **`start.bat`**: Windows server startup with health checks

## Environment & Configuration
- **`.env.example`**: Comprehensive environment template
- **`pyproject.toml`**: Python dependencies and project metadata
- **`pytest.ini`**: Test runner configuration
- **`.gitignore`**: Version control exclusions

## Entry Points
- **`src/__main__.py`**: Primary module entry point
- **`run_server.py`**: Alternative server startup script
- **CLI Integration**: `uv run -m src` (recommended)

## Key Architectural Patterns
- **Modular Design**: Clear separation of concerns across directories
- **Async Architecture**: Event-driven with async/await patterns
- **Singleton Pattern**: Context and service management
- **Factory Pattern**: Application and tool creation
- **Service Layer**: Business logic abstraction
- **Client Abstraction**: External service wrappers