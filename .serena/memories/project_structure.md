# Project Structure

## Root Directory
```
mcp-crawl4ai-rag/
├── src/                    # Main source code
├── tests/                  # Test suites
├── knowledge_graphs/       # Neo4j knowledge graph modules
├── scripts/               # Utility scripts
├── backups/               # Database backups
├── PRPs/                  # Pull Request Proposals
├── .env.example           # Environment configuration template
├── docker-compose.yaml    # Docker services configuration
├── pyproject.toml         # Python project configuration
├── setup.bat              # Windows Docker setup script
├── start.bat              # Windows server startup script
└── README.md              # Project documentation
```

## Source Code Structure (`src/`)
```
src/
├── crawl4ai_mcp.py        # Main MCP server with all tools
├── device_manager.py      # GPU/CPU device management
├── embedding_cache.py     # Redis caching for embeddings
├── embedding_config.py    # Embedding dimension configuration
├── event_loop_fix.py      # Windows ConnectionResetError fix
├── qdrant_wrapper.py      # Qdrant vector database client
├── sparse_vector_types.py # Hybrid search vector types
├── utils.py               # Core utility functions
├── utils/
│   ├── github_processor.py # GitHub repository processing
│   ├── validation.py      # Input validation utilities
│   └── __init__.py        # Utils module initialization
├── clients/               # API client wrappers
├── embeddings/            # Embedding generation modules
├── storage/               # Storage layer abstractions
└── __main__.py            # Module entry point
```

## Test Structure (`tests/`)
```
tests/
├── conftest.py            # Pytest configuration and fixtures
├── test_mcp_basic.py      # Basic MCP functionality tests
├── test_qdrant_wrapper.py # Vector database tests
├── test_hybrid_search.py  # Hybrid search functionality
├── test_github_processor.py # GitHub integration tests
├── test_device_manager.py # GPU/CPU device tests
├── integration_test.py    # Full system integration tests
├── performance_benchmark.py # Performance testing
└── test_*.py              # Additional test modules
```

## Knowledge Graphs (`knowledge_graphs/`)
```
knowledge_graphs/
├── ai_hallucination_detector.py # AI code validation
├── parse_repo_into_neo4j.py     # Repository parsing
├── knowledge_graph_validator.py  # Graph validation
├── hallucination_reporter.py    # Reporting utilities
└── ai_script_analyzer.py        # Script analysis
```

## Configuration Files
- **pyproject.toml**: Python dependencies and project metadata
- **.env.example**: Complete environment configuration template
- **docker-compose.yaml**: Qdrant, Neo4j, and Redis services
- **CLAUDE.md**: Claude Code specific instructions

## Key Entry Points
- **Main Server**: `src/crawl4ai_mcp.py` or `uv run -m src`
- **Alternative**: `run_server.py` for direct execution
- **Windows Setup**: `setup.bat` for Docker services
- **Windows Startup**: `start.bat` for server launch