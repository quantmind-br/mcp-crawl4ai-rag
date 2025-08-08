# Project Structure

## Root Directory
```
mcp-crawl4ai-rag/
├── src/                     # Main source code
├── tests/                   # Test suite
├── scripts/                 # Utility scripts
├── knowledge_graphs/        # Neo4j knowledge graph tools
├── .claude/                 # Claude IDE configuration
├── .serena/                 # Serena project files
├── pyproject.toml          # Python project configuration
├── docker-compose.yaml     # Docker services
├── setup.bat              # Windows setup script
├── start.bat              # Windows server start script
└── run_server.py          # Alternative entry point
```

## Source Code Structure (`src/`)
```
src/
├── __main__.py             # Module entry point
├── clients/                # External service clients
│   ├── llm_api_client.py  # LLM API integration
│   ├── qdrant_client.py   # Qdrant vector database
│   └── __init__.py
├── core/                   # Application core
│   ├── app.py             # Main application setup
│   ├── context.py         # Shared context management
│   └── __init__.py
├── features/               # Feature implementations
│   ├── github_processor.py # GitHub repository processing
│   └── __init__.py
├── services/               # Business logic services
│   ├── embedding_service.py      # Embedding generation
│   ├── rag_service.py            # RAG query processing
│   ├── unified_indexing_service.py # Multi-destination indexing
│   └── __init__.py
├── tools/                  # MCP tool implementations
│   ├── github_tools.py    # GitHub-related MCP tools
│   ├── kg_tools.py        # Knowledge graph tools
│   ├── rag_tools.py       # RAG query tools
│   ├── web_tools.py       # Web crawling tools
│   └── __init__.py
├── utils/                  # Utility functions
│   ├── file_id_generator.py      # File ID management
│   ├── grammar_initialization.py # Tree-sitter setup
│   ├── validation.py             # Input validation
│   └── __init__.py
├── device_manager.py      # GPU/CPU detection
├── embedding_cache.py     # Redis caching
├── embedding_config.py    # Embedding configuration
├── event_loop_fix.py      # Windows async fixes
└── sparse_vector_types.py # Vector type definitions
```

## Testing Structure (`tests/`)
- **Unit Tests**: Individual component testing
- **Integration Tests**: Full system testing with databases
- **Performance Tests**: Benchmarking and load testing
- **Fixtures**: Test data and mock configurations

## Key Entry Points
- `src/__main__.py`: Primary entry point (`uv run -m src`)
- `run_server.py`: Alternative entry point for standalone execution
- `setup.bat`: Windows Docker services setup
- `start.bat`: Complete Windows server startup script