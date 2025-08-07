# Project Structure

## Root Level Files
- `README.md` - Comprehensive project documentation
- `CLAUDE.md` - Claude Code specific development guidance
- `pyproject.toml` - Python project configuration and dependencies
- `docker-compose.yaml` - Docker services orchestration
- `setup.bat` / `start.bat` - Windows automation scripts
- `.env.example` - Environment configuration template
- `run_server.py` - Alternative server entry point

## Source Code Structure (`src/`)
```
src/
├── crawl4ai_mcp.py         # Main MCP server with all tools
├── __main__.py             # Package entry point
├── utils.py                # Legacy utility functions
├── embedding_config.py     # Embedding configuration singleton
├── device_manager.py       # GPU/CPU device management
├── embedding_cache.py      # Redis-based embedding cache
├── event_loop_fix.py       # Windows asyncio compatibility
├── sparse_vector_types.py  # Qdrant sparse vector configuration
├── clients/                # API and database client wrappers
│   ├── qdrant_client.py    # Qdrant vector database client
│   ├── llm_api_client.py   # Multi-provider LLM API clients
│   └── __init__.py
├── services/               # Business logic services
│   ├── embedding_service.py # Embedding generation service
│   ├── rag_service.py      # RAG query processing service
│   └── __init__.py
├── features/               # Feature-specific implementations
│   ├── github_processor.py # GitHub repository processing
│   └── __init__.py
├── core/                   # Core data structures
│   └── context.py          # MCP context dataclass
├── tools/                  # MCP tool implementations (future expansion)
└── utils/                  # Utility modules
    ├── validation.py       # Input validation functions
    └── __init__.py
```

## Knowledge Graph (`knowledge_graphs/`)
```
knowledge_graphs/
├── parse_repo_into_neo4j.py      # GitHub repository indexing
├── ai_hallucination_detector.py  # AI code validation
├── ai_script_analyzer.py         # AST-based code analysis
├── knowledge_graph_validator.py  # Neo4j validation logic
├── hallucination_reporter.py     # Validation result reporting
└── query_knowledge_graph.py      # Interactive graph exploration
```

## Testing Structure (`tests/`)
```
tests/
├── conftest.py                    # Pytest fixtures and test setup
├── test_mcp_basic.py             # Basic MCP server functionality
├── test_qdrant_wrapper.py        # Vector database tests
├── test_deepinfra_config.py      # API configuration tests
├── test_github_processor.py      # Knowledge graph tests
├── integration_test.py           # Full integration tests
├── performance_benchmark.py      # Performance benchmarks
├── integration_test_hybrid.py    # Hybrid search integration
├── performance_benchmark_hybrid.py # Hybrid search benchmarks
├── run_hybrid_tests.py           # Hybrid search test runner
└── test_*.py                     # Additional test modules
```

## Utility Scripts (`scripts/`)
```
scripts/
├── clean_qdrant.py              # Interactive Qdrant database cleaner
├── define_qdrant_dimensions.py  # Fix dimension mismatches
└── README.md                    # Scripts documentation
```

## Configuration Files
- `.env` - Local environment configuration
- `.env.example` - Environment template with all options
- `.env.test` - Test-specific environment variables
- `.gitignore` - Git ignore rules
- `docker-compose.yaml` - Services: Qdrant, Neo4j, Redis

## Documentation and Projects
- `PRPs/` - Project Requirement Proposals
- `.serena/` - Serena project files and cache
- `.kiro/` - Additional tooling configuration