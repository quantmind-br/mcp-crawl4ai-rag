# Project Structure

## Root Directory
```
mcp-crawl4ai-rag/
├── src/                    # Main source code
├── tests/                  # Test suite
├── knowledge_graphs/       # Knowledge graph and AI analysis tools
├── scripts/               # Utility scripts
├── PRPs/                  # Project Request Patterns (documentation)
├── .serena/               # Serena AI assistant configuration
├── pyproject.toml         # Project configuration and dependencies
├── docker-compose.yaml    # Docker services (Qdrant, Neo4j)
├── Dockerfile            # Container build configuration
├── .env.example          # Environment variable template
└── CRUSH.md              # Development quick reference
```

## Source Code (`src/`)
- **`crawl4ai_mcp.py`**: Main MCP server with all tools and handlers
- **`device_manager.py`**: GPU/CPU device detection and management
- **`qdrant_wrapper.py`**: Qdrant client wrapper and connection management
- **`utils.py`**: Utility functions for embeddings, search, and API clients
- **`__main__.py`**: Module entry point
- **`__init__.py`**: Package initialization

## Key Tools (MCP Functions)
1. **`crawl_single_page`**: Crawl and store single webpage
2. **`smart_crawl_url`**: Intelligent crawling based on URL type
3. **`get_available_sources`**: List available data sources
4. **`perform_rag_query`**: Semantic search with optional filtering
5. **`search_code_examples`**: Code-specific search (conditional)
6. **`parse_github_repository`**: Extract repo to knowledge graph
7. **`check_ai_script_hallucinations`**: Validate AI-generated code
8. **`query_knowledge_graph`**: Explore Neo4j graph database

## Test Structure (`tests/`)
- **`conftest.py`**: Pytest fixtures and test environment setup
- **`test_device_manager.py`**: GPU/CPU device management tests
- **`test_flexible_api_config.py`**: API configuration tests
- **`test_gpu_integration.py`**: GPU acceleration tests
- **`test_integration_docker.py`**: Docker service integration tests
- **`test_mcp_*.py`**: MCP server functionality tests
- **`test_qdrant_*.py`**: Vector database tests
- **`test_utils_integration.py`**: Utility function tests

## Knowledge Graph (`knowledge_graphs/`)
- **`parse_repo_into_neo4j.py`**: Repository code extraction
- **`ai_script_analyzer.py`**: AST-based Python script analysis
- **`knowledge_graph_validator.py`**: Code validation against graph
- **`hallucination_reporter.py`**: Report generation for validation
- **`query_knowledge_graph.py`**: Graph exploration utilities

## Entry Points
- **Server**: `uv run -m src` or `uv run src/crawl4ai_mcp.py`
- **Scripts**: `start.bat` (Windows) for server startup
- **Setup**: `setup.bat` (Windows) for Docker services
- **Direct**: `uv run run_server.py` (alternative entry point)