# Project Structure

## Root Directory (`E:\mcp-crawl4ai-rag\`)
- **Entry Points**: `src/__main__.py`, `run_server.py`
- **Configuration**: `pyproject.toml`, `.env.example`, `docker-compose.yaml`
- **Setup Scripts**: `setup.bat`, `start.bat`
- **Testing**: `tests/` directory with comprehensive test suite

## Source Code (`src/`)
- **`crawl4ai_mcp.py`**: Main MCP server with 11+ async tools
- **`qdrant_wrapper.py`**: Qdrant client wrapper with collection management
- **`device_manager.py`**: GPU detection and memory management (Windows optimized)
- **`embedding_cache.py`**: Redis-based embedding caching
- **`event_loop_fix.py`**: Windows-specific event loop handling
- **`utils/`**: GitHub processing, validation, configuration utilities

## Knowledge Graphs (`knowledge_graphs/`)
- **AI Analysis**: Code parsing, AST extraction, hallucination detection
- **Integration**: Neo4j drivers, repository analysis tools

## Test Infrastructure (`tests/`)
- **Basic**: `test_mcp_basic.py` - Core MCP functionality
- **Integration**: `test_integration_docker.py` - Full stack testing
- **Performance**: Validation and benchmark tests
- **Configuration**: Multi-provider API testing

## Scripts & Utilities
- **`scripts/`**: Utilities for maintenance and debugging
- **Batch Files**: Windows-specific setup and startup scripts
- **Docker**: Production testing with containerized services