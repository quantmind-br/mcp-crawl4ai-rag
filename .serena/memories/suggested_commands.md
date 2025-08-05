# Suggested Commands

## Setup and Installation
```bash
# Install dependencies
uv sync

# Setup Docker services (Windows)
setup.bat
# or (Linux/Mac)
docker-compose up -d

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start MCP server (Windows)
start.bat
# or (Linux/Mac)
uv run -m src
```

## Development Commands
```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/test_mcp_basic.py          # Basic MCP functionality
uv run pytest tests/test_qdrant_wrapper.py     # Vector database tests
uv run pytest tests/test_deepinfra_config.py   # API configuration tests
uv run pytest tests/test_github_processor.py   # Knowledge graph tests
uv run pytest tests/integration_test.py        # Full integration tests
uv run pytest tests/performance_benchmark.py   # Performance benchmarks

# Run hybrid search tests
uv run pytest tests/test_hybrid_search.py
uv run python run_hybrid_tests.py

# Run linting
uv run ruff check
uv run ruff format
```

## Utility Scripts
```bash
# Clean Qdrant database
uv run python scripts/clean_qdrant.py

# Fix Qdrant dimensions for existing collections
uv run python scripts/fix_qdrant_dimensions.py

# Analyze repository for hallucinations
uv run python knowledge_graphs/ai_hallucination_detector.py script.py

# Parse repository into Neo4j knowledge graph
uv run python knowledge_graphs/parse_repo_into_neo4j.py <repo_url>

# Validate installation
uv run python validate-installation.py
```

## Docker Management
```bash
# View service logs
docker-compose logs qdrant
docker-compose logs neo4j
docker-compose logs redis

# Restart services
docker-compose restart

# Stop all services
docker-compose down --volumes

# Check service status
docker-compose ps
```

## Windows-Specific Commands
```cmd
REM Setup Docker services
setup.bat

REM Start MCP server
start.bat

REM Check Docker status
docker --version
docker-compose ps
```

## MCP Client Integration
```bash
# Add to Claude Code CLI
claude mcp add-json crawl4ai-rag '{"type":"http","url":"http://localhost:8051/sse"}' --scope user
```

## Alternative Startup Methods
```bash
# Direct Python execution
uv run python run_server.py

# Module execution
uv run -m src

# Development with hot reload
uv run python src/crawl4ai_mcp.py
```