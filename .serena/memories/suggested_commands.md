# Essential Commands for Development

## Setup & Installation
```bash
# Install dependencies and setup environment
uv sync

# Start Docker services (Qdrant + Neo4j + Redis)
setup.bat          # Windows
# or: docker-compose up -d

# Configure environment from example
cp .env.example .env
# Edit .env with your API keys
```

## Development Workflow
```bash
# Start the MCP server
start.bat          # Windows (includes all checks)
# or: uv run -m src

# Direct server start
uv run run_server.py
```

## Testing Commands
```bash
# Run all tests
uv run pytest

# Specific test suites
uv run pytest tests/test_mcp_basic.py          # MCP functionality
uv run pytest tests/test_qdrant_wrapper.py     # Vector database
uv run pytest tests/integration_test.py        # Full integration
uv run pytest tests/performance_benchmark.py   # Performance tests
```

## Maintenance Commands
```bash
# Clean vector database
uv run python scripts/clean_qdrant.py

# Fix collection dimension issues
uv run python scripts/fix_qdrant_dimensions.py

# Repository analysis for knowledge graph
uv run python knowledge_graphs/parse_repo_into_neo4j.py <repo_url>

# AI code validation
uv run python knowledge_graphs/ai_hallucination_detector.py script.py
```

## Docker Management
```bash
# View service logs
docker-compose logs [qdrant|neo4j|redis]

# Service status
docker-compose ps

# Clean restart (removes all data)
docker-compose down --volumes
docker-compose up -d

# Health checks
curl http://localhost:6333/health      # Qdrant
curl http://localhost:7474             # Neo4j web interface
redis-cli ping                         # Redis
```

## Windows-Specific
```bash
# Check Python and uv versions
python --version
uv --version

# Environment verification
echo %CHAT_API_KEY%
echo %EMBEDDINGS_API_KEY%
```