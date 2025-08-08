# Essential Commands - Crawl4AI MCP RAG

## Setup & Installation Commands

### Initial Setup
```bash
# Clone and install dependencies
git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
cd mcp-crawl4ai-rag
uv sync
```

### Windows Setup (Recommended)
```cmd
# Start Docker services (Qdrant, Neo4j, Redis)
setup.bat

# Start the MCP server
start.bat
```

### Manual Docker Management
```bash
# Start all services
docker-compose up -d

# View service logs
docker-compose logs qdrant
docker-compose logs neo4j
docker-compose logs redis

# Stop all services
docker-compose down --volumes
```

## Development Commands

### Running the Server
```bash
# Primary method (module execution)
uv run -m src

# Alternative method
uv run run_server.py

# With specific transport
TRANSPORT=sse uv run -m src
TRANSPORT=stdio uv run -m src
```

### Testing Commands
```bash
# Run all tests
uv run pytest

# Specific test suites
uv run pytest tests/test_mcp_basic.py          # MCP functionality
uv run pytest tests/test_qdrant_wrapper.py     # Vector database
uv run pytest tests/integration_test.py        # Full integration
uv run pytest tests/test_hybrid_search.py      # Hybrid search
uv run pytest tests/performance_benchmark.py   # Performance tests

# Run tests with verbose output
uv run pytest -v

# Run tests with coverage
uv run pytest --cov=src
```

### Code Quality Commands
```bash
# Lint and auto-fix code issues
uv run ruff check --fix

# Format code
uv run ruff format

# Type checking (if mypy is configured)
mypy src/
```

## Utility Commands

### Database Management
```bash
# Clean vector database
uv run python scripts/clean_qdrant.py

# Fix dimension mismatches
uv run python scripts/define_qdrant_dimensions.py

# Clean all databases
uv run python scripts/cleanup_databases.py
```

### Debugging & Analysis
```bash
# Analyze repository for hallucinations
uv run python knowledge_graphs/ai_hallucination_detector.py script.py

# Debug specific components
python debug_neo4j_test.py
python debug_analysis_test.py
```

## Windows-Specific Commands
```cmd
# Check service status
netstat -an | find "6333"    # Qdrant
netstat -an | find "7474"    # Neo4j  
netstat -an | find "6379"    # Redis
netstat -an | find "8051"    # MCP Server

# Kill process on port (if needed)
taskkill /F /PID <process_id>

# Service health check
curl -s http://localhost:6333/health    # Qdrant health
curl -s http://localhost:7474           # Neo4j web interface
```

## Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit environment configuration
# (Configure API keys, database URLs, RAG strategies)
```

## Package Management
```bash
# Install new dependencies
uv add package_name

# Install development dependencies  
uv add --group dev package_name

# Update all dependencies
uv sync --upgrade

# Check outdated packages
uv tree
```

## Git Workflow
```bash
# Standard git operations
git status
git add .
git commit -m "message"
git push

# Branch management
git checkout -b feature-branch
git merge main
```

## Quick Health Check
```bash
# Verify all services are running
curl -s http://localhost:6333/health && echo "Qdrant OK"
curl -s http://localhost:7474 && echo "Neo4j OK" 
docker exec mcp-redis redis-cli ping && echo "Redis OK"
curl -s http://localhost:8051 && echo "MCP Server OK"
```