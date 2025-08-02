# Essential Development Commands

## Environment Setup
```bash
# Create virtual environment and install dependencies
uv venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux/macOS
uv pip install -e .
crawl4ai-setup            # Initialize Crawl4AI
```

## Server Operations
```bash
# Start Docker services (Qdrant + Neo4j)
setup.bat                 # Windows automated setup

# Start MCP server (various methods)
start.bat                 # Windows automated startup
uv run -m src             # Module entry point (preferred)
uv run src/crawl4ai_mcp.py  # Direct script execution
uv run run_server.py      # Alternative entry point
```

## Testing
```bash
# Run all tests (quiet mode)
pytest -q

# Run specific test file
pytest tests/test_qdrant_wrapper.py -q

# Run single test method
pytest tests/test_qdrant_wrapper.py::TestQdrantClientWrapper::test_init_default_config -q

# Run with verbose output
pytest tests/ -v

# Run async tests with debugging
pytest -vvs tests/test_integration_docker.py
```

## Code Quality
```bash
# Lint and format code
ruff check .              # Check for linting errors
ruff format .             # Format code according to style guide

# Type checking
mypy .                    # Static type analysis

# Combined quality check (run all)
ruff check . && ruff format . && mypy .
```

## Docker Operations
```bash
# Build Docker image
docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .

# Run in Docker
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag

# Docker Compose operations
docker-compose up -d      # Start services
docker-compose down       # Stop services
docker-compose logs       # View logs
docker-compose logs qdrant  # Service-specific logs
```

## Knowledge Graph Tools
```bash
# Analyze script for AI hallucinations
python knowledge_graphs/ai_hallucination_detector.py [script_path]

# Interactive knowledge graph query (if available)
python knowledge_graphs/query_knowledge_graph.py
```

## Windows-Specific Utilities
```bash
# Check running services
netstat -an | find "6333"   # Qdrant port
netstat -an | find "7474"   # Neo4j HTTP port
netstat -an | find "7687"   # Neo4j Bolt port

# Kill process on port (if needed)
netstat -aon | findstr ":8051"  # Find PID
taskkill /F /PID [PID]          # Kill specific process

# Check Docker status
docker --version
docker-compose ps
```

## Environment Configuration
```bash
# Copy environment template
copy .env.example .env    # Windows
# cp .env.example .env    # Linux/macOS

# Edit environment variables (key variables to set)
# OPENAI_API_KEY=your_key_here
# QDRANT_HOST=localhost
# QDRANT_PORT=6333
# NEO4J_URI=bolt://localhost:7687
```

## Debugging & Monitoring
```bash
# Check service health
curl http://localhost:6333/health    # Qdrant health
curl http://localhost:7474           # Neo4j web interface

# View service dashboards
# Qdrant: http://localhost:6333/dashboard
# Neo4j: http://localhost:7474 (neo4j/password)

# Server health endpoint
curl http://localhost:8051/health    # When server is running
```

## Task Completion Checklist
After completing any code changes, run:
1. `pytest -q` - Ensure all tests pass
2. `ruff check .` - Check for linting errors
3. `ruff format .` - Format code
4. `mypy .` - Verify type annotations