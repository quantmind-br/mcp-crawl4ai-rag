# Suggested Development Commands

## Setup and Installation
```bash
# Install dependencies
uv sync

# Setup Docker services (Windows)
setup.bat

# Setup Docker services (Linux/Mac)
docker-compose up -d

# Start MCP server (Windows)
start.bat

# Start MCP server (Linux/Mac)
uv run -m src
```

## Testing Commands
```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/test_mcp_basic.py          # Basic MCP functionality
uv run pytest tests/test_qdrant_wrapper.py     # Vector database tests
uv run pytest tests/test_deepinfra_config.py   # API configuration tests
uv run pytest tests/test_github_processor.py   # Knowledge graph tests

# Performance benchmarks
uv run pytest tests/performance_benchmark.py

# Integration tests (requires Docker services)
uv run pytest tests/integration_test.py

# Run tests with verbose output
uv run pytest -v

# Run specific test pattern
uv run pytest -k "test_pattern_name"
```

## Development Scripts
```bash
# Clean Qdrant database
uv run python scripts/clean_qdrant.py

# Fix Qdrant dimensions for existing collections
uv run python scripts/define_qdrant_dimensions.py

# Knowledge graph tools
uv run python knowledge_graphs/parse_repo_into_neo4j.py <repo_url>
uv run python knowledge_graphs/ai_hallucination_detector.py <script_path>
```

## Docker Service Management
```bash
# View logs
docker-compose logs
docker-compose logs qdrant
docker-compose logs neo4j
docker-compose logs redis

# Restart services
docker-compose restart

# Stop everything
docker-compose down --volumes

# Check service status
docker-compose ps

# Pull latest images
docker-compose pull
```

## Windows Specific Commands
```bash
# Windows batch files
setup.bat         # Initialize Docker stack
start.bat         # Start MCP server with checks

# Check running processes
netstat -an | find ":8051"    # Check MCP server port
netstat -an | find ":6333"    # Check Qdrant
netstat -an | find ":7474"    # Check Neo4j
netstat -an | find ":6379"    # Check Redis

# Kill processes on specific ports
FOR /F "tokens=5" %P IN ('netstat -a -n -o ^| findstr :8051') DO taskkill /F /PID %P
```

## Code Quality Commands
```bash
# Linting and formatting (when ruff is configured)
ruff check .
ruff format .

# Type checking (if mypy is added)
mypy src/

# Security scanning (if bandit is added)
bandit -r src/
```

## Environment Management
```bash
# Create .env from template
cp .env.example .env      # Linux/Mac
copy .env.example .env    # Windows

# Check Python version
python --version

# Check UV version
uv --version

# Update UV
pip install --upgrade uv
```