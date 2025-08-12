# Suggested Commands for Development

## Server Management
```bash
# Start MCP server (primary method)
uv run -m src

# Alternative entry point
uv run python run_server.py

# Start with Docker services (Windows)
setup.bat

# Start services (Linux/Mac)
docker-compose up -d
```

## Testing Commands
```bash
# Run all tests
uv run pytest

# Run by test category
uv run pytest tests/unit/                    # Unit tests by module
uv run pytest tests/specialized/             # Domain-specific tests
uv run pytest tests/infrastructure/          # Infrastructure tests
uv run pytest tests/integration/             # End-to-end tests

# Run specific test modules
uv run pytest tests/unit/tools/              # MCP tools tests
uv run pytest tests/specialized/embedding/   # Embedding system tests
uv run pytest tests/specialized/knowledge_graphs/ # Knowledge graph tests

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/unit/tools/test_web_tools.py -v
```

## Code Quality Commands
```bash
# Check linting issues
uv run ruff check .

# Fix linting issues automatically
uv run ruff check --fix .

# Format code
uv run ruff format .
```

## Database Management
```bash
# Clean Qdrant vector database
uv run python scripts/clean_qdrant.py

# Fix dimension mismatches
uv run python scripts/define_qdrant_dimensions.py

# View Docker service logs
docker-compose logs qdrant
docker-compose logs neo4j
docker-compose logs redis

# Restart services
docker-compose restart

# Stop all services
docker-compose down --volumes
```

## Development Utilities
```bash
# Package management (use uv for all operations)
uv add package-name                    # Add dependency
uv add --dev pytest-package           # Add dev dependency
uv sync                               # Sync after pyproject.toml changes

# Knowledge graph utilities
uv run python scripts/query_knowledge_graph.py
```

## Windows-Specific Commands
```bash
# File operations
dir                    # List directory contents
type filename.txt      # Display file contents
findstr "pattern" *.py # Search for pattern in files
cd /d path             # Change directory with drive change

# Process management
tasklist               # List running processes
netstat -an           # Show network connections
```

## Environment Setup
```bash
# Copy environment template
copy .env.example .env

# Edit environment file (Windows)
notepad .env
```