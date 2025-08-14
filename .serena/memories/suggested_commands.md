# Suggested Commands

## Essential Development Commands

### Project Setup
```bash
# Initial setup and dependency installation
uv sync

# Start Docker services (required for databases)
setup.bat                    # Windows (recommended)
docker-compose up -d         # Linux/Mac alternative
```

### Server Management
```bash
# Start the MCP server (primary method)
uv run -m src

# Alternative server startup
uv run python run_server.py

# Start with Docker services on Windows
start.bat
```

### Testing Commands
```bash
# Run all tests
uv run pytest

# Run by test category (hierarchical structure)
uv run pytest tests/unit/                    # Unit tests by module
uv run pytest tests/specialized/             # Domain-specific tests
uv run pytest tests/infrastructure/          # Infrastructure tests
uv run pytest tests/integration/             # End-to-end tests

# Run specific test modules
uv run pytest tests/unit/tools/              # MCP tools tests
uv run pytest tests/specialized/embedding/   # Embedding system tests
uv run pytest tests/specialized/knowledge_graphs/  # Knowledge graph tests

# Run with coverage report
uv run pytest --cov=src --cov-report=html

# Run specific test file with verbose output
uv run pytest tests/unit/tools/test_web_tools.py -v
```

### Code Quality Commands
```bash
# Check linting (find issues)
uv run ruff check .

# Fix linting issues automatically
uv run ruff check --fix .

# Format code
uv run ruff format .

# Run all code quality checks
uv run ruff check . && uv run ruff format .
```

### Database Management
```bash
# Clean Qdrant vector database
uv run python scripts/clean_qdrant.py

# Fix dimension mismatches in Qdrant
uv run python scripts/define_qdrant_dimensions.py

# Query knowledge graph
uv run python scripts/query_knowledge_graph.py
```

### Docker Services Management
```bash
# View service logs
docker-compose logs              # All services
docker-compose logs qdrant       # Qdrant only
docker-compose logs neo4j        # Neo4j only
docker-compose logs redis        # Redis only

# Check service status
docker-compose ps

# Restart services
docker-compose restart

# Stop and clean everything
docker-compose down --volumes
```

### Package Management (uv)
```bash
# Add new dependency
uv add package-name

# Add development dependency
uv add --dev pytest-package

# Sync dependencies after changes
uv sync

# Never edit pyproject.toml directly - always use uv commands
```

## Windows-Specific Commands
```bash
# Setup script (preferred for Windows)
setup.bat

# Start script
start.bat

# Cleanup script
scripts\cleanup.bat
```

## Utilities and Debugging
```bash
# Debug method count issues
uv run python scripts/debug_method_count.py

# Build tree-sitter grammars
uv run python scripts/build_grammars.py

# Clean up databases
uv run python scripts/cleanup_databases.py
```

## Environment Configuration
```bash
# Copy environment template
copy .env.example .env          # Windows
cp .env.example .env           # Linux/Mac

# Edit environment file
# Add your API keys and configuration
```

## Common Development Workflow
1. `setup.bat` - Start Docker services
2. `uv sync` - Install/update dependencies
3. `uv run ruff check --fix .` - Fix linting
4. `uv run pytest` - Run tests
5. `uv run -m src` - Start server