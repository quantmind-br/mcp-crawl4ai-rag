# Suggested Commands for Development

## Server Management
```bash
# Start the MCP server (primary method)
uv run -m src

# Alternative entry point
uv run python run_server.py

# Windows batch scripts
setup.bat          # Start Docker services
start.bat          # Start MCP server

# Start Docker services manually
docker-compose up -d
```

## Package Management (use uv exclusively)
```bash
# Install/sync dependencies
uv sync

# Add new dependency
uv add package-name

# Add development dependency
uv add --dev pytest-package

# NEVER edit pyproject.toml directly - always use uv commands
```

## Testing (hierarchical test structure)
```bash
# Run all tests
uv run pytest

# Run by category
uv run pytest tests/unit/                    # Unit tests by module
uv run pytest tests/specialized/             # Domain-specific tests
uv run pytest tests/infrastructure/          # Infrastructure tests
uv run pytest tests/integration/             # End-to-end tests

# Run specific modules
uv run pytest tests/unit/tools/              # MCP tools tests
uv run pytest tests/specialized/embedding/   # Embedding system tests
uv run pytest tests/specialized/knowledge_graphs/ # Knowledge graph tests

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/unit/tools/test_web_tools.py -v
```

## Code Quality
```bash
# Check linting
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

# Clean up all databases
uv run python scripts/cleanup_databases.py

# View Docker service logs
docker-compose logs qdrant
docker-compose logs neo4j

# Restart services
docker-compose restart
```

## Knowledge Graph Operations
```bash
# Query knowledge graph
uv run python scripts/query_knowledge_graph.py

# Debug method count
uv run python scripts/debug_method_count.py
```

## Windows-Specific Commands
```bash
# System utilities (Windows equivalents)
dir              # List files (instead of ls)
cd               # Change directory
findstr          # Search text (instead of grep)
where            # Find files (instead of which)
type             # Display file content (instead of cat)
```

## Git Operations
```bash
git status       # Check working tree status
git add .        # Stage all changes
git commit -m "message"  # Commit changes
git pull         # Pull latest changes
git push         # Push changes
```

## Environment Setup
```bash
# Copy environment template
copy .env.example .env    # Windows
cp .env.example .env      # Linux/Mac

# Edit environment file with your API keys and configuration
```

## Docker Management
```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Stop with volume cleanup
docker-compose down --volumes

# View running containers
docker ps

# View service logs
docker-compose logs [service_name]
```