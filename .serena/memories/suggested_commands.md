# Essential Development Commands

## Server Management

### Windows (Primary Platform)
```batch
# Setup Docker services
setup.bat

# Start MCP server
start.bat

# Alternative server start
uv run -m src
uv run python run_server.py
```

### Linux/Mac
```bash
# Start Docker services
docker-compose up -d

# Start MCP server  
uv run -m src
uv run python run_server.py
```

## Package Management (uv - Modern Python)
```bash
# Install all dependencies
uv sync

# Add new dependency
uv add package-name

# Add development dependency
uv add --dev pytest-package

# NEVER edit pyproject.toml directly - always use uv commands
```

## Testing (Hierarchical Structure)
```bash
# Run all tests
uv run pytest

# Test by category
uv run pytest tests/unit/                    # Unit tests by module
uv run pytest tests/specialized/             # Domain-specific tests
uv run pytest tests/infrastructure/          # Infrastructure tests
uv run pytest tests/integration/             # End-to-end tests

# Test specific modules
uv run pytest tests/unit/tools/              # MCP tools tests
uv run pytest tests/specialized/embedding/   # Embedding system tests
uv run pytest tests/specialized/knowledge_graphs/ # Knowledge graph tests

# Test with coverage
uv run pytest --cov=src --cov-report=html

# Test specific file
uv run pytest tests/unit/tools/test_web_tools.py -v
```

## Code Quality (ruff - Fast Linter/Formatter)
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

# Complete database cleanup (WARNING: Destructive)
python scripts/cleanup_databases.py --confirm

# Clean specific database only
python scripts/cleanup_databases.py --qdrant-only --confirm
python scripts/cleanup_databases.py --neo4j-only --confirm

# Safe preview (dry run)
python scripts/cleanup_databases.py --dry-run
```

## Docker Services
```bash
# View service status
docker-compose ps

# View logs
docker-compose logs
docker-compose logs qdrant
docker-compose logs neo4j
docker-compose logs redis

# Stop services
docker-compose down

# Stop and remove volumes (full cleanup)
docker-compose down --volumes
```

## Windows System Commands
```cmd
# List files
dir

# Navigate directories
cd path\to\directory

# Find files
where filename.ext

# Search in files (use rg instead)
findstr "pattern" *.py

# Network checks
netstat -an | find "7474"     # Check Neo4j port
curl http://localhost:6333    # Check Qdrant
```

## Git Operations
```bash
# Check status
git status

# Add and commit
git add .
git commit -m "feat: description"

# View recent commits
git log --oneline -10
```

## Performance Testing
```bash
# Run performance benchmarks
uv run pytest tests/performance/ -v

# Monitor resource usage during tests
uv run pytest tests/integration/ --verbose
```

## Environment Configuration
```bash
# Copy environment template
copy .env.example .env        # Windows
cp .env.example .env          # Linux/Mac

# Edit environment variables
notepad .env                  # Windows
nano .env                     # Linux/Mac
```

## Quick Diagnostics
```bash
# Check service health
curl http://localhost:6333/health    # Qdrant
curl http://localhost:7474           # Neo4j web interface

# Test Python environment
uv run python -c "import src; print('OK')"

# Test MCP tools registration
uv run python -c "from src.core.app import create_app; app = create_app(); print('MCP app OK')"
```