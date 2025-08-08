# Suggested Commands

## Development Setup
```bash
# Install dependencies
uv sync

# Setup Docker services (Windows)
setup.bat

# Setup Docker services (Linux/Mac)
docker-compose up -d

# Create environment file
cp .env.example .env
# Then edit .env with your API keys
```

## Running the Application
```bash
# Start server (Windows)
start.bat

# Start server (Linux/Mac)
uv run -m src

# Alternative entry point
uv run run_server.py

# Check server health
curl http://localhost:8051/health
```

## Testing
```bash
# Run all tests
uv run pytest

# Run specific test files
uv run pytest tests/test_mcp_basic.py
uv run pytest tests/test_qdrant_wrapper.py
uv run pytest tests/integration_test.py

# Run tests with verbose output
uv run pytest -v

# Run tests with coverage
uv run pytest --cov=src
```

## Code Quality
```bash
# Run linter and formatter
uv run ruff check .
uv run ruff check . --fix

# Format code
uv run ruff format .
```

## Utility Scripts
```bash
# Clean Qdrant database
uv run python scripts/clean_qdrant.py

# Fix dimension mismatches
uv run python scripts/define_qdrant_dimensions.py

# Analyze repository for hallucinations
uv run python knowledge_graphs/ai_hallucination_detector.py script.py
```

## Docker Management
```bash
# View service logs
docker-compose logs qdrant
docker-compose logs neo4j
docker-compose logs redis

# Restart services
docker-compose restart

# Stop and remove all services
docker-compose down --volumes

# Check service status
docker-compose ps
```

## Windows-Specific Commands
- Use `start.bat` for complete server startup with checks
- Use `setup.bat` for Docker services initialization
- Commands use Windows-specific tools like `netstat`, `taskkill`, `curl`
- Portuguese language messages in batch files

## Development Workflow
1. Start Docker services: `setup.bat` or `docker-compose up -d`
2. Configure environment: Edit `.env` file
3. Install dependencies: `uv sync`
4. Run tests: `uv run pytest`
5. Start development server: `start.bat` or `uv run -m src`
6. Lint code before commits: `uv run ruff check . --fix`