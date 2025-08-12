# Essential Commands for Development

## Server Management
```bash
# Start MCP server (primary entry point)
uv run -m src

# Alternative entry point
uv run python run_server.py

# Start Docker services first (Windows)
setup.bat

# Start Docker services (Linux/Mac)  
docker-compose up -d
```

## Testing Commands
```bash
# Run all tests
uv run pytest

# Run specific test suites
uv run pytest tests/integration/     # Full workflow tests
uv run pytest tests/tools/           # MCP tool tests  
uv run pytest tests/rag/             # RAG service tests
uv run pytest tests/knowledge_graphs/ # Tree-sitter parser tests

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_web_tools.py -v

# Integration tests (requires Docker services running)
uv run pytest tests/integration_test.py
```

## Code Quality & Formatting
```bash
# Check linting (must pass before commits)
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Format code (standardized style)
uv run ruff format .

# Combined quality check
uv run ruff check --fix . && uv run ruff format .
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

## Package Management (uv only)
```bash
# Install dependencies
uv sync

# Add new dependency
uv add package-name

# Add dev dependency
uv add --dev pytest-package

# NEVER edit pyproject.toml directly - always use uv commands
```

## Knowledge Graph & Analysis
```bash
# Parse repository for hallucination detection
uv run python scripts/query_knowledge_graph.py

# Check AI-generated Python code
uv run python knowledge_graphs/ai_hallucination_detector.py script.py
```

## Environment Setup
```bash
# Copy environment template (first time setup)
cp .env.example .env

# Edit .env with your API keys and configuration
# Required: CHAT_API_KEY, EMBEDDINGS_API_KEY
```

## Windows-Specific Commands
```bash
# Service initialization
setup.bat

# Start server  
start.bat

# Use 'dir' instead of 'ls'
dir

# Use 'type' instead of 'cat'
type filename.txt
```

## Git Workflow
```bash
# Standard git commands work normally
git status
git add .
git commit -m "message"
git push
```

## Performance & Debugging
```bash
# Performance benchmarks
uv run pytest tests/performance/

# Debug specific components
uv run python debug_analysis_test.py
uv run python debug_neo4j_test.py
```