# Task Completion Checklist

## Before Submitting Code Changes

### 1. Code Quality Checks
```bash
# Run linting and fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### 2. Testing Requirements
```bash
# Run relevant test suite based on changes
uv run pytest tests/unit/                    # For service/client changes
uv run pytest tests/specialized/             # For embedding/KG changes
uv run pytest tests/integration/             # For end-to-end functionality

# Ensure no test failures before committing
```

### 3. Documentation Updates
- Update CLAUDE.md if adding new commands or workflows
- Add docstrings for new functions and classes
- Update README.md for new features or configuration changes

### 4. Environment Validation
```bash
# Ensure Docker services are running
docker-compose ps

# Verify MCP server starts successfully
uv run -m src
```

## For New Features

### 1. MCP Tool Development
- Register new tools in `src/core/app.py`
- Add comprehensive error handling
- Include proper type hints and docstrings
- Add unit tests in `tests/unit/tools/`

### 2. Service Layer Changes
- Update unified indexing service if needed
- Ensure proper async patterns
- Add service tests in `tests/unit/services/`

### 3. Database Schema Changes
- Test with clean Qdrant database
- Verify Neo4j compatibility
- Add migration scripts if needed

## Critical Requirements
- **No Unicode characters**: Use ASCII-only text to avoid Windows console errors
- **Environment variables**: Never commit .env files with real API keys
- **Dependencies**: Use `uv add` for new dependencies, never edit pyproject.toml directly
- **Docker services**: Ensure Qdrant, Neo4j, and Redis are running for integration tests

## Pre-Commit Validation
1. All tests pass
2. Code is properly formatted with ruff
3. No linting warnings
4. MCP server starts without errors
5. Documentation is updated
6. No sensitive data in commits