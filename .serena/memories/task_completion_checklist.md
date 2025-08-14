# Task Completion Checklist

## Before Committing Code Changes

### 1. Code Quality Checks (MANDATORY)
```bash
# Fix linting issues automatically
uv run ruff check --fix .

# Format code properly
uv run ruff format .

# Verify no linting errors remain
uv run ruff check .
```

### 2. Testing Requirements
```bash
# Run relevant test suites
uv run pytest tests/unit/                    # Always run unit tests
uv run pytest tests/specialized/             # If working on specialized features
uv run pytest tests/integration/             # If making significant changes

# For specific modules worked on:
uv run pytest tests/unit/tools/              # If MCP tools were modified
uv run pytest tests/specialized/embedding/   # If embedding system was modified
uv run pytest tests/specialized/knowledge_graphs/  # If KG system was modified

# Run with coverage if adding new functionality
uv run pytest --cov=src --cov-report=html
```

### 3. Environment and Dependencies
```bash
# Ensure dependencies are properly managed
uv sync

# If new dependencies were added, verify they're in pyproject.toml
# NEVER manually edit pyproject.toml - use uv commands only
```

### 4. Database Services Check
```bash
# Verify required services are running
docker-compose ps

# Check service health
curl -s http://localhost:6333/health    # Qdrant
curl -s http://localhost:7474/          # Neo4j
```

### 5. Windows Compatibility Check
- **Unicode Characters**: Ensure no Unicode/emoji characters in console output
- **Path Handling**: Use pathlib for cross-platform paths
- **Line Endings**: Verify Windows line endings are handled correctly

### 6. Documentation Updates
- Update CLAUDE.md if architecture changes
- Update README.md if user-facing features change
- Add/update docstrings for new functions/classes
- Update .env.example if new environment variables added

## After Implementation

### 7. Manual Testing
```bash
# Start the server
uv run -m src

# Test basic functionality if MCP tools were modified
# Use MCP client or direct tool testing
```

### 8. Memory Management
- Verify no memory leaks in long-running operations
- Check resource cleanup in context managers
- Test graceful shutdown behavior

### 9. Performance Considerations
- Profile CPU-intensive operations if performance-critical code was added
- Verify batch processing settings are appropriate
- Check GPU acceleration works if reranking features modified

## Pre-Commit Validation Script
Create a validation script that runs:
```bash
#!/bin/bash
# validation.bat (Windows) or validation.sh (Linux/Mac)

echo "Running code quality checks..."
uv run ruff check --fix . && uv run ruff format .

echo "Running tests..."
uv run pytest tests/unit/

echo "Checking dependencies..."
uv sync

echo "Validation complete!"
```

## Red Flags - DO NOT COMMIT IF:
- Linting errors remain after `uv run ruff check`
- Unit tests fail
- New code lacks proper type hints
- Windows Unicode errors occur during testing
- Docker services required but not documented
- Environment variables added without .env.example update
- Performance significantly degraded without documentation

## Integration Testing Checklist
For major changes, also verify:
- MCP server starts successfully
- All registered tools are accessible
- Database connections work properly
- Cross-platform compatibility maintained
- Error handling works as expected