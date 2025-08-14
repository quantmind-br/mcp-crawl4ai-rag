# Task Completion Checklist

## Before Committing Code

### 1. Code Quality Checks
```bash
# ALWAYS run these commands before committing:
uv run ruff check --fix .     # Fix linting issues automatically
uv run ruff format .          # Format code according to project standards
```

### 2. Testing Requirements
```bash
# Run relevant tests based on changes:

# For MCP tools changes:
uv run pytest tests/unit/tools/ -v

# For service layer changes:
uv run pytest tests/unit/services/ -v

# For embedding system changes:
uv run pytest tests/specialized/embedding/ -v

# For knowledge graph changes:
uv run pytest tests/specialized/knowledge_graphs/ -v

# Full test suite (recommended for major changes):
uv run pytest
```

### 3. Infrastructure Tests (if Docker services are modified)
```bash
# Ensure Docker services are running:
docker-compose up -d

# Run infrastructure tests:
uv run pytest tests/infrastructure/ -v
uv run pytest tests/integration/ -v
```

### 4. Environment Validation
- Ensure `.env` file is properly configured
- Verify Docker services are running (Qdrant, Neo4j)
- Test MCP server starts successfully:
  ```bash
  uv run -m src
  ```

### 5. Documentation Updates
- Update CLAUDE.md if commands or architecture change
- Update README.md if new features are added
- NO automatic creation of new documentation files unless requested

## When Adding New Dependencies
```bash
# NEVER edit pyproject.toml directly
# ALWAYS use uv commands:
uv add package-name               # Add regular dependency
uv add --dev package-name         # Add development dependency
uv sync                          # Sync after changes
```

## Database Management Tasks
```bash
# If vector database schema changes:
uv run python scripts/clean_qdrant.py
uv run python scripts/define_qdrant_dimensions.py

# If knowledge graph changes:
uv run python scripts/query_knowledge_graph.py
```

## Windows-Specific Considerations
- Ensure no Unicode characters in console output
- Test Windows batch scripts if modified (setup.bat, start.bat)
- Verify Windows-specific dependencies are working

## Integration Testing Checklist
- [ ] MCP server starts without errors
- [ ] Docker services are accessible
- [ ] Basic MCP tools respond correctly
- [ ] Vector search functions properly
- [ ] Knowledge graph operations work
- [ ] No Unicode encoding errors in console

## Pre-Deployment Checklist
- [ ] All tests pass
- [ ] Code is formatted and linted
- [ ] Environment variables are documented
- [ ] Docker services are working
- [ ] No sensitive information in code
- [ ] Memory usage is reasonable
- [ ] Performance is acceptable

## Common Issues to Check
- **Unicode errors**: Ensure ASCII-only output
- **Memory leaks**: Check context managers are properly closed
- **API rate limits**: Verify retry logic is working
- **Docker connectivity**: Ensure services are reachable
- **File permissions**: Check Windows file access issues
- **Embedding dimensions**: Verify consistency across providers