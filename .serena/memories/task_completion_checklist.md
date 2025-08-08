# Task Completion Checklist

## Pre-Completion Requirements
When completing any development task in this project, ensure these steps are followed:

### 1. Code Quality Checks
```bash
# Run linter and fix issues
uv run ruff check . --fix

# Verify no remaining linting issues
uv run ruff check .
```

### 2. Testing Requirements
```bash
# Run all tests to ensure no regressions
uv run pytest

# Run specific integration tests if applicable
uv run pytest tests/integration_test.py
uv run pytest tests/test_qdrant_wrapper.py
```

### 3. Service Validation
```bash
# Ensure Docker services are running
docker-compose ps

# Test server startup
uv run -m src
# Or use start.bat on Windows

# Verify MCP server responds
curl http://localhost:8051/health
```

### 4. Environment Configuration
- Verify `.env` file is properly configured
- Check all required API keys are present
- Ensure database connections work (Qdrant, Neo4j, Redis)

### 5. Documentation Updates
- Update docstrings for new/modified functions
- Update README.md if public API changes
- Add comments for complex logic

### 6. Performance Considerations
- Check memory usage for large operations
- Verify async patterns are properly implemented
- Test with realistic data volumes

## Critical Validation Steps
1. **API Integration**: Test with actual API providers (OpenAI, DeepInfra)
2. **Database Operations**: Verify Qdrant and Neo4j operations work correctly
3. **Error Handling**: Test failure scenarios and fallback mechanisms
4. **Resource Cleanup**: Ensure proper cleanup of connections and resources
5. **Cross-Platform**: Test on Windows (primary target platform)

## Pre-Commit Checklist
- [ ] All tests pass (`uv run pytest`)
- [ ] Linting passes (`uv run ruff check .`)
- [ ] Server starts successfully (`uv run -m src`)
- [ ] No regression in existing functionality
- [ ] New features have appropriate tests
- [ ] Documentation is updated
- [ ] Environment variables documented if added
- [ ] Error handling implemented for new features

## Deployment Readiness
- [ ] Docker services start cleanly (`docker-compose up -d`)
- [ ] MCP server responds to health checks
- [ ] All required environment variables documented
- [ ] Integration tests pass with real services
- [ ] Performance requirements met
- [ ] Windows compatibility verified (primary platform)