# Task Completion Checklist - Crawl4AI MCP RAG

## Code Quality Requirements

### 1. Linting and Formatting
```bash
# REQUIRED: Run before committing any code changes
uv run ruff check --fix      # Auto-fix linting issues
uv run ruff format           # Format code consistently
```

### 2. Type Checking (if available)
```bash
# Check type annotations (if mypy configured)
mypy src/
```

## Testing Requirements

### 3. Unit Tests
```bash
# REQUIRED: Run relevant test suites
uv run pytest tests/test_<relevant_module>.py -v

# For core changes, run comprehensive tests
uv run pytest tests/test_core_*.py
```

### 4. Integration Tests
```bash
# REQUIRED for major changes: Run integration tests
uv run pytest tests/integration_test.py -v

# For RAG/search changes:
uv run pytest tests/test_rag_service.py -v
```

### 5. Performance Validation
```bash
# For performance-critical changes: Run benchmarks
uv run pytest tests/performance_benchmark.py
```

## Service Validation

### 6. Database Health Checks
```bash
# Verify Qdrant is accessible
curl -s http://localhost:6333/health

# Check Neo4j availability (if using knowledge graph)
curl -s http://localhost:7474

# Verify Redis (if using caching)
docker exec mcp-redis redis-cli ping
```

### 7. Server Startup Validation
```bash
# Test server starts correctly
uv run -m src
# Verify no startup errors in logs
```

## Environment and Configuration

### 8. Environment Configuration
- [ ] Verify `.env` file contains required API keys
- [ ] Check environment variables are properly loaded
- [ ] Validate configuration doesn't expose secrets

### 9. Docker Services (if modified)
```bash
# Restart and verify Docker services
docker-compose down --volumes
docker-compose up -d

# Check service logs for errors
docker-compose logs qdrant
docker-compose logs neo4j
docker-compose logs redis
```

## Documentation and Code Review

### 10. Documentation Updates
- [ ] Update docstrings for new/modified functions
- [ ] Update README.md if public API changed
- [ ] Add comments for complex logic
- [ ] Update type hints for all new functions

### 11. Code Review Checklist
- [ ] Follow naming conventions (snake_case for functions/variables)
- [ ] Include proper error handling with JSON error responses
- [ ] Use async/await patterns for I/O operations
- [ ] Add comprehensive docstrings with Args/Returns sections
- [ ] Ensure proper resource cleanup in async contexts

## Deployment Readiness

### 12. Final Integration Test
```bash
# Complete end-to-end test
uv run pytest tests/integration_test.py -v --tb=short
```

### 13. Clean State Verification
```bash
# Ensure clean database state for tests
uv run python scripts/clean_qdrant.py  # If needed
```

## Pre-Commit Verification Commands
```bash
# Run this sequence before every commit:
uv run ruff check --fix
uv run ruff format  
uv run pytest tests/ -x  # Stop on first failure
```

## Performance Verification (for changes affecting core functionality)
```bash
# Run performance benchmarks
uv run pytest tests/performance_benchmark.py
uv run pytest tests/performance_benchmark_hybrid.py
```

## Emergency Rollback Procedures
If issues arise after deployment:
1. **Stop the server**: `Ctrl+C` or kill MCP server process
2. **Check logs**: Review console output for error messages
3. **Restart services**: `docker-compose restart`
4. **Clean state**: `uv run python scripts/cleanup_databases.py`
5. **Restart server**: `uv run -m src`

## Success Criteria
- [ ] All linting passes without errors
- [ ] All tests pass
- [ ] Server starts without errors
- [ ] Database health checks pass
- [ ] No security issues (secrets properly managed)
- [ ] Documentation is up-to-date
- [ ] Code follows project conventions