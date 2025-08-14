# Task Completion Checklist

## Before Starting Development
- [ ] **Environment Setup**
  - [ ] Copy `.env.example` to `.env` and configure API keys
  - [ ] Run `setup.bat` (Windows) or `docker-compose up -d` (Linux/Mac)
  - [ ] Verify services: Qdrant (6333), Neo4j (7474), Redis (6379)
  - [ ] Run `uv sync` to install dependencies

## During Development

### Code Quality Standards
- [ ] **Code Style**
  - [ ] Use snake_case for functions and variables
  - [ ] Add comprehensive docstrings with type hints
  - [ ] Follow async/await patterns for I/O operations
  - [ ] Use descriptive variable names

- [ ] **Type Safety**
  - [ ] Add type hints to all function parameters
  - [ ] Specify return types for all functions
  - [ ] Use Union types for optional parameters

- [ ] **Error Handling**
  - [ ] Wrap external API calls in try-catch blocks
  - [ ] Provide graceful degradation for optional features
  - [ ] Log errors with appropriate context

### Testing Requirements
- [ ] **Unit Tests**
  - [ ] Create tests for new functions in appropriate `tests/unit/` subdirectory
  - [ ] Mock external dependencies (APIs, databases)
  - [ ] Test error conditions and edge cases

- [ ] **Integration Tests** (if applicable)
  - [ ] Add end-to-end tests for new MCP tools
  - [ ] Test database integration with Docker services
  - [ ] Verify cross-system functionality (Qdrant + Neo4j)

## After Code Changes

### Immediate Validation
- [ ] **Linting and Formatting**
  ```bash
  uv run ruff check .
  uv run ruff check --fix .
  uv run ruff format .
  ```

- [ ] **Type Checking** (if available)
  ```bash
  # Check if mypy is configured
  uv run mypy src/ || echo "mypy not configured"
  ```

- [ ] **Unit Tests**
  ```bash
  uv run pytest tests/unit/ -v
  ```

### Comprehensive Testing
- [ ] **Full Test Suite**
  ```bash
  uv run pytest
  ```

- [ ] **Test Coverage** (optional but recommended)
  ```bash
  uv run pytest --cov=src --cov-report=html
  ```

- [ ] **Integration Tests** (requires Docker services)
  ```bash
  uv run pytest tests/integration/ -v
  ```

### Service Validation
- [ ] **MCP Server Startup**
  ```bash
  # Test server can start without errors
  uv run -m src
  # Or alternative
  uv run python run_server.py
  ```

- [ ] **Database Connectivity**
  - [ ] Qdrant: `curl http://localhost:6333/health`
  - [ ] Neo4j: `curl http://localhost:7474`
  - [ ] Redis: `docker exec mcp-redis redis-cli ping`

### Documentation Updates
- [ ] **Code Documentation**
  - [ ] Update docstrings for modified functions
  - [ ] Add inline comments for complex logic
  - [ ] Update type hints if function signatures changed

- [ ] **Memory Files** (if major changes)
  - [ ] Update relevant Serena memory files if architecture changed
  - [ ] Add new commands to `suggested_commands.md` if needed

## Before Task Completion

### Final Validation
- [ ] **Clean Test Run**
  ```bash
  # Start fresh with clean databases if needed
  python scripts/cleanup_databases.py --confirm
  uv run pytest
  ```

- [ ] **Performance Check** (for significant changes)
  ```bash
  uv run pytest tests/performance/ -v
  ```

- [ ] **Cross-Platform Compatibility** (if applicable)
  - [ ] Test on Windows using `.bat` scripts
  - [ ] Verify Unix commands work on Linux/Mac

### Code Review Checklist
- [ ] **Security**
  - [ ] No API keys or secrets in code
  - [ ] Proper input validation for user data
  - [ ] Safe file path handling

- [ ] **Performance**
  - [ ] Async patterns used for I/O operations
  - [ ] Batch operations for database calls
  - [ ] Appropriate use of connection pooling

- [ ] **Maintainability**
  - [ ] Code follows established patterns
  - [ ] Functions have single responsibility
  - [ ] Dependencies are properly managed with `uv`

## Troubleshooting Common Issues

### Test Failures
- [ ] **Docker Services Down**
  ```bash
  setup.bat  # Windows
  docker-compose up -d  # Linux/Mac
  ```

- [ ] **Port Conflicts**
  ```bash
  netstat -an | find "6333"  # Check Qdrant port
  netstat -an | find "7474"  # Check Neo4j port
  ```

- [ ] **Environment Variables**
  ```bash
  # Verify .env file exists and has required keys
  cat .env | grep API_KEY
  ```

### Performance Issues
- [ ] **Database Dimension Mismatches**
  ```bash
  uv run python scripts/define_qdrant_dimensions.py
  ```

- [ ] **Memory Usage**
  ```bash
  # Monitor during heavy operations
  docker stats
  ```

## Success Criteria
- [ ] All tests pass (`uv run pytest`)
- [ ] Code style compliant (`uv run ruff check .`)
- [ ] MCP server starts successfully
- [ ] New functionality works with existing integrations
- [ ] Documentation updated appropriately