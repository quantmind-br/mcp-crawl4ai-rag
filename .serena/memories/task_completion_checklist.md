# Task Completion Checklist

## Pre-Development Setup
- [ ] Ensure Docker services are running (`setup.bat`)
- [ ] Verify `.env` file is configured with API keys
- [ ] Activate virtual environment (`.venv\Scripts\activate`)
- [ ] Install dependencies (`uv pip install -e .`)

## During Development

### Code Changes
- [ ] Follow naming conventions (snake_case, PascalCase, UPPER_CASE)
- [ ] Add type annotations to public functions
- [ ] Use `@dataclass` for simple data structures
- [ ] Return JSON strings from MCP tools
- [ ] Handle errors with structured JSON responses
- [ ] Avoid blocking calls in async MCP tool handlers

### Security & Best Practices
- [ ] Never log or print API keys/secrets
- [ ] Validate and sanitize all user inputs
- [ ] Use logging module instead of print statements
- [ ] Guard file/network access behind appropriate flags
- [ ] Test with mocked external services

## Code Quality Gates (Run Before Commit)

### 1. Testing
```bash
pytest -q                                    # All tests must pass
pytest tests/test_specific_feature.py -v    # Feature-specific tests
```

### 2. Linting
```bash
ruff check .                                 # No linting errors
ruff check . --fix                          # Auto-fix what's possible
```

### 3. Formatting
```bash
ruff format .                                # Apply consistent formatting
```

### 4. Type Checking
```bash
mypy .                                       # No type errors
mypy src/specific_module.py                  # Module-specific checking
```

### 5. Combined Quality Check
```bash
ruff check . && ruff format . && mypy .     # All quality checks
```

## Integration Testing

### Local Services
- [ ] Qdrant accessible at `http://localhost:6333`
- [ ] Neo4j accessible at `http://localhost:7474` (if knowledge graph enabled)
- [ ] Server starts successfully (`uv run -m src`)
- [ ] Health endpoint responds (`curl http://localhost:8051/health`)

### MCP Tools Testing
- [ ] Test basic crawling functionality
- [ ] Verify vector search operations
- [ ] Check RAG query responses
- [ ] Validate error handling and edge cases

## Documentation Updates
- [ ] Update README.md if adding new features
- [ ] Add/update docstrings for new functions
- [ ] Update `.env.example` for new environment variables
- [ ] Update CRUSH.md for new commands or patterns

## Performance Considerations
- [ ] Check memory usage during large crawling operations
- [ ] Verify GPU acceleration works (if using reranking)
- [ ] Test with different RAG strategy combinations
- [ ] Monitor API rate limits and costs

## Final Validation
- [ ] All tests pass: `pytest -q`
- [ ] No linting errors: `ruff check .`
- [ ] Code properly formatted: `ruff format .`
- [ ] No type errors: `mypy .`
- [ ] Server starts without errors
- [ ] Basic MCP functionality works
- [ ] Environment variables properly configured
- [ ] Documentation updated where necessary

## Optional: Knowledge Graph Features
If using knowledge graph functionality:
- [ ] Neo4j service running and accessible
- [ ] Repository parsing works correctly
- [ ] Hallucination detection functions properly
- [ ] Graph queries return expected results

## Windows-Specific Checks
- [ ] `.bat` scripts work correctly
- [ ] Paths use proper Windows format where needed
- [ ] Port conflicts resolved (8051, 6333, 7474, 7687)
- [ ] Environment variables properly loaded from `.env`

## Notes
- Always test with realistic data volumes
- Consider testing with different API providers
- Verify error handling with network/service failures
- Test RAG quality with domain-specific content
- Monitor costs when using external APIs