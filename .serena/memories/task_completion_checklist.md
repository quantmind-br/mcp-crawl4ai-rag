# Task Completion Checklist

## Before Submitting Any Code Changes

### 1. Code Quality Checks (MANDATORY)
```bash
# Must pass without errors
uv run ruff check --fix .
uv run ruff format .
```

### 2. Testing Requirements
```bash
# Run relevant tests based on changes made
uv run pytest                    # All tests
uv run pytest tests/tools/       # If tool changes
uv run pytest tests/services/    # If service changes
uv run pytest tests/integration/ # If major changes

# Specific test patterns
uv run pytest tests/test_web_tools.py      # Web tool changes
uv run pytest tests/test_github_tools.py   # GitHub tool changes  
uv run pytest tests/test_rag_tools.py      # RAG tool changes
uv run pytest tests/test_kg_tools.py       # Knowledge graph changes
```

### 3. Integration Testing (if applicable)
```bash
# Ensure Docker services are running
setup.bat  # Windows
docker-compose up -d  # Linux/Mac

# Run integration tests
uv run pytest tests/integration_test.py
```

### 4. Environment Validation
- [ ] `.env` file configured with required API keys
- [ ] Docker services (Qdrant, Neo4j, Redis) running and accessible
- [ ] No hardcoded secrets or API keys in code
- [ ] Environment variables used for all configuration

### 5. Documentation Updates (if needed)
- [ ] Function docstrings updated for new/modified functions
- [ ] CLAUDE.md updated if new commands or workflows added
- [ ] No Unicode characters used (Windows compatibility)

### 6. Dependency Management
- [ ] If new dependencies added, use `uv add package-name` 
- [ ] Never edit pyproject.toml directly
- [ ] Run `uv sync` after dependency changes

### 7. Performance Considerations
- [ ] Memory usage reasonable for large crawling operations
- [ ] Async patterns maintained for I/O operations
- [ ] Database connections properly managed through context

### 8. Security Checks
- [ ] No API keys logged or exposed
- [ ] Input validation for user-provided URLs and queries
- [ ] SQL injection protection (parameterized queries)
- [ ] Path traversal protection for file operations

## Service-Specific Validation

### Web Crawling Changes
- [ ] Test with various URL types (single pages, sitemaps, GitHub repos)
- [ ] Verify content extraction and chunking
- [ ] Check Qdrant indexing and searchability

### GitHub Tools Changes  
- [ ] Test with public repositories
- [ ] Verify multi-language parsing (Python, JS, Java, etc.)
- [ ] Check both Qdrant and Neo4j indexing if unified tools

### RAG Tools Changes
- [ ] Test search accuracy and relevance
- [ ] Verify filtering by source and file_id
- [ ] Check reranking functionality if enabled

### Knowledge Graph Changes
- [ ] Test Tree-sitter parsing for target languages
- [ ] Verify Neo4j schema and relationships
- [ ] Test hallucination detection accuracy

## Pre-Commit Commands (run in order)
```bash
# 1. Fix code style
uv run ruff check --fix .
uv run ruff format .

# 2. Run tests
uv run pytest

# 3. Verify server starts
uv run -m src &
# Check that it starts without errors, then stop

# 4. Integration check (if major changes)
uv run pytest tests/integration_test.py
```

## Common Issues to Check
- [ ] Async/await patterns consistent
- [ ] JSON serialization works for all return values  
- [ ] Error handling provides meaningful messages
- [ ] Context managers properly closed
- [ ] Unicode/encoding issues resolved (Windows)
- [ ] Performance acceptable for expected usage