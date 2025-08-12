# CRUSH.md - Development Guide for Agentic Coding Agents

## Build Commands
```bash
# Install/sync dependencies
uv sync

# Start the MCP server (primary method)
uv run -m src

# Alternative entry point
uv run python run_server.py
```

## Lint/Format Commands
```bash
# Check linting issues
uv run ruff check .

# Fix linting issues automatically
uv run ruff check --fix .

# Format code
uv run ruff format .
```

## Test Commands
```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test directory
uv run pytest tests/integration/
uv run pytest tests/tools/
uv run pytest tests/rag/

# Run specific test file
uv run pytest tests/test_web_tools.py -v

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Run performance benchmarks
uv run pytest tests/performance/

# Run a single test function
uv run pytest path/to/test_file.py::test_function_name -v
```

## Code Style Guidelines

### Naming Conventions
- Functions/Variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private Methods: `_leading_underscore`

### Imports
- Use Ruff for import sorting and organization
- Group imports in order: standard library, third-party, local imports
- Use absolute imports when possible

### Type Hints
- Use throughout codebase for better IDE support and documentation
- Include return types for all functions
- Use generics where appropriate

### Documentation
- Google-style docstrings for public functions and classes
- Include module headers with purpose and main functionality description

### Code Organization
- Async/Await: Used throughout for I/O operations
- Context Managers: For resource management (databases, web crawlers)
- Singleton Patterns: Used for shared resources (contexts, models)
- Error Handling: Comprehensive exception handling with informative messages

### Testing Conventions
- Mirror source structure in tests/ directory
- Use pytest fixtures in conftest.py for common setup
- Use pytest-asyncio for async test functions
- Mock external services (APIs, databases) in unit tests
- Test full workflows with real or containerized services in integration tests