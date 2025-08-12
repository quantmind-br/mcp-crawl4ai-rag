# AGENTS.md - Development Guide for AI Coding Agents

## Build/Lint/Test Commands

```bash
# Start the MCP server
uv run -m src
# Alternative: uv run python run_server.py

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/unit/tools/test_web_tools.py -v

# Run single test function
uv run pytest tests/unit/tools/test_web_tools.py::test_crawl_single_page -v

# Run tests by category
uv run pytest tests/unit/              # Unit tests
uv run pytest tests/specialized/       # Domain-specific tests
uv run pytest tests/infrastructure/    # Infrastructure tests
uv run pytest tests/integration/       # Integration tests

# Linting and formatting
uv run ruff check .                    # Check for linting issues
uv run ruff check --fix .              # Fix linting issues automatically
uv run ruff format .                   # Format code

# Code coverage
uv run pytest --cov=src --cov-report=html
```

## Code Style Guidelines

### Imports
1. Standard library imports first
2. Third-party imports second (crawl4ai, qdrant, openai, etc.)
3. Local application imports last
4. Use explicit imports rather than wildcard imports

### Formatting
- Use `ruff` for linting and formatting (configured in pyproject.toml)
- Line length: 88 characters (default ruff setting)
- Indentation: 4 spaces
- No trailing whitespace

### Types
- Use type hints for all function signatures and class attributes
- Prefer built-in types (str, int, bool) over typing aliases where possible
- Use Optional[T] for values that can be None
- Use Union[T1, T2] for multiple possible types

### Naming Conventions
- Functions/Variables: snake_case
- Classes: PascalCase
- Constants: UPPER_SNAKE_CASE
- Private methods: _leading_underscore
- MCP Tools: descriptive names with underscores (e.g., `crawl_single_page`)

### Error Handling
- Use specific exception types rather than generic Exception
- Include comprehensive logging with structured messages
- Implement graceful fallback for API failures
- Use tenacity for retry logic where appropriate

### Documentation Style
- Google-style docstrings for all public functions and classes
- Include Args, Returns, and Raises sections in docstrings
- Write clear, concise comments for complex logic
- Keep README.md and CLAUDE.md updated with major changes

### Async Patterns
- Use async/await for all I/O operations
- Use context managers for resource management
- Singleton pattern for shared resources (crawlers, databases)
- Proper async cleanup in lifespan managers

### Unicode Guidelines (Windows Compatibility)
- NEVER use Unicode characters (✅❌🚀) in code, comments, or output
- Use ASCII alternatives for all text
- Avoid emojis and special symbols in any code or documentation files