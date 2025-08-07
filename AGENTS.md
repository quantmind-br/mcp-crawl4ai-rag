# AGENTS.md

## Build/Lint/Test Commands
```bash
uv run pytest tests/test_specific.py::test_function  # Single test
uv run pytest -xvs tests/test_specific.py              # Verbose single file
uv run ruff check src/                                 # Lint
uv run ruff format src/                                # Format
uv run pytest                                          # All tests
```

## Code Style Guidelines
- **Types**: Use type hints for all functions/parameters
- **Async**: All I/O operations must be async/await
- **Imports**: Group stdlib, third-party, local; absolute imports preferred
- **Naming**: snake_case for functions/vars, PascalCase for classes, UPPER_SNAKE for constants
- **Error Handling**: Use tenacity for retries, graceful fallbacks (GPU→CPU)
- **No emojis** in code
- **Docstrings**: Use Google style for public functions
- **Line length**: 88 characters (Black/ruff default)