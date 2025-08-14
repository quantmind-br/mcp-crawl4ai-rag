# Code Style and Conventions

## Language and Version
- **Python 3.12+** as minimum version
- **Type hints** are used throughout the codebase
- **Async/await** patterns for all I/O operations

## Code Formatting and Linting
- **Ruff** for both linting and formatting (version 0.12.7+)
- **Commands**: 
  - `uv run ruff check .` - Check linting
  - `uv run ruff check --fix .` - Auto-fix issues
  - `uv run ruff format .` - Format code

## Naming Conventions
- **snake_case** for variables, functions, modules, and files
- **PascalCase** for classes
- **UPPER_CASE** for constants
- **Private methods/attributes** prefixed with underscore `_`

## Documentation Style
- **Docstrings** in Google style format
- **Type hints** for all function parameters and return values
- **Clear, descriptive variable and function names**

Example:
```python
def create_app() -> FastMCP:
    \"\"\"
    Create and configure the FastMCP application instance.

    This function creates the main application instance with proper configuration
    including server name, host, port, and lifespan management.

    Returns:
        FastMCP: Configured FastMCP server instance ready for tool registration
    \"\"\"
```

## Project Structure Patterns
- **Modular architecture** with clear separation of concerns
- **Feature-based organization** in `src/features/`
- **Singleton patterns** for resource management (Context, models)
- **Factory patterns** for parser and processor selection

## Async Programming
- **Async context managers** for resource management
- **Proper async/await usage** throughout
- **Event loop compatibility** with Windows-specific fixes

## Error Handling
- **Tenacity** for retry logic and fault tolerance
- **Proper exception handling** with specific exception types
- **Logging** throughout the application using Python logging

## Import Organization
- **Standard library imports** first
- **Third-party imports** second
- **Local imports** last
- **Absolute imports** preferred over relative

## Testing Conventions
- **pytest** with async support
- **Hierarchical test organization** by functionality
- **Fixtures** in `conftest.py` for shared test resources
- **Mocking** for external dependencies

## Unicode and Windows Compatibility
- **CRITICAL**: NO Unicode characters (emojis, special symbols) in any output
- **ASCII-only** for all console output and logs
- **Windows-specific optimizations** in utils/windows_unicode_fix.py

## Environment Configuration
- **Environment variables** for all configuration
- **Fallback configurations** for API providers
- **Type validation** for environment variables

## File Organization
- **Clear module boundaries** with `__init__.py` files
- **Single responsibility** per module
- **Consistent file naming** using snake_case

## Comments and Code Quality
- **Self-documenting code** preferred over excessive comments
- **TODO comments** for known technical debt
- **Clear separation** between public and private interfaces