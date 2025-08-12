# Code Style and Conventions

## Language Standards
- **Python 3.12+**: Use modern Python features and async/await patterns
- **Type Hints**: Required for all function signatures and class attributes
- **Docstrings**: Google-style docstrings for all public functions and classes

## Code Organization
### Import Order
1. Standard library imports
2. Third-party imports (crawl4ai, qdrant, openai, etc.)
3. Local application imports

### Naming Conventions
- **Functions/Variables**: snake_case
- **Classes**: PascalCase
- **Constants**: UPPER_SNAKE_CASE
- **Private methods**: _leading_underscore
- **MCP Tools**: descriptive names with underscores (e.g., `crawl_single_page`)

## Async Patterns
- Use async/await for all I/O operations
- Context managers for resource management
- Singleton pattern for shared resources (crawlers, databases)

## Error Handling
- Use specific exception types
- Comprehensive logging with structured messages
- Graceful fallback for API failures

## Documentation Style
### Function Docstrings
```python
def example_function(param: str) -> bool:
    \"\"\"
    Brief description of the function.

    Detailed explanation if needed, including behavior,
    side effects, and usage examples.

    Args:
        param: Description of the parameter

    Returns:
        Description of return value

    Raises:
        SpecificException: When this exception occurs
    \"\"\"
```

### Class Docstrings
```python
class ExampleClass:
    \"\"\"
    Brief description of the class purpose.
    
    Detailed explanation of the class responsibilities,
    usage patterns, and important implementation details.
    \"\"\"
```

## File Organization Patterns
- One main class per file
- Related utility functions in the same module
- Clear separation between interfaces, implementations, and utilities
- Factory patterns for object creation (e.g., parser_factory.py)