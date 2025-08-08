# Code Style and Conventions

## Python Style
- **Formatter**: Ruff (replaces Black)
- **Linter**: Ruff (replaces Flake8, isort, etc.)
- **Line Length**: 88 characters (Ruff default)
- **Python Version**: 3.12+ features allowed

## Import Organization
```python
# Standard library imports
import asyncio
import os

# Third-party imports  
from mcp import types
from qdrant_client import QdrantClient

# Local imports
from .core.context import AppContext
from .services.embedding_service import EmbeddingService
```

## Type Hints
- **Required**: All function parameters and return types
- **Style**: Use modern Python 3.12+ syntax (e.g., `list[str]` not `List[str]`)
- **Optional**: Use `Optional[Type]` or `Type | None`

## Docstrings
```python
def function_name(param: str) -> dict[str, Any]:
    """
    Brief description of the function.
    
    Args:
        param: Description of parameter
        
    Returns:
        Description of return value
        
    Raises:
        SpecificError: When this error occurs
    """
```

## Error Handling
- Use specific exception types
- Include context in error messages
- Log errors appropriately
- Use `tenacity` for retries where appropriate

## Async/Await
- Use `async def` for all MCP tool functions
- Prefer `asyncio.gather()` for parallel operations
- Use context managers for resource management

## File Organization
- One class per file for major components
- Group related functionality in modules
- Use `__init__.py` for clean imports
- Keep tools separate from business logic

## Naming Conventions
- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`  
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: Leading underscore `_private`
- **MCP Tools**: Descriptive names like `crawl_single_page`