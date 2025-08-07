# Code Style and Conventions

## General Python Conventions
- **Python 3.12+** syntax and features
- **Type hints** for all function parameters and return values
- **Async/await** patterns for I/O operations
- **Docstrings** using Google/NumPy style with Args/Returns sections
- **Error handling** with try/except blocks and context logging

## Code Organization Patterns
```python
# Import order: standard library, third-party, local imports
import os
import json
from typing import Dict, List, Optional

from crawl4ai import AsyncWebCrawler
from qdrant_client import QdrantClient

from .clients.qdrant_client import QdrantClientWrapper
```

## Async Function Patterns
```python
@mcp.tool()
async def function_name(ctx: Context, param: str) -> str:
    """
    Brief description of the function.

    Args:
        ctx: The MCP server provided context
        param: Description of parameter

    Returns:
        Description of return value
    """
    try:
        # Implementation
        result = await some_async_operation()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)
```

## Error Handling Conventions
- Use **tenacity** for retrying failed operations (especially API calls)
- Implement **graceful fallbacks** (GPU → CPU, contextual → basic embeddings)
- **Log errors** with context using the configured logger
- Return **JSON responses** from MCP tools with success/error structure

## File and Directory Structure
- `src/` - Main source code
  - `clients/` - API and database client wrappers
  - `services/` - Business logic services
  - `features/` - Feature-specific implementations
  - `utils/` - Utility functions and helpers
- `tests/` - Test files with `test_` prefix
- `knowledge_graphs/` - Neo4j integration scripts
- `scripts/` - Utility scripts for database management

## Variable Naming
- **snake_case** for variables, functions, and modules
- **PascalCase** for classes
- **UPPER_CASE** for constants
- **Descriptive names** preferred over abbreviations
- **Context-specific prefixes**: `mock_` for test mocks, `temp_` for temporary variables

## Configuration Patterns
- **Environment variables** for configuration with `.env` support
- **Flexible API configuration** supporting multiple providers
- **Graceful defaults** with fallback options
- **Validation functions** for configuration correctness

## Testing Conventions
- **Class-based tests** using `TestClassName` pattern
- **Descriptive test names** explaining what is being tested
- **Mocking external services** (APIs, databases) in unit tests
- **Integration tests** requiring actual Docker services
- **Async test support** where needed