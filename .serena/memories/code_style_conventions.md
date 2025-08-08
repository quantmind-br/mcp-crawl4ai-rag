# Code Style and Conventions - Crawl4AI MCP RAG

## Python Style Guidelines
- **Type Hints**: Extensive use of type annotations for all function parameters and returns
  - `Optional[str]` for nullable parameters
  - `async def` for all async functions
  - Return type annotations: `-> str`, `-> None`, etc.

## Documentation Standards
- **Docstrings**: Comprehensive Google-style docstrings for all public functions
  - Multi-line docstrings with detailed parameter descriptions
  - Args, Returns, and Raises sections where applicable
  - Example from codebase:
    ```python
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.

    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain and/or specific file_id.

    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results
        match_count: Maximum number of results to return (default: 5)
        file_id: Optional file_id to filter results to specific files

    Returns:
        JSON string with the search results
    """
    ```

## Naming Conventions
- **Variables**: snake_case (`qdrant_client`, `filter_metadata`)
- **Functions**: snake_case (`perform_rag_query`, `run_server`)  
- **Classes**: PascalCase with descriptive suffixes (`ContextSingleton`, `RagService`)
- **Constants**: UPPER_SNAKE_CASE (`USE_CONTEXTUAL_EMBEDDINGS`)
- **Private methods**: Leading underscore (`_internal_method`)

## Code Organization
- **Modular Structure**: Clear separation by functionality
  - `src/core/`: Application core (app.py, context.py)
  - `src/tools/`: MCP tool implementations
  - `src/services/`: Business logic services
  - `src/utils/`: Utility functions
  - `src/clients/`: External service clients

## Error Handling
- **Try-catch blocks**: Comprehensive exception handling with descriptive error messages
- **JSON error responses**: Consistent error response format
  ```python
  return json.dumps({"success": False, "query": query, "error": str(e)}, indent=2)
  ```

## Async Patterns
- **Async/await**: Consistent use of async functions for I/O operations
- **Context management**: Proper resource cleanup in async contexts
- **Lifespan management**: Application-level resource management

## Import Style
- **Relative imports**: Use relative imports within the package (`from ..services.rag_service import RagService`)
- **Standard library first**: Follow PEP 8 import ordering
- **Type imports**: Separate type imports when needed (`from typing import Optional`)

## Configuration Management
- **Environment variables**: Extensive use of environment-based configuration
- **Default values**: Sensible defaults with environment override capability
- **Validation**: Configuration validation at startup