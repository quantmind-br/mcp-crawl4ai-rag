# Code Style and Conventions

## General Coding Standards

### Function and Class Documentation
- **Google-style docstrings** with comprehensive parameter and return documentation
- **Type hints** required for all functions and methods
- **Return type annotations** mandatory
- **Parameter descriptions** with types and purposes

### Example Function Documentation:
```python
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in the vector database.

    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in the vector database for later retrieval and querying.

    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl

    Returns:
        Summary of the crawling operation and storage in the vector database
    """
```

## Naming Conventions

### Functions and Variables
- **snake_case** for functions, variables, and module names
- **Descriptive names** that clearly indicate purpose
- **Async functions** prefixed with async keyword

### Classes
- **PascalCase** for class names
- **Singleton pattern** for shared resources (ContextSingleton, RerankingModelSingleton)

### Constants
- **UPPER_SNAKE_CASE** for module-level constants
- **Environment variables** in ALL_CAPS

## Code Organization

### Module Structure
```
src/
├── core/           # Core application functionality
├── tools/          # MCP tools by category
├── services/       # Business logic services  
├── clients/        # External API clients
├── utils/          # Utility functions
└── k_graph/        # Knowledge graph parsing
```

### Import Organization
- **Standard library** imports first
- **Third-party** imports second
- **Local** imports last
- **Relative imports** for local modules

## Error Handling
- **Comprehensive try-catch blocks** for external API calls
- **Graceful degradation** for optional features
- **Detailed error logging** with context
- **JSON error responses** for MCP tools

## Async Programming
- **async/await** for all I/O operations
- **Context managers** for resource cleanup
- **AsyncWebCrawler** for web operations
- **Proper lifespan management** for MCP server

## Type Safety
- **Full type annotations** including Union types
- **Optional parameters** explicitly typed
- **Return types** always specified
- **Context types** properly annotated

## Performance Patterns
- **Batch processing** for database operations
- **Concurrent execution** using ThreadPoolExecutor and ProcessPoolExecutor
- **Connection pooling** for database clients
- **Caching strategies** with Redis integration

## Unicode and Cross-Platform
- **ASCII-only** console output to avoid Windows encoding errors
- **UTF-8** encoding for file operations
- **Cross-platform** path handling with pathlib

## Windows Compatibility
- **Batch scripts** (.bat) for Windows automation
- **Shell scripts** (.sh) for Linux/Mac compatibility
- **Platform-specific** logging configuration