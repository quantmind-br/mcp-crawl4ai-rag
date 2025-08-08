# Code Style and Conventions

## Python Style Guidelines
- **Python Version**: 3.12+ required
- **Type Hints**: Used throughout the codebase
- **Docstrings**: Present for classes and functions
- **Import Style**: Relative imports within package, absolute for external
- **Async/Await**: Extensive use of asyncio patterns

## Code Formatting
- **Linter**: Ruff (configured in dev dependencies)
- **Line Length**: Not explicitly configured (Ruff default: 88 chars)
- **Import Sorting**: Handled by Ruff
- **Code Formatting**: Ruff handles both linting and formatting

## File Organization
- **Module Structure**: Clear separation of concerns
  - `clients/`: External service integrations
  - `core/`: Application foundation
  - `services/`: Business logic
  - `tools/`: MCP tool implementations
  - `utils/`: Helper functions
- **Naming Conventions**:
  - Files: `snake_case.py`
  - Classes: `PascalCase`
  - Functions: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private members: `_leading_underscore`

## Async Patterns
- **Singleton Pattern**: Used for shared resources (ContextSingleton, RerankingModelSingleton)
- **Context Managers**: For resource management and cleanup
- **Lifespan Management**: Proper startup/shutdown handling
- **Error Handling**: Comprehensive exception handling with logging

## Documentation Style
- **Docstrings**: Triple quotes with clear descriptions
- **Comments**: Inline comments for complex logic
- **Type Annotations**: Full type hinting including return types
- **Module Docstrings**: File-level documentation explaining purpose

## Project-Specific Patterns
- **MCP Tools**: Standardized async function signatures
- **Service Layer**: Clear separation between tools and business logic
- **Client Abstractions**: Consistent interfaces for external services
- **Configuration Management**: Environment-based configuration
- **Logging**: Structured logging throughout application

## Error Handling
- **Exception Classes**: Custom exceptions where appropriate
- **Graceful Degradation**: Fallback mechanisms for API failures
- **Logging**: Comprehensive error logging and debugging information
- **Resource Cleanup**: Proper cleanup in finally blocks and context managers