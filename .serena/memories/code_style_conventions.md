# Code Style and Conventions

## Code Style
- **Linter**: ruff >=0.12.7 for code quality and formatting
- **Python Version**: 3.12+ with modern Python features
- **Import Style**: Absolute imports with try/except for relative imports
- **Docstrings**: Module-level docstrings with detailed descriptions

## Naming Conventions
- **Classes**: PascalCase (e.g., `QdrantClientWrapper`, `GitHubRepoManager`)
- **Functions**: snake_case (e.g., `get_optimal_device`, `create_embeddings_batch`)
- **Variables**: snake_case (e.g., `embedding_dimensions`, `qdrant_client`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `TORCH_AVAILABLE`, `COLLECTIONS`)
- **Private/Internal**: Leading underscore (e.g., `_qdrant_client_instance`, `_create_embeddings_api_call`)

## Code Organization
- **Imports**: External imports first, then try/except blocks for relative imports
- **Error Handling**: Use tenacity for retrying operations, comprehensive exception handling
- **Async Patterns**: All MCP tools are async with proper context management
- **Type Hints**: Extensive use of typing with List, Dict, Any, Optional

## File Structure Patterns
- **Main Module**: `src/crawl4ai_mcp.py` - Primary MCP server with all tools
- **Utilities**: `src/utils/` - Helper functions and client wrappers
- **Configuration**: Environment-based with `.env` files and validation
- **Tests**: Comprehensive test suite with mocking for external services

## Documentation Style
- **Module Docstrings**: Detailed purpose and functionality descriptions
- **Function Docstrings**: Parameters, return values, and usage examples
- **Inline Comments**: Explanatory comments for complex logic
- **README**: Comprehensive setup, configuration, and usage documentation

## Error Handling Patterns
- **Graceful Fallbacks**: GPU → CPU, primary API → fallback API
- **Validation**: Input validation with clear error messages
- **Logging**: Structured logging with appropriate levels
- **Windows Compatibility**: Special handling for ConnectionResetError issues

## Testing Conventions
- **Test Organization**: Test files mirror source structure (`test_*.py`)
- **Mocking**: Extensive use of unittest.mock for external dependencies
- **Environment Setup**: conftest.py with automatic test environment configuration
- **Coverage**: Multiple test categories (unit, integration, performance)