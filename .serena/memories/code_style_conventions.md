# Code Style Conventions

## General Python Style
- **PEP 8 compliance** enforced by Ruff linter/formatter
- **Type hints** used throughout codebase for better IDE support and documentation
- **Docstrings** in Google/NumPy style with clear parameter and return descriptions
- **Async/await patterns** consistently used for I/O operations

## Naming Conventions
- **snake_case** for functions, variables, and module names
- **PascalCase** for classes
- **UPPER_CASE** for constants and environment variables
- **Descriptive names** that clearly indicate purpose

## Code Organization
- **Separation of concerns** with clear module boundaries
- **Factory patterns** for dynamic object creation
- **Singleton patterns** for shared resources (Context management)
- **Modular architecture** with feature-based organization

## Function and Class Structure
```python
async def function_name(param: Type) -> ReturnType:
    """
    Clear description of what the function does.

    Args:
        param: Description of the parameter

    Returns:
        Description of return value
    """
    # Implementation with clear logic flow
```

## Error Handling
- **Structured exception handling** with specific error types
- **Logging** for debugging and monitoring
- **Graceful degradation** for non-critical failures
- **Context managers** for resource cleanup

## Import Organization
1. Standard library imports
2. Third-party imports
3. Local application imports
4. Relative imports (if needed)

## Windows-Specific Considerations
- **Unicode handling** - ASCII-only characters in console output to avoid UnicodeEncodeError
- **Path handling** - Use pathlib for cross-platform compatibility
- **Logging filters** - Special handling for Windows ConnectionResetError messages

## MCP Tool Patterns
```python
@mcp.tool()
async def tool_name(ctx: Context, param: str) -> str:
    """Tool description for MCP clients."""
    try:
        # Tool implementation
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)
```

## Configuration Management
- **Environment variables** for all configuration
- **Default values** with clear documentation
- **Type conversion** and validation for config values
- **.env file** support with .env.example template