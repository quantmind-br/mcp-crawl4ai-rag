# Code Style and Conventions

## Code Formatting & Linting
- **Ruff** - Primary linting and formatting tool (>=0.12.7)
  - Auto-fixing: `uv run ruff check --fix .`
  - Formatting: `uv run ruff format .`
  - Configuration in pyproject.toml

## Python Style Guidelines
- **Function signatures**: Type hints for all parameters and return values
- **Async/await**: Consistent async patterns throughout codebase
- **Docstrings**: Google-style docstrings for all public functions/classes
- **Imports**: Organized with `# ruff: noqa: E402` when needed for test files

## Naming Conventions
- **Functions/variables**: snake_case (e.g., `perform_rag_query`, `match_count`)
- **Classes**: PascalCase (e.g., `RagService`, `ContextSingleton`) 
- **Constants**: UPPER_SNAKE_CASE (e.g., `USE_HYBRID_SEARCH`, `QDRANT_HOST`)
- **Private members**: Leading underscore (e.g., `_internal_method`)

## Type Annotations
- **Required**: All function parameters and return types
- **Optional types**: `Optional[str] = None` pattern
- **Context typing**: `ctx: Context` for MCP tools
- **JSON returns**: Functions returning JSON use `-> str` with `json.dumps()`

## Error Handling
- **Try-catch blocks**: Comprehensive error handling in MCP tools
- **JSON error responses**: Consistent error format with `{"success": False, "error": str(e)}`
- **Logging**: Use `logger` from `logging` module for debug info

## Documentation Standards
- **Docstring format**:
  ```python
  async def function_name(param: type) -> return_type:
      \"\"\"
      Brief description.

      Detailed explanation of functionality and purpose.

      Args:
          param: Description of parameter

      Returns:
          Description of return value
      \"\"\"
  ```

## File Organization
- **MCP Tools**: One tool per function in `src/tools/`
- **Services**: Business logic in `src/services/`
- **Clients**: External API wrappers in `src/clients/`
- **Tests**: Mirror source structure in `tests/`

## Import Organization
- **Standard library** first
- **Third-party packages** second  
- **Local imports** last
- **Relative imports**: Use `from ..services import` pattern

## Configuration Patterns
- **Environment variables**: All config via .env files
- **Singleton patterns**: Context, reranking models, knowledge graph
- **Default values**: Provide sensible defaults for optional parameters

## Unicode Guidelines (CRITICAL for Windows)
- **NEVER use Unicode characters**: ✅ ❌ 🚀 🔧 etc.
- **ASCII alternatives only**: "SUCCESS", "FAILED", "ERROR"
- **Console compatibility**: All output must work in Windows cmd (cp1252)