# Code Style and Conventions

## Formatting Standards
- **Line Length**: 120 characters (soft limit)
- **Formatter**: Prefer `ruff format` (or black defaults)
- **Trailing Commas**: Required in multi-line literals
- **Blank Lines**: Between logical groups for readability

## Import Organization
- **Grouping**: stdlib, third-party, local (with blank lines between)
- **Style**: One import per line, avoid deep relatives beyond package root
- **`__all__`**: Keep minimal, only export public API
- **Relative Imports**: Avoid going beyond package root

## Type Annotations
- **Public Functions**: Must be annotated with types
- **Type Hints**: Use `Optional`, `Dict`, `List`, `Tuple` from typing
- **Data Structures**: Prefer `@dataclass` for simple structs
- **MCP Tools**: Return JSON strings (not objects)

## Naming Conventions
- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants/Env Vars**: `UPPER_CASE`
- **Private Helpers**: Prefix with underscore `_private_function`

## Error Handling
- **Security**: Never print or log secrets/API keys
- **Exception Scope**: Catch broad exceptions only at tool boundaries
- **Return Format**: Structured JSON `{"success": false, "error": "..."}`
- **Logging**: Use `logging` module, not print statements (except CLI tools)

## Async/Concurrency
- **Preferred Pattern**: `async/await` with `@mcp.tool()` decorators
- **Tool Handlers**: Avoid blocking calls in MCP tool handlers
- **Parallel Processing**: Use `concurrent.futures` for CPU-bound tasks

## Configuration Management
- **Environment**: Load `.env` via dotenv on startup
- **Feature Flags**: Support boolean flags (USE_CONTEXTUAL_EMBEDDINGS, etc.)
- **API Compatibility**: Support OpenAI-compatible providers via base URL
- **Validation**: Validate and sanitize URLs/paths for security

## RAG & AI Patterns
- **Default Embeddings**: OpenAI text-embedding-3-small
- **Vector Database**: Use QdrantClientWrapper for all operations
- **Search Strategy**: Prefer hybrid search when enabled
- **Reranking**: Use CrossEncoder when USE_RERANKING=true

## Testing Guidelines
- **Network Isolation**: Mock QdrantClient and OpenAI API calls
- **Deterministic**: Tests should be repeatable and predictable
- **Location**: Unit tests in `tests/` directory
- **Async Testing**: Use pytest-asyncio for async test functions

## Security & Access Control
- **API Keys**: Never log or expose in error messages
- **File Access**: Guard behind appropriate feature flags
- **Network Access**: Validate URLs before making requests
- **Input Sanitization**: Always validate user inputs

## CLI & Entry Points
- **Script Execution**: `uv run path/to/script.py` for single scripts
- **Module Entry**: `uv run -m src` for main server
- **Health Endpoint**: Available at `/health` when using SSE transport
- **Error Handling**: Return proper exit codes from scripts