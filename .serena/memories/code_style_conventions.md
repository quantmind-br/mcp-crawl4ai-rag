# Code Style & Conventions

## Language Standards
- **Type Hints**: Required for all function parameters and return types
- **Documentation**: Docstrings for all public functions and classes
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Error Handling**: tenacity for retries, explicit error propagation

## Async/Concurrent Patterns
- **Asynchronous First**: All MCP tools use `@mcp.tool()` decorator with async functions
- **Connection Pooling**: Qdrant and Redis clients use connection pooling
- **Resource Management**: Automatic cleanup for GPU memory after operations

## Import Organization
- **External**: Standard library → Third-party → Local imports
- **Lazy Initialization**: Single instance patterns for heavy resources
- **Singleton Patterns**: Reranking models, knowledge graphs, clients

## Configuration Management
- **Modern API**: CHAT_MODEL/EMBEDDINGS_MODEL + API_KEY pattern
- **Fallback**: Automatic fallback configuration validation
- **Environment**: dotenv loading with comprehensive validation

## Error Handling
- **Tenacity**: Retry policies for API calls and external services
- **Graceful Degradation**: GPU → CPU fallback, temporary failures handling
- **Logging**: Structured logging with appropriate levels (INFO, WARNING, ERROR)