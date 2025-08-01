# Task Completion Checklist

## When Completing Development Tasks

### Code Quality Checks
- **Syntax Check**: Ensure Python 3.12+ compatibility
- **Async/Await**: Verify proper async function usage
- **Type Hints**: Add appropriate Optional/Any type hints where needed
- **Import Organization**: Follow standard library → third-party → local pattern

### Testing and Validation
- **MCP Server Test**: Restart server and verify it starts without errors
- **Tool Registration**: Ensure new tools appear in MCP client
- **Error Handling**: Test error scenarios return proper JSON responses
- **Environment Variables**: Verify new config options work with .env

### Documentation Updates
- **Docstrings**: Update function docstrings with Google-style format
- **README Updates**: Add new tool descriptions to README.md if needed
- **Environment Example**: Update .env.example if new variables added

### Database and External Services
- **Supabase Schema**: Run any schema changes via crawled_pages.sql
- **Neo4j Compatibility**: Test knowledge graph features if modified
- **API Integration**: Verify OpenAI API calls work correctly

### Deployment Verification
- **Docker Build**: Test `docker build -t mcp/crawl4ai-rag .`
- **Container Run**: Verify `docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag`
- **uv Installation**: Test `uv pip install -e .` and `crawl4ai-setup`

### Integration Testing
- **MCP Registration**: Test MCP client connection
- **Tool Execution**: Verify all tools work via MCP interface
- **SSE Transport**: Test Server-Sent Events transport mode
- **Error Recovery**: Ensure graceful error handling

## No Specific Linting/Formatting Commands
This project doesn't have configured linting tools like black, flake8, or mypy. Follow the established code style patterns in the existing codebase.

## Pre-commit Checklist
1. Server starts without errors: `uv run src/crawl4ai_mcp.py`
2. Docker builds successfully: `docker build -t mcp/crawl4ai-rag .`
3. Environment template updated if needed
4. Documentation reflects any new features
5. Async patterns maintained throughout
6. JSON response formatting consistent