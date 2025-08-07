# Task Completion Checklist

## Before Making Changes
- [ ] Read relevant existing code to understand patterns and conventions
- [ ] Check if similar functionality already exists to avoid duplication
- [ ] Understand the MCP tool structure and async patterns used
- [ ] Verify environment configuration requirements

## During Development
- [ ] Follow existing code style and naming conventions
- [ ] Use type hints for all function parameters and return types
- [ ] Implement proper error handling with try/except blocks
- [ ] Add comprehensive docstrings using Google/NumPy style
- [ ] Use async/await patterns for I/O operations
- [ ] Return JSON responses from MCP tools with consistent structure
- [ ] Test with different API providers if applicable
- [ ] Consider GPU/CPU fallback scenarios where relevant

## Code Quality Checks
- [ ] **No linting specified** - Follow existing code patterns for consistency
- [ ] **No formatting tool specified** - Maintain existing indentation and style
- [ ] **No type checking tool specified** - Use type hints but manual verification
- [ ] Ensure proper import organization (standard, third-party, local)
- [ ] Check for proper exception handling and logging

## Testing Requirements
- [ ] Run all existing tests: `uv run pytest`
- [ ] Run specific relevant tests for the area you modified
- [ ] Test with Docker services running: `uv run pytest tests/integration_test.py`
- [ ] If modifying MCP tools, test: `uv run pytest tests/test_mcp_basic.py`
- [ ] If modifying Qdrant integration, test: `uv run pytest tests/test_qdrant_wrapper.py`
- [ ] Run performance benchmarks if performance-critical changes: `uv run pytest tests/performance_benchmark.py`
- [ ] Test with both SSE and stdio transports if modifying MCP server

## Environment Testing
- [ ] Test with different API providers (OpenAI, DeepInfra, etc.)
- [ ] Test with various RAG strategy flag combinations
- [ ] Test with and without Docker services running
- [ ] Test GPU acceleration if modifying reranking features
- [ ] Verify Windows batch script compatibility if applicable

## Documentation Updates
- [ ] Update relevant docstrings and inline comments
- [ ] Update CLAUDE.md if architecture or commands change
- [ ] Update README.md if user-facing features change
- [ ] Update .env.example if new environment variables are added

## Final Verification
- [ ] Ensure the MCP server starts correctly: `start.bat` or `uv run -m src`
- [ ] Test basic MCP functionality through a client
- [ ] Verify Docker services are properly utilized
- [ ] Check that no secrets or API keys are hardcoded
- [ ] Confirm backward compatibility with existing configurations
- [ ] Test error scenarios and ensure graceful error handling

## Deployment Considerations
- [ ] Verify Windows compatibility if using batch scripts
- [ ] Test both development and production-like environments
- [ ] Ensure proper resource cleanup (GPU memory, database connections)
- [ ] Test with resource constraints (limited memory, CPU)
- [ ] Verify proper logging and monitoring capabilities