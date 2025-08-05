# Task Completion Checklist

## Code Quality and Testing
- [ ] **Run linting**: `uv run ruff check` and `uv run ruff format`
- [ ] **Run tests**: `uv run pytest` for all test suites
- [ ] **Test specific modules**: Run relevant test files for modified components
- [ ] **Integration tests**: `uv run pytest tests/integration_test.py` for full system tests
- [ ] **Performance validation**: `uv run pytest tests/performance_benchmark.py` if performance-related

## Configuration and Environment
- [ ] **Environment variables**: Verify `.env` configuration is correct
- [ ] **Docker services**: Ensure Qdrant, Neo4j, Redis are running (`docker-compose ps`)
- [ ] **API keys**: Validate API configuration and fallback settings
- [ ] **Dependencies**: Run `uv sync` if new dependencies were added

## Functionality Testing
- [ ] **MCP server startup**: Test server starts without errors (`start.bat` or `uv run -m src`)
- [ ] **MCP tools**: Verify all relevant MCP tools work correctly
- [ ] **GPU acceleration**: Test GPU functionality if reranking/GPU features modified
- [ ] **Windows compatibility**: Verify Windows-specific fixes work correctly

## Documentation and Communication
- [ ] **Code comments**: Add/update inline documentation for complex logic
- [ ] **Type hints**: Ensure all new functions have proper type annotations
- [ ] **Error handling**: Implement graceful error handling with appropriate logging
- [ ] **Changelog**: Document significant changes in commit messages

## Specific Component Checks

### Vector Database Changes
- [ ] **Qdrant collections**: Verify collection schemas are correct
- [ ] **Embedding dimensions**: Check dimension consistency
- [ ] **Hybrid search**: Test sparse/dense vector functionality if applicable

### API Client Changes  
- [ ] **Fallback mechanisms**: Test primary and fallback API configurations
- [ ] **Rate limiting**: Verify retry logic with tenacity
- [ ] **Multi-provider**: Test different API providers if modified

### Knowledge Graph Changes
- [ ] **Neo4j connection**: Test connection and query functionality
- [ ] **Repository parsing**: Verify GitHub repository processing
- [ ] **Hallucination detection**: Test AI script validation if applicable

### Device Management Changes
- [ ] **GPU detection**: Test automatic GPU/CPU fallback
- [ ] **Memory cleanup**: Verify GPU memory is properly cleaned up
- [ ] **Device configuration**: Test different GPU precision settings

## Pre-Commit Validation
- [ ] **No syntax errors**: Python files parse correctly
- [ ] **Import resolution**: All imports resolve correctly
- [ ] **Environment compatibility**: Works with minimal .env configuration
- [ ] **Docker compatibility**: Services start cleanly with docker-compose

## Windows-Specific Validation
- [ ] **Event loop fix**: Windows ConnectionResetError handling works
- [ ] **Batch scripts**: setup.bat and start.bat execute successfully
- [ ] **PyTorch CUDA**: GPU acceleration works on Windows if applicable