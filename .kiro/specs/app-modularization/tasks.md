# Implementation Plan

- [x] 1. Validate current test suite and create foundation structure




  - Run complete test suite to establish baseline functionality
  - Create new directory structure with proper __init__.py files
  - _Requirements: 4.1, 7.1, 8.1_

- [x] 2. Extract and refactor Qdrant client wrapper





  - Move QdrantClientWrapper class from src/qdrant_wrapper.py to src/clients/qdrant_client.py
  - Update all imports throughout the codebase to use new location
  - Ensure get_qdrant_client function is properly exposed
  - _Requirements: 3.1, 3.3, 5.2_

- [x] 3. Create LLM API client module




  - Extract get_embeddings_client, get_chat_client, and fallback client functions from src/utils.py
  - Create src/clients/llm_api_client.py with all API client creation logic
  - Include configuration validation functions for API clients
  - Update imports and ensure backward compatibility during transition
  - _Requirements: 1.1, 3.1, 5.1_


- [x] 4. Implement EmbeddingService class









  - Create src/services/embedding_service.py with EmbeddingService class
  - Move create_embeddings_batch, create_sparse_embeddings_batch, and related functions from src/utils.py
  - Integrate EmbeddingCache as a dependency of the service
  - Include SparseVectorEncoder class and sparse embedding logic
  - Write unit tests for the embedding service functionality
  - _Requirements: 1.1, 2.3, 5.1_

- [x] 5. Implement RagService class





  - Create src/services/rag_service.py with RagService class
  - Move search_documents, search_code_examples, and update_source_info functions from src/utils.py
  - Extract re-ranking logic from src/crawl4ai_mcp.py and integrate into RagService
  - Implement hybrid search functionality that combines dense and sparse search results
  - Add methods for result fusion and CrossEncoder-based re-ranking
  - _Requirements: 1.1, 2.1, 6.2_

- [x] 6. Move GitHub processor to features layer






  - Move src/utils/github_processor.py to src/features/github_processor.py
  - Update src/utils/__init__.py to import from new location
  - Remove the confusing src/utils/ directory structure
  - Update any scripts or tests that reference the old location
  - _Requirements: 3.1, 3.2, 5.2_


- [x] 7. Create core application structure


  - Create src/core/context.py with enhanced Crawl4AIContext dataclass
  - Include references to new service instances in the context
  - Create src/core/app.py with FastMCP instance creation and lifespan management
  - Move application initialization logic from src/crawl4ai_mcp.py
  - _Requirements: 6.1, 6.2, 3.1_

- [x] 8. Extract web crawling tools

  - Create src/tools/web_tools.py module
  - Move crawl_single_page and smart_crawl_url tool implementations from src/crawl4ai_mcp.py
  - Update tools to use new service dependencies (RagService, EmbeddingService)
  - Ensure tools maintain exact same interface and behavior
  - _Requirements: 2.1, 4.2, 6.2_

- [x] 9. Extract GitHub-specific tools

  - Create src/tools/github_tools.py module
  - Move smart_crawl_github tool implementation from src/crawl4ai_mcp.py
  - Update tool to use GitHubProcessor from features layer and new services
  - Maintain compatibility with existing GitHub crawling functionality
  - _Requirements: 2.1, 4.2, 6.2_

- [x] 10. Extract knowledge graph tools

  - Create src/tools/kg_tools.py module
  - Move knowledge graph and AI hallucination detection tools from src/crawl4ai_mcp.py
  - Update tools to use new service architecture
  - Ensure integration with existing knowledge_graphs/ modules remains intact
  - _Requirements: 2.1, 4.2, 6.2_

- [x] 11. Implement tool registration system

  - Update src/core/app.py to import and register all tools from tool modules
  - Create register_tools function that adds all MCP tools to FastMCP instance
  - Ensure proper dependency injection of services into tools
  - _Requirements: 6.1, 6.3, 2.2_

- [x] 12. Update application entry points

  - Update src/__main__.py to use new src/core/app.py structure
  - Update run_server.py to import from new application structure
  - Ensure application starts correctly with all tools registered
  - _Requirements: 4.3, 7.3_

- [x] 13. Update all import statements systematically

  - Search and replace all imports of old utils functions throughout codebase
  - Update test files to import from new module locations
  - Update any scripts in scripts/ directory that depend on old structure
  - Ensure no broken import references remain
  - _Requirements: 3.3, 8.2_

- [x] 14. Remove legacy monolithic files

  - Delete src/crawl4ai_mcp.py after confirming all functionality is migrated
  - Delete src/utils.py after confirming all functions are moved to appropriate services/clients
  - Remove empty src/utils/ directory
  - Clean up any remaining unused import statements
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 15. Run comprehensive validation and testing

  - Execute complete test suite to ensure 100% pass rate
  - Run static analysis tools (ruff, mypy) to ensure code quality
  - Test application startup and tool functionality manually
  - Verify all existing scripts and entry points work correctly
  - _Requirements: 4.1, 7.1, 7.4, 8.3_