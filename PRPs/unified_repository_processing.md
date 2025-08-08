# Unified Repository Processing Feature - PRP

---

## Goal

**Feature Goal**: Create a single, unified tool `index_github_repository` that eliminates duplicate repository processing by combining the functionality of `smart_crawl_github` (RAG indexing) and `parse_github_repository` (Knowledge Graph indexing) into one efficient pipeline with cross-system data linkage.

**Deliverable**: A unified MCP tool that processes GitHub repositories once to populate both Qdrant (vector search) and Neo4j (knowledge graph) with bidirectional data linking via `file_id` metadata, reducing processing time by 50-70% while enabling RAG-KG synergy queries.

**Success Definition**: 
- Repository cloned and processed only once for both destinations
- Both Qdrant and Neo4j populated with linked data using consistent `file_id`
- Existing tool compatibility maintained during transition
- Processing performance improved by 50%+ for combined operations

## User Persona

**Target User**: AI coding assistants, developers using MCP Crawl4AI RAG for comprehensive code analysis and documentation search

**Use Case**: Index GitHub repositories for both semantic search (documentation, code examples) and structural analysis (classes, methods, dependencies) in a single operation

**User Journey**: 
1. Call `index_github_repository(repo_url, destination="both")`
2. Repository is cloned once and all files processed appropriately
3. Text content goes to Qdrant for RAG, code structure goes to Neo4j for KG
4. Cross-system queries enabled via `file_id` linking

**Pain Points Addressed**: 
- Eliminates duplicate repository cloning (saves 50% processing time)
- Removes need to call two separate tools for complete indexing
- Enables cross-system queries between documentation and code structure
- Reduces complexity for AI agents using the system

## Why

- **Efficiency**: Single repository cloning eliminates 50% of I/O and network operations
- **Data Consistency**: Unified file discovery ensures both systems see identical repository state
- **RAG-KG Synergy**: Bidirectional linking enables queries like "find documentation for method X" and "what code implements concept Y"
- **Maintenance**: Single codebase for repository processing reduces technical debt
- **Scalability**: Unified pipeline supports future enhancements more easily

## What

**User-visible behavior**: Replace two separate tools with one unified tool that supports destination selection

```python
# Instead of calling both:
await smart_crawl_github(repo_url, file_types=[".md"])
await parse_github_repository(repo_url)  

# Call unified tool:
await index_github_repository(repo_url, destination="both", file_types=[".md", ".py", ".ts"])
```

**Technical requirements**:
- Maintain backward compatibility with existing tool parameters
- Support destination selection: "qdrant", "neo4j", or "both"
- Create bidirectional data linking via `file_id` metadata
- Preserve existing service integration patterns
- Support all current file types and processors

### Success Criteria

- [ ] Single repository cloning for all destinations
- [ ] Qdrant documents linked to Neo4j nodes via `file_id`
- [ ] Processing time reduced by 50%+ for combined operations  
- [ ] All existing file types supported (12+ languages)
- [ ] Backward compatibility maintained during transition
- [ ] Cross-system RAG-KG queries functional

## All Needed Context

### Context Completeness Check

_Validated: This PRP provides complete implementation context including existing patterns, shared components, integration points, external documentation, and specific test patterns from the codebase._

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- docfile: PRPs/ai_docs/treesitter_multi_language_integration.md
  why: Complete Tree-sitter integration patterns for multi-language parsing
  section: Core Architecture Patterns, Unified Data Structure Pattern
  critical: Standardized output format for Neo4j compatibility

- docfile: PRPs/ai_docs/neo4j_rag_kg_integration_patterns.md  
  why: Bidirectional RAG-KG linking patterns and hybrid search implementation
  section: Unified Data Record Pattern, Cross-System Queries
  critical: file_id linking strategy between vector and graph databases

- docfile: PRPs/ai_docs/python_async_patterns_unified_pipelines.md
  why: Async processing patterns for unified data pipelines
  section: Producer-Consumer Pattern, Resource Management
  critical: Progress tracking and cleanup for long-running operations

- file: src/tools/github_tools.py
  why: smart_crawl_github implementation pattern for RAG processing
  pattern: Async MCP tool structure, resource cleanup, processor dispatch
  gotcha: Always cleanup temporary directories in finally block

- file: src/tools/kg_tools.py  
  why: parse_github_repository implementation pattern for Neo4j integration
  pattern: Environment variable checks, context access, error responses
  gotcha: Requires USE_KNOWLEDGE_GRAPH=true and repo_extractor in context

- file: src/features/github_processor.py
  why: Shared components (GitHubRepoManager, MultiFileDiscovery, processors)
  pattern: Factory pattern for file processors, cleanup with read-only handling
  gotcha: Windows path issues, size validation, excluded directories

- file: knowledge_graphs/parse_repo_into_neo4j.py
  why: Neo4j population patterns and Tree-sitter integration
  pattern: Semaphore-based deadlock prevention, batch processing
  gotcha: Language-specific queries, import filtering, duplicate prevention

- file: src/services/rag_service.py
  why: Vector database integration patterns
  pattern: Batch document processing, metadata structure, source management  
  gotcha: Collection schema must match, sparse vectors for hybrid search

- url: https://neo4j.com/blog/developer/unstructured-text-to-knowledge-graph/
  why: Neo4j GraphRAG best practices for document-code linking
  critical: Unified entity resolution patterns for cross-system queries

- url: https://tree-sitter.github.io/tree-sitter/using-parsers#pattern-matching-with-queries
  why: Tree-sitter query patterns for consistent multi-language parsing
  critical: Standardized capture names for unified output structure
```

### Current Codebase Tree

```bash
src/
├── tools/
│   ├── github_tools.py          # smart_crawl_github implementation
│   ├── kg_tools.py              # parse_github_repository implementation  
│   ├── rag_tools.py             # RAG search functionality
│   └── web_tools.py             # Supporting utilities
├── features/
│   └── github_processor.py      # Shared components (GitHubRepoManager, etc.)
├── services/
│   ├── rag_service.py           # Vector database operations
│   └── embedding_service.py     # Embedding generation
├── clients/
│   ├── qdrant_client.py         # Vector database client
│   └── llm_api_client.py        # LLM API client
├── core/
│   ├── app.py                   # MCP server setup and tool registration
│   └── context.py               # Application context management
└── utils/
    └── validation.py            # URL validation utilities

knowledge_graphs/
├── parse_repo_into_neo4j.py    # Neo4j population logic
├── tree_sitter_parser.py       # Multi-language parsing
├── parser_factory.py           # Language detection and parser management
└── grammars/                    # Tree-sitter language grammars
```

### Desired Codebase Tree with Files to be Added

```bash
src/
├── tools/
│   ├── github_tools.py          # MODIFY: Add index_github_repository tool
│   ├── kg_tools.py              # KEEP: Maintain existing tools during transition
│   └── rag_tools.py             # MODIFY: Add file_id filtering support
├── features/
│   └── github_processor.py      # MODIFY: Add unified processor dispatcher
├── services/
│   ├── rag_service.py           # MODIFY: Add file_id metadata support
│   └── unified_indexing_service.py  # CREATE: Orchestration service
└── utils/
    └── file_id_generator.py     # CREATE: Consistent file_id generation

# Test files to create:
tests/
├── test_unified_repository_processor.py    # CREATE: Comprehensive tests
├── test_file_id_linking.py                 # CREATE: Cross-system linking tests  
└── fixtures/
    └── test_repo_mixed_languages/           # CREATE: Multi-language test repo
```

### Known Gotchas of Codebase & Library Quirks

```python
# CRITICAL: MCP context access pattern
ctx.request_context.lifespan_context.qdrant_client  # May be None if disabled
ctx.request_context.lifespan_context.repo_extractor  # Requires USE_KNOWLEDGE_GRAPH=true

# CRITICAL: Resource cleanup on Windows (github_processor.py:680-695)
def handle_remove_readonly(func, path, exc):
    try:
        if os.path.exists(path):
            os.chmod(path, 0o777)  # Windows read-only file handling
            func(path)
    except PermissionError:
        pass  # Skip files in use

# CRITICAL: Neo4j semaphore protection (parse_repo_into_neo4j.py:1240-1245)  
async with _neo4j_init_semaphore:  # Prevents concurrent initialization deadlocks
    # Neo4j operations here

# CRITICAL: File size limits by type (github_processor.py:664-714)
FILE_SIZE_LIMITS = {
    ".py": 1_000_000,    # 1MB for Python files
    ".json": 100_000,    # 100KB for JSON configs  
    # Exceeding limits causes files to be skipped
}

# CRITICAL: Tree-sitter language availability check
ParserFactory.is_supported_file(file_path)  # Returns False if grammar not built
# Must run knowledge_graphs/build_grammars.py during setup

# CRITICAL: Qdrant collection schema compatibility
# Vector dimensions must match embedding model (auto-detected or configured)
# Collection recreation triggered if dimensions mismatch existing schema
```

## Implementation Blueprint

### Data Models and Structure

Create unified data structures ensuring cross-system compatibility:

```python
# src/models/unified_indexing_models.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class IndexingDestination(Enum):
    QDRANT = "qdrant"
    NEO4J = "neo4j"  
    BOTH = "both"

@dataclass
class FileProcessingResult:
    """Unified result structure for processed files."""
    file_id: str  # Format: "repo_name:relative_path"
    file_path: str
    relative_path: str
    language: str
    file_type: str
    processed_for_rag: bool = False
    processed_for_kg: bool = False
    rag_chunks: int = 0
    kg_entities: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

@dataclass 
class UnifiedIndexingRequest:
    """Request model for unified repository indexing."""
    repo_url: str
    destination: IndexingDestination
    file_types: List[str] = None
    max_files: int = 50
    chunk_size: int = 5000
    max_size_mb: int = 500
    
    def __post_init__(self):
        if self.file_types is None:
            self.file_types = [".md"]

@dataclass
class UnifiedIndexingResponse:
    """Response model with cross-system statistics."""
    success: bool
    repo_url: str
    repo_name: str
    destination: str
    files_processed: int
    qdrant_documents: int = 0
    neo4j_nodes: int = 0
    processing_time_seconds: float = 0.0
    file_results: List[FileProcessingResult] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.file_results is None:
            self.file_results = []
        if self.errors is None:
            self.errors = []
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE src/utils/file_id_generator.py
  - IMPLEMENT: generate_file_id(repo_url, relative_path) -> str function
  - FOLLOW pattern: Deterministic format "repo_name:relative_path"
  - NAMING: snake_case functions, descriptive parameter names
  - PLACEMENT: Utility module for cross-service usage
  - VALIDATION: URL normalization, path sanitization

Task 2: CREATE src/models/unified_indexing_models.py  
  - IMPLEMENT: Pydantic models for request/response with validation
  - FOLLOW pattern: src/models/ (if exists) or create new models directory
  - NAMING: CamelCase for classes, snake_case for fields
  - PLACEMENT: Centralized data models
  - DEPENDENCIES: Import from typing, enum, dataclasses

Task 3: MODIFY src/services/rag_service.py
  - IMPLEMENT: Add file_id metadata support to add_documents_to_vector_db
  - FOLLOW pattern: Existing metadata structure with file_id field addition
  - NAMING: Maintain existing function names, add file_id parameter
  - DEPENDENCIES: Import file_id_generator utility
  - PRESERVE: All existing functionality and backwards compatibility

Task 4: CREATE src/services/unified_indexing_service.py
  - IMPLEMENT: UnifiedIndexingService class with async methods
  - FOLLOW pattern: src/services/ structure, dependency injection via context
  - NAMING: UnifiedIndexingService class, async def process_repository method
  - DEPENDENCIES: Import models from Task 2, services, github_processor
  - PLACEMENT: Service orchestration layer

Task 5: MODIFY src/features/github_processor.py
  - IMPLEMENT: Unified file processor dispatcher based on destination
  - FOLLOW pattern: Existing processor factory pattern
  - NAMING: add process_for_destination method to existing classes
  - DEPENDENCIES: Import unified models, file_id_generator
  - PRESERVE: All existing processor functionality

Task 6: CREATE src/tools/github_tools.py:index_github_repository
  - IMPLEMENT: New MCP tool function with destination parameter
  - FOLLOW pattern: Existing smart_crawl_github structure and error handling
  - NAMING: async def index_github_repository MCP tool
  - DEPENDENCIES: Import UnifiedIndexingService, validation utilities
  - PLACEMENT: Add to existing github_tools.py file

Task 7: MODIFY src/core/app.py
  - IMPLEMENT: Register index_github_repository tool with MCP server
  - FIND pattern: Existing tool registrations around line 412
  - ADD: app.tool()(github_tools.index_github_repository) registration
  - PRESERVE: All existing tool registrations unchanged

Task 8: MODIFY src/tools/rag_tools.py
  - IMPLEMENT: Add file_id filtering to perform_rag_query function
  - FOLLOW pattern: Existing query parameter handling
  - NAMING: Add optional file_id parameter to existing function
  - DEPENDENCIES: Import Qdrant filtering utilities
  - PRESERVE: Existing query functionality

Task 9: CREATE tests/test_unified_repository_processor.py
  - IMPLEMENT: Comprehensive test suite covering all integration scenarios
  - FOLLOW pattern: tests/test_github_processor.py structure and mocking
  - NAMING: TestUnifiedRepositoryProcessor class with test_* methods
  - COVERAGE: Happy path, error cases, resource cleanup, cross-system linking
  - PLACEMENT: Main test suite in tests/ directory

Task 10: CREATE tests/test_file_id_linking.py
  - IMPLEMENT: Cross-system data linking verification tests
  - FOLLOW pattern: tests/conftest.py fixture usage
  - NAMING: TestFileIdLinking class focusing on Qdrant-Neo4j connection
  - MOCK: Both Qdrant and Neo4j services for isolated testing
  - COVERAGE: Bidirectional data queries, metadata consistency
```

### Implementation Patterns & Key Details

```python
# Core unified processing pattern
async def process_repository(
    self, 
    request: UnifiedIndexingRequest, 
    context: Context
) -> UnifiedIndexingResponse:
    """Unified repository processing with destination dispatch."""
    
    # PATTERN: Single resource acquisition (follow GitHubRepoManager pattern)
    repo_manager = GitHubRepoManager()
    
    try:
        # 1. Clone once (follow existing cloning pattern)
        repo_path = repo_manager.clone_repository(request.repo_url, request.max_size_mb)
        
        # 2. Discover files once (follow MultiFileDiscovery pattern)
        file_discovery = MultiFileDiscovery()
        discovered_files = file_discovery.discover_files(
            repo_path, request.file_types, request.max_files
        )
        
        # 3. Process by destination (NEW pattern)
        results = []
        for file_info in discovered_files:
            file_id = generate_file_id(request.repo_url, file_info['relative_path'])
            
            result = FileProcessingResult(
                file_id=file_id,
                file_path=file_info['file_path'],
                relative_path=file_info['relative_path'],
                language=file_info['language'],
                file_type=file_info['file_type']
            )
            
            # Dispatch to appropriate processors
            if request.destination in [IndexingDestination.QDRANT, IndexingDestination.BOTH]:
                await self._process_for_rag(file_info, file_id, context)
                result.processed_for_rag = True
                result.rag_chunks = file_info.get('chunk_count', 0)
            
            if request.destination in [IndexingDestination.NEO4J, IndexingDestination.BOTH]:
                await self._process_for_kg(file_info, file_id, context) 
                result.processed_for_kg = True
                result.kg_entities = file_info.get('entity_count', 0)
            
            results.append(result)
        
        return UnifiedIndexingResponse(
            success=True,
            repo_url=request.repo_url,
            destination=request.destination.value,
            files_processed=len(results),
            file_results=results
        )
    
    except Exception as e:
        # PATTERN: Detailed error responses (follow existing error handling)
        return UnifiedIndexingResponse(
            success=False, 
            repo_url=request.repo_url,
            errors=[str(e)]
        )
    finally:
        # CRITICAL: Always cleanup (follow github_processor cleanup pattern)
        repo_manager.cleanup()

# File ID generation pattern (ensure consistency)
def generate_file_id(repo_url: str, relative_path: str) -> str:
    """Generate consistent file_id for cross-system linking."""
    # PATTERN: Extract repo name from URL
    repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
    if '/' in repo_url.replace('https://github.com/', ''):
        owner_repo = '/'.join(repo_url.replace('https://github.com/', '').split('/')[:2])
        repo_name = owner_repo.replace('/', '__')  # Avoid path conflicts
    
    # CRITICAL: Normalize path separators for Windows compatibility  
    normalized_path = relative_path.replace(os.sep, '/')
    return f"{repo_name}:{normalized_path}"

# RAG processing with file_id (modify existing pattern)
async def _process_for_rag(self, file_info: dict, file_id: str, context: Context):
    """Process file for RAG with file_id metadata."""
    # PATTERN: Follow smart_crawl_github processor dispatch
    processor_map = {
        ".md": MarkdownProcessor,
        ".py": PythonProcessor,
        ".ts": TypeScriptProcessor,
        # ... existing processors
    }
    
    processor = processor_map.get(file_info['file_type'])
    if not processor:
        return
    
    # Extract and chunk content (follow existing pattern)
    extracted_items = processor().process_file(file_info['file_path'], file_info['relative_path'])
    
    for item in extracted_items:
        chunks = smart_chunk_content(item['content'], chunk_size=5000)
        
        for i, chunk in enumerate(chunks):
            # CRITICAL: Add file_id to metadata
            metadata = {
                **item.get('metadata', {}),
                'file_id': file_id,  # KEY ADDITION for cross-system linking
                'chunk_index': i,
                'source_type': 'unified_github_repository'
            }
            
            # Use existing RAG service
            await add_documents_to_vector_db(
                documents=[chunk],
                metadatas=[metadata], 
                source=file_id.split(':')[0],  # repo name
                qdrant_client=context.request_context.lifespan_context.qdrant_client
            )

# Neo4j processing with file_id (modify existing pattern)  
async def _process_for_kg(self, file_info: dict, file_id: str, context: Context):
    """Process file for knowledge graph with file_id property."""
    # PATTERN: Follow parse_repo_into_neo4j logic
    repo_extractor = context.request_context.lifespan_context.repo_extractor
    if not repo_extractor:
        return
    
    # Use existing Tree-sitter parsing
    language = file_info['language']
    if not TreeSitterParser.is_supported(language):
        return
    
    # Parse file structure (follow existing parsing pattern)
    with open(file_info['file_path'], 'r', encoding='utf-8') as f:
        content = f.read()
    
    parsed_elements = tree_sitter_parser.parse(content, file_info['file_path'], language)
    
    # CRITICAL: Add file_id property to Neo4j nodes
    file_node_query = """
    MERGE (f:File {path: $relative_path})
    SET f.file_id = $file_id,  // KEY ADDITION for cross-system linking
        f.name = $filename,
        f.language = $language,
        f.module_name = $module_name
    """
    
    await neo4j_session.run(file_node_query, {
        'relative_path': file_info['relative_path'],
        'file_id': file_id,  # Enable bidirectional queries
        'filename': os.path.basename(file_info['file_path']),
        'language': language,
        'module_name': file_info.get('module_name', '')
    })
```

### Integration Points

```yaml
QDRANT:
  - collection: "crawled_pages" (existing schema)  
  - modification: "Add file_id to metadata schema"
  - index: "CREATE INDEX file_id_filter FOR metadata.file_id"

NEO4J:
  - schema: "File nodes with file_id property"
  - modification: "ADD file_id STRING property to :File nodes"
  - constraint: "CREATE CONSTRAINT file_id_unique FOR (f:File) REQUIRE f.file_id IS UNIQUE"

MCP_CONTEXT:
  - access: "Both qdrant_client and repo_extractor from lifespan_context"
  - validation: "Check service availability before processing"
  - pattern: "Graceful degradation if services unavailable"

ENVIRONMENT_VARS:
  - required: "USE_KNOWLEDGE_GRAPH=true for Neo4j processing"
  - optional: "USE_HYBRID_SEARCH, USE_RERANKING for enhanced features"
  - pattern: "Feature flag validation before service calls"
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# Run after each file creation - fix before proceeding
ruff check src/tools/github_tools.py src/services/unified_indexing_service.py --fix
ruff format src/tools/github_tools.py src/services/unified_indexing_service.py

# Project-wide validation
ruff check src/ knowledge_graphs/ --fix
ruff format src/ knowledge_graphs/

# Expected: Zero errors. If errors exist, READ output and fix before proceeding.
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test unified processor components
pytest tests/test_unified_repository_processor.py -v
pytest tests/test_file_id_linking.py -v

# Test existing functionality preservation
pytest tests/test_github_processor.py -v  
pytest tests/test_rag_tools.py -v

# Cross-system integration tests
pytest tests/test_unified_repository_processor.py::TestCrossSystemLinking -v

# Expected: All tests pass. If failing, debug root cause and fix implementation.
```

### Level 3: Integration Testing (System Validation)

```bash
# Service startup validation with unified tool
python -c "
import asyncio
from src.core.app import create_app
app = create_app()
print('✓ MCP server starts with unified tool registered')
"

# Docker services health check
docker-compose ps
curl -f http://localhost:6333/health || echo "Qdrant health check failed"
docker exec mcp-neo4j cypher-shell -u neo4j -p password123 "RETURN 1;" || echo "Neo4j health check failed"

# Test unified tool functionality
python -c "
import asyncio
from unittest.mock import Mock
from src.tools.github_tools import index_github_repository

async def test_unified_tool():
    mock_context = Mock()
    # Setup context mocks based on test_github_processor.py patterns
    result = await index_github_repository(
        mock_context, 
        'https://github.com/test/small-repo',
        destination='both',
        file_types=['.md', '.py'],
        max_files=5
    )
    print(f'Unified tool result: {result}')

asyncio.run(test_unified_tool())
"

# Cross-system data linking validation  
python -c "
# Verify file_id linking between Qdrant and Neo4j
from src.clients.qdrant_client import QdrantClient
from neo4j import GraphDatabase

# Test that same file_id exists in both systems
qdrant_files = qdrant_client.search_by_metadata('file_id', 'test-repo:README.md')
neo4j_files = neo4j_session.run('MATCH (f:File {file_id: \"test-repo:README.md\"}) RETURN f')

print(f'Found in Qdrant: {len(qdrant_files)}, Found in Neo4j: {len(list(neo4j_files))}')
"

# Expected: All integrations working, cross-system queries functional
```

### Level 4: Performance & Domain-Specific Validation

```bash
# Performance comparison: unified vs separate tools
python -c "
import time
import asyncio

async def benchmark_performance():
    # Test 1: Separate tools (current approach)
    start = time.time()
    await smart_crawl_github(repo_url, file_types=['.md'])
    await parse_github_repository(repo_url) 
    separate_time = time.time() - start
    
    # Test 2: Unified tool
    start = time.time()
    await index_github_repository(repo_url, destination='both', file_types=['.md', '.py'])
    unified_time = time.time() - start
    
    improvement = ((separate_time - unified_time) / separate_time) * 100
    print(f'Performance improvement: {improvement:.1f}% faster')
    assert improvement >= 40, 'Should be at least 40% faster'

asyncio.run(benchmark_performance())
"

# RAG-KG synergy queries validation
python -c "
# Test cross-system queries enabled by file_id linking
from src.tools.rag_tools import perform_rag_query
from src.tools.kg_tools import query_knowledge_graph

async def test_synergy_queries():
    # 1. Find documentation for a specific class
    class_info = await query_knowledge_graph(
        'MATCH (c:Class {name: \"TestClass\"})-[:DEFINED_IN]->(f:File) RETURN f.file_id'
    )
    
    if class_info:
        file_id = class_info[0]['f.file_id'] 
        related_docs = await perform_rag_query(
            'TestClass documentation', 
            file_id=file_id
        )
        print(f'Found {len(related_docs)} related documentation chunks')
    
    # 2. Find code implementing mentioned concepts
    # Similar pattern in reverse direction
    
asyncio.run(test_synergy_queries())
"

# Resource usage validation
python tests/performance_benchmark.py

# Memory leak detection for unified processing
python -c "
import psutil
import gc
import asyncio

async def memory_leak_test():
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Process multiple repositories
    for i in range(5):
        await index_github_repository(f'https://github.com/test/repo{i}')
        gc.collect()
    
    final_memory = process.memory_info().rss
    memory_growth = final_memory - initial_memory
    print(f'Memory growth: {memory_growth / 1024 / 1024:.1f} MB')
    assert memory_growth < 100 * 1024 * 1024, 'Memory growth should be < 100MB'

asyncio.run(memory_leak_test())
"

# Expected: 40%+ performance improvement, functional synergy queries, stable memory usage
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] All tests pass: `pytest tests/ -v`  
- [ ] No linting errors: `ruff check src/ knowledge_graphs/`
- [ ] No formatting issues: `ruff format src/ knowledge_graphs/ --check`
- [ ] Performance improvement ≥40% measured for combined operations

### Feature Validation

- [ ] Single repository cloning for all destinations confirmed
- [ ] Qdrant documents contain file_id metadata
- [ ] Neo4j nodes contain file_id property  
- [ ] Cross-system queries work: Qdrant ↔ Neo4j via file_id
- [ ] All existing file types supported (12+ languages)
- [ ] Backward compatibility: existing tools still functional

### Code Quality Validation

- [ ] Follows existing MCP tool patterns from github_tools.py
- [ ] Service layer follows patterns from rag_service.py structure
- [ ] Error handling matches existing error response patterns
- [ ] Resource cleanup follows GitHubRepoManager cleanup patterns
- [ ] Async patterns follow existing MCP context usage

### Documentation & Deployment

- [ ] File_id generation logic documented and tested
- [ ] Cross-system query examples provided
- [ ] Environment variable requirements documented
- [ ] Migration path from separate tools planned
- [ ] Performance benchmarks documented

---

## Anti-Patterns to Avoid

- ❌ Don't skip resource cleanup - always use try/finally blocks
- ❌ Don't ignore feature flag checks - validate service availability
- ❌ Don't hardcode file_id format - use consistent generator utility
- ❌ Don't bypass existing processors - reuse established patterns
- ❌ Don't break backward compatibility - maintain existing tool signatures
- ❌ Don't ignore Windows path issues - use cross-platform patterns
- ❌ Don't skip Neo4j semaphore - prevent concurrent initialization deadlocks
- ❌ Don't assume services available - graceful degradation for missing services

**Confidence Score**: 9/10 for one-pass implementation success

This PRP provides comprehensive context including existing patterns, shared components, external documentation, specific gotchas, and detailed validation approaches required to successfully implement the unified repository processing feature.