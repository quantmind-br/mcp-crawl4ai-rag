# Repository Indexing Pipeline Optimization PRP

## Goal

**Feature Goal**: Transform the repository indexing pipeline from sequential file-by-file processing to a high-performance multi-stage batch processing pipeline that eliminates network latency bottlenecks and utilizes multiple CPU cores efficiently.

**Deliverable**: Optimized `UnifiedIndexingService` class with batch processing stages for file reading, CPU-bound parsing, embedding generation, and database writes, achieving 5-10x performance improvement for large repositories.

**Success Definition**: Repository indexing time reduced by 70%+ for repositories with 50+ files, with true parallel CPU utilization for tree-sitter parsing and bulk database operations for both Qdrant and Neo4j.

## User Persona (if applicable)

**Target User**: AI agents and developers using the MCP server for large-scale repository analysis and indexing

**Use Case**: Indexing enterprise repositories (100-1000 files) for RAG-based code analysis, AI hallucination detection, and knowledge graph construction

**User Journey**: 
1. User calls `index_github_repository` MCP tool with large repository URL
2. System processes repository efficiently using all available CPU cores
3. User receives completion notification in significantly reduced time
4. Indexed data available for immediate querying in both Qdrant and Neo4j

**Pain Points Addressed**: 
- Long indexing times (10+ minutes for large repositories)
- Poor CPU utilization (single-core processing despite multi-core systems)
- Memory inefficiency from small batch operations
- Network latency from individual database operations

## Why

- **Performance Optimization**: Current pipeline processes files sequentially, severely underutilizing modern multi-core systems and creating network bottlenecks
- **Scalability Requirements**: Enterprise repositories with 500+ files require hours to index with current implementation
- **Resource Efficiency**: Existing small batch sizes (5 files, 100 Qdrant documents) create excessive network round trips and poor throughput
- **CPU Utilization**: Tree-sitter parsing is CPU-bound but limited by Python's GIL when using ThreadPoolExecutor

## What

Transform the current file-by-file processing into a multi-stage pipeline:

1. **Stage 1**: Parallel file reading using aiofiles for non-blocking I/O
2. **Stage 2**: CPU-bound parsing using ProcessPoolExecutor to bypass GIL limitations  
3. **Stage 3**: Batch embedding generation with accumulation across multiple files
4. **Stage 4**: Bulk database writes (500+ documents for Qdrant, 5000+ operations for Neo4j)

### Success Criteria

- [ ] Processing time reduced by 70%+ for repositories with 50+ files
- [ ] CPU utilization increases to use all available cores during tree-sitter parsing
- [ ] Network round trips reduced by 90%+ through bulk database operations
- [ ] Memory usage remains stable despite larger batch sizes
- [ ] All existing functionality preserved (RAG chunking, Neo4j parsing, error handling)
- [ ] Configurable parallelism and batch sizes via environment variables

## All Needed Context

### Context Completeness Check

_This PRP provides complete implementation context including current architecture analysis, specific optimization patterns from the codebase, external documentation references, and detailed implementation patterns. An implementing agent will have all necessary information to successfully optimize the pipeline without prior codebase knowledge._

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- url: https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor
  why: Official Python documentation for ProcessPoolExecutor integration with asyncio
  critical: Understanding loop.run_in_executor() for CPU-bound tasks and error handling patterns

- url: https://qdrant.tech/documentation/database-tutorials/bulk-upload/
  why: Qdrant batch insertion best practices and memory optimization strategies
  critical: Optimal batch sizes (500-1000 vectors), memory management, and indexing configuration

- url: https://neo4j.com/docs/python-manual/current/performance/
  why: Neo4j bulk operations using UNWIND and transaction batching patterns
  critical: UNWIND pattern for 5000+ record batches and transaction optimization

- file: src/services/unified_indexing_service.py
  why: Current pipeline implementation with batch processing patterns to preserve
  pattern: _process_files_unified method structure and error handling
  gotcha: Neo4j analysis accumulation pattern must be preserved

- file: src/services/embedding_service.py
  why: Existing batch embedding patterns and Redis cache integration
  pattern: create_embeddings_batch method and cache hit/miss handling
  gotcha: OpenAI API has 1000-2000 input limit for batch requests

- file: src/clients/qdrant_client.py
  why: Current Qdrant batch patterns and memory management strategies
  pattern: add_documents_to_qdrant generator pattern for memory efficiency
  gotcha: Batch size of 100 is too small, should increase to 500-1000

- file: src/k_graph/parsing/tree_sitter_parser.py
  why: CPU-intensive parsing logic that needs ProcessPoolExecutor optimization
  pattern: Tree-sitter parsing workflow and language detection
  gotcha: Tree-sitter objects are not pickle-serializable, need module-level functions

- file: src/services/rag_service.py
  why: Existing ThreadPoolExecutor patterns for contextual processing
  pattern: Lines 435-457 show proper executor usage with asyncio.gather
  gotcha: Keep ThreadPoolExecutor for I/O-bound operations, use ProcessPoolExecutor only for CPU-bound

- file: src/core/context.py
  why: Singleton context pattern for managing executors and connections
  pattern: Async context management and resource cleanup
  gotcha: Need to add ProcessPoolExecutor to context for proper lifecycle management
```

### Current Codebase Tree

```bash
src/
├── core/
│   ├── app.py                      # FastMCP server setup and tool registration
│   └── context.py                  # Singleton context management
├── tools/
│   └── github_tools.py             # index_github_repository tool (entry point)
├── services/
│   ├── unified_indexing_service.py # MAIN OPTIMIZATION TARGET
│   ├── embedding_service.py        # Batch embedding patterns
│   └── rag_service.py             # ThreadPoolExecutor patterns
├── clients/
│   └── qdrant_client.py           # Qdrant batch operation patterns
└── k_graph/
    └── parsing/
        └── tree_sitter_parser.py  # CPU-bound parsing for ProcessPoolExecutor
```

### Desired Codebase Tree

```bash
src/
├── services/
│   ├── unified_indexing_service.py # MODIFIED: Multi-stage pipeline
│   └── batch_processing/           # NEW: Batch processing utilities
│       ├── __init__.py
│       ├── file_processor.py      # NEW: Module-level functions for ProcessPoolExecutor
│       └── pipeline_stages.py     # NEW: Pipeline stage coordination
├── core/
│   └── context.py                 # MODIFIED: Add ProcessPoolExecutor management
└── utils/
    └── performance_config.py      # NEW: Configurable batch sizes and worker counts
```

### Known Gotchas of our codebase & Library Quirks

```python
# CRITICAL: Tree-sitter Parser objects cannot be pickled for ProcessPoolExecutor
# Solution: Extract parsing logic to module-level functions that recreate parsers
def parse_file_for_multiprocessing(file_path: str, content: str, language: str) -> Dict:
    # Must recreate parser in each process
    pass

# CRITICAL: OpenAI embeddings API has batch size limits
MAX_EMBEDDING_BATCH_SIZE = 1000  # Prevent API errors

# CRITICAL: Neo4j driver sessions are not thread-safe
# Current code properly handles this with async session management

# CRITICAL: Qdrant batch size 100 is too small for optimal performance
# Research shows 500-1000 is optimal for throughput vs memory

# CRITICAL: aiofiles does not support cancellation - can cause deadlocks
# Avoid using in task groups that might be cancelled

# CRITICAL: ProcessPoolExecutor has significant startup overhead
# Should be initialized once and reused, managed in context.py
```

## Implementation Blueprint

### Data Models and Structure

Current data models are sufficient - no new models needed. Key structures to preserve:

```python
# Existing models to maintain
@dataclass
class UnifiedIndexingRequest:
    repo_url: str
    should_process_rag: bool
    should_process_kg: bool
    chunk_size: int = 5000

@dataclass
class FileProcessingResult:
    file_id: str
    processed_for_rag: bool
    processed_for_kg: bool
    processing_time_seconds: float
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE src/utils/performance_config.py
  - IMPLEMENT: Configuration class for batch sizes and worker counts
  - FOLLOW pattern: Environment variable loading with defaults
  - NAMING: BATCH_SIZE_QDRANT, BATCH_SIZE_NEO4J, CPU_WORKERS, IO_WORKERS
  - PLACEMENT: Utils directory for cross-service configuration

Task 2: CREATE src/services/batch_processing/file_processor.py
  - IMPLEMENT: Module-level functions for ProcessPoolExecutor compatibility
  - FUNCTIONS: parse_file_for_kg(file_path, content, language), read_file_async(file_path)
  - CRITICAL: Functions must be picklable - no class dependencies
  - FOLLOW pattern: src/k_graph/parsing/tree_sitter_parser.py (parsing logic)
  - GOTCHA: Recreate parsers in each worker process, handle serialization

Task 3: CREATE src/services/batch_processing/pipeline_stages.py
  - IMPLEMENT: Pipeline coordination class with stage management
  - METHODS: stage_read_files(), stage_parse_files(), stage_generate_embeddings(), stage_write_databases()
  - FOLLOW pattern: src/services/rag_service.py (asyncio.gather patterns)
  - DEPENDENCIES: Import file_processor functions from Task 2

Task 4: MODIFY src/core/context.py
  - ADD: ProcessPoolExecutor to Crawl4AIContext dataclass
  - IMPLEMENT: Proper lifecycle management (startup/shutdown)
  - FOLLOW pattern: Existing ThreadPoolExecutor and async context patterns
  - CRITICAL: Use context manager for proper cleanup

Task 5: MODIFY src/services/unified_indexing_service.py
  - REFACTOR: _process_files_unified method to use multi-stage pipeline
  - REPLACE: Sequential file processing with batch accumulation
  - INTEGRATE: Pipeline stages from Task 3
  - PRESERVE: Error handling, progress tracking, and existing functionality
  - CRITICAL: Maintain backward compatibility with tool interface

Task 6: MODIFY src/clients/qdrant_client.py
  - INCREASE: Default batch size from 100 to 500-1000
  - OPTIMIZE: Memory management for larger batches
  - FOLLOW pattern: Existing generator-based batching
  - PRESERVE: Hybrid search functionality and error handling

Task 7: ENHANCE src/k_graph/analysis/ Neo4j operations
  - IMPLEMENT: Bulk UNWIND operations for node/relationship creation
  - REPLACE: Individual session.run() calls with batch operations
  - BATCH SIZE: 5000+ operations per transaction
  - FOLLOW pattern: Official Neo4j bulk loading documentation

Task 8: CREATE tests/unit/services/test_batch_processing.py
  - IMPLEMENT: Unit tests for all new batch processing components
  - COVER: ProcessPoolExecutor integration, pipeline stages, error handling
  - MOCK: File system operations and database calls
  - VALIDATE: Performance improvements and resource management

Task 9: CREATE tests/integration/test_pipeline_optimization.py
  - IMPLEMENT: End-to-end integration tests
  - VALIDATE: Complete pipeline with real repository data
  - MEASURE: Performance improvements vs current implementation
  - TEST: Error recovery and resource cleanup
```

### Implementation Patterns & Key Details

```python
# Pipeline stage coordination pattern
class OptimizedIndexingPipeline:
    def __init__(self, io_executor, cpu_executor, config):
        self.io_executor = io_executor  # ThreadPoolExecutor for I/O
        self.cpu_executor = cpu_executor  # ProcessPoolExecutor for CPU
        self.config = config
    
    async def process_files_optimized(self, files: List[Path]) -> List[ProcessingResult]:
        # Stage 1: Async file reading (I/O bound)
        file_contents = await self._stage_read_files(files)
        
        # Stage 2: Parallel parsing (CPU bound) 
        parsed_data = await self._stage_parse_files(file_contents)
        
        # Stage 3: Batch embedding generation
        if should_process_rag:
            embeddings = await self._stage_generate_embeddings(parsed_data)
        
        # Stage 4: Bulk database writes
        await self._stage_write_databases(embeddings, parsed_data)

# Module-level function for ProcessPoolExecutor (must be importable)
def parse_file_with_tree_sitter(file_path: str, content: str, language: str) -> Dict[str, Any]:
    """Parse file using tree-sitter - module level for multiprocessing."""
    # CRITICAL: Recreate parser in each process
    from src.k_graph.parsing.parser_factory import get_global_factory
    
    factory = get_global_factory()
    parser = factory.get_parser_for_language(language)
    
    if parser:
        result = parser.parse(content, file_path)
        return {
            "file_path": file_path,
            "classes": result.classes,
            "functions": result.functions,
            # ... other serializable data
        }
    return None

# Async file reading with aiofiles
async def read_file_async(file_path: str) -> Tuple[str, str]:
    """Read file asynchronously."""
    import aiofiles
    async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = await f.read()
    return file_path, content

# Bulk Qdrant operations
async def bulk_add_to_qdrant(self, all_documents: List[DocumentBatch]):
    """Add documents in large batches for optimal performance."""
    batch_size = self.config.BATCH_SIZE_QDRANT  # 500-1000
    
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i + batch_size]
        # Process large batch...

# Bulk Neo4j operations using UNWIND
async def bulk_create_neo4j_entities(self, all_analyses: List[Analysis]):
    """Create Neo4j entities using bulk UNWIND operations."""
    batch_size = self.config.BATCH_SIZE_NEO4J  # 5000
    
    # Bulk node creation
    await session.run("""
        UNWIND $nodes AS node
        MERGE (n:Class {full_name: node.full_name})
        ON CREATE SET n.name = node.name, n.file_path = node.file_path
    """, nodes=class_nodes)
```

### Integration Points

```yaml
ENVIRONMENT_CONFIG:
  - add to: .env.example and environment configuration
  - variables: 
    - "CPU_WORKERS=4  # ProcessPoolExecutor workers for parsing"
    - "IO_WORKERS=10  # ThreadPoolExecutor workers for I/O"
    - "BATCH_SIZE_QDRANT=500  # Qdrant batch size"
    - "BATCH_SIZE_NEO4J=5000  # Neo4j UNWIND batch size"
    - "BATCH_SIZE_EMBEDDINGS=1000  # OpenAI embeddings batch size"

CONTEXT_MANAGEMENT:
  - modify: src/core/context.py
  - pattern: "Add ProcessPoolExecutor to async context lifecycle"
  - integration: "Ensure proper startup/shutdown coordination"

TOOL_INTERFACE:
  - preserve: src/tools/github_tools.py interface unchanged
  - maintain: All existing parameters and response format
  - enhance: Add performance metrics to response

MONITORING:
  - add: Performance metrics collection
  - track: Processing time per stage, CPU utilization, memory usage
  - pattern: Similar to existing progress tracking in UnifiedIndexingService
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# Run after each file creation
uv run ruff check src/services/batch_processing/ --fix
uv run mypy src/services/batch_processing/
uv run ruff format src/services/batch_processing/

# Check modified files
uv run ruff check src/services/unified_indexing_service.py --fix
uv run mypy src/services/unified_indexing_service.py

# Project-wide validation
uv run ruff check src/ --fix
uv run mypy src/
uv run ruff format src/

# Expected: Zero errors. Fix any issues before proceeding.
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test new batch processing components
uv run pytest tests/unit/services/test_batch_processing.py -v

# Test modified unified indexing service
uv run pytest tests/unit/services/test_unified_indexing_service.py -v

# Test integration points
uv run pytest tests/unit/core/test_context.py -v
uv run pytest tests/unit/clients/test_qdrant_client.py -v

# Full test suite for affected areas
uv run pytest tests/unit/services/ -v
uv run pytest tests/unit/tools/ -v

# Coverage validation
uv run pytest tests/ --cov=src/services --cov-report=term-missing

# Expected: All tests pass. Debug and fix any failures.
```

### Level 3: Integration Testing (System Validation)

```bash
# Start required services
docker-compose up -d qdrant neo4j

# Start MCP server
uv run -m src &
sleep 5

# Test optimized indexing with small repository
curl -X POST http://localhost:8080/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "index_github_repository", 
    "arguments": {
      "repo_url": "https://github.com/small-test/repo",
      "max_files": 10,
      "destination": "both"
    }
  }' | jq .

# Performance comparison test
time curl -X POST http://localhost:8080/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "index_github_repository",
    "arguments": {
      "repo_url": "https://github.com/medium-test/repo", 
      "max_files": 50,
      "destination": "both"
    }
  }'

# Validate database operations
docker exec qdrant-container curl http://localhost:6333/collections/default/points/count
docker exec neo4j-container cypher-shell "MATCH (n) RETURN count(n)"

# Resource utilization check during processing
htop  # Verify multi-core CPU usage during parsing stage

# Expected: Significantly faster processing, multi-core utilization, no errors
```

### Level 4: Performance & Load Testing

```bash
# Performance benchmarking
python scripts/benchmark_pipeline.py \
  --old-implementation \
  --new-implementation \
  --repository-sizes 10,50,100,200

# Memory usage profiling
python -m memory_profiler scripts/profile_indexing.py

# Concurrent processing test
for i in {1..3}; do
  curl -X POST http://localhost:8080/tools/call \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"index_github_repository\", \"arguments\": {\"repo_url\": \"https://github.com/test-repo-${i}\"}}" &
done
wait

# CPU utilization validation
# Monitor with htop during large repository processing - should see all cores active

# Database performance validation
# Check Qdrant and Neo4j for proper batch operations and no connection issues

# Expected: 70%+ performance improvement, stable memory usage, all cores utilized
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check src/`
- [ ] No type errors: `uv run mypy src/`
- [ ] No formatting issues: `uv run ruff format src/ --check`

### Feature Validation

- [ ] Repository indexing time reduced by 70%+ for repositories with 50+ files
- [ ] CPU utilization shows all cores active during tree-sitter parsing phase
- [ ] Network round trips reduced (verified via logging/monitoring)
- [ ] Qdrant batch sizes increased to 500+ documents per operation
- [ ] Neo4j operations use UNWIND for bulk processing (5000+ records)
- [ ] All existing functionality preserved (RAG chunking, Neo4j parsing, error handling)
- [ ] Memory usage remains stable despite larger batch sizes

### Code Quality Validation

- [ ] Follows existing async/await patterns from codebase
- [ ] ProcessPoolExecutor properly managed in context lifecycle
- [ ] Module-level functions properly designed for multiprocessing
- [ ] Error handling preserves existing patterns and adds timeout protection
- [ ] Configuration properly externalized via environment variables
- [ ] Backwards compatibility maintained for all tool interfaces

### Performance Validation

- [ ] Benchmark results show 70%+ improvement for target repository sizes
- [ ] CPU monitoring confirms multi-core utilization during parsing
- [ ] Memory profiling shows stable usage patterns
- [ ] Database operation logs confirm bulk processing patterns
- [ ] No performance regressions for small repositories (< 10 files)

---

## Anti-Patterns to Avoid

- ❌ Don't use ProcessPoolExecutor for I/O-bound operations (file reading, database writes)
- ❌ Don't pickle complex objects like tree-sitter parsers - use module-level functions
- ❌ Don't ignore OpenAI API batch size limits - implement proper chunking
- ❌ Don't use aiofiles in cancellable task groups - can cause deadlocks
- ❌ Don't forget to properly shutdown ProcessPoolExecutor in context cleanup
- ❌ Don't make batch sizes too large - balance memory usage vs performance
- ❌ Don't remove existing error handling - enhance it with timeout protection
- ❌ Don't break backwards compatibility - preserve all existing tool interfaces