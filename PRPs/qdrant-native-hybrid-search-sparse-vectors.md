# PRP: Qdrant Native Hybrid Search with Sparse Vectors

## Goal

Implement native Qdrant hybrid search using sparse vectors (FastBM25) to combine semantic search (dense vectors) with keyword search (sparse vectors), replacing the current client-side hybrid search implementation that uses scroll-based keyword filtering.

## Why

- **Precision Enhancement**: Dramatically improves search accuracy by combining semantic understanding with exact term matching
- **Performance Optimization**: Replaces inefficient client-side keyword filtering (scroll + filter) with native Qdrant hybrid search
- **Scalability**: Native Qdrant implementation scales better than fetching `match_count * 10` for client-side filtering
- **Technical Debt Reduction**: Eliminates the current hacky client-side hybrid approach that uses fixed 0.5 similarity scores
- **User Experience**: Critical for technical documentation where exact terms (like `QDRANT__SERVICE__ENABLE_CORS`) must be found alongside conceptual matches

## What

Transform the search system to use Qdrant's native multi-vector capabilities:

### Current Behavior (Client-Side Hybrid)
```python
# Dense search (semantic)
vector_results = search_documents(query, match_count * 2)

# Sparse search (keyword via scroll + filter)  
keyword_results = keyword_search_documents(query, match_count * 2)

# Client-side combination with 1.2x boost for hybrid matches
combined_results = combine_results(vector_results, keyword_results)
```

### Target Behavior (Native Qdrant Hybrid)
```python
# Single native hybrid search with both vectors
results = client.search_batch([
    SearchRequest(vector=NamedVector("dense", dense_vector), limit=match_count),
    SearchRequest(vector=NamedSparseVector("sparse", sparse_vector), limit=match_count)
])

# Qdrant handles fusion and scoring natively
```

### Success Criteria

- [ ] All existing search functionality works with improved accuracy
- [ ] Performance improvement: hybrid search <200ms (vs current >500ms)
- [ ] Exact term matching: queries like `QDRANT__SERVICE__ENABLE_CORS` return exact results first
- [ ] Semantic matching: conceptual queries still work with related terms
- [ ] Collection migration: existing data preserved during schema upgrade
- [ ] Backward compatibility: API interfaces unchanged
- [ ] Tests pass: All existing search tests continue working

## All Needed Context

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- url: https://qdrant.tech/articles/sparse-vectors/
  why: Core concepts and sparse vector theory

- url: https://qdrant.tech/articles/hybrid-search/
  why: Implementation patterns and best practices

- url: https://qdrant.tech/documentation/fastembed/fastembed-minicoil/
  why: FastBM25 integration details

- url: https://python.lang.chat/v0.1/docs/integrations/retrievers/qdrant-sparse/
  why: LangChain integration patterns

- file: src/qdrant_wrapper.py
  why: Current collection config and search patterns to maintain

- file: src/utils.py
  why: Embedding generation patterns to extend

- file: src/crawl4ai_mcp.py
  why: MCP tool interfaces to preserve

- file: tests/test_qdrant_wrapper.py
  why: Testing patterns and mock structures to follow

- url: https://gist.github.com/NirantK/b806c5c9a4812304f47693b641233f6e
  why: Working implementation example with named vectors
```

### Current Codebase Tree

```bash
src/
├── crawl4ai_mcp.py          # MCP tools (perform_rag_query)
├── qdrant_wrapper.py        # Collection config & search functions  
├── utils.py                 # Embedding generation & document processing
├── embedding_config.py      # Dynamic dimension detection
└── device_manager.py        # GPU/CPU handling

tests/
├── test_qdrant_wrapper.py   # Mock-based unit tests
├── integration_test.py      # Function-based integration tests
└── performance_benchmark.py # Timing and throughput tests

scripts/
├── fix_qdrant_dimensions.py # Collection recreation utility
└── clean_qdrant.py         # Database cleanup utility
```

### Target Codebase Changes

```bash
# MODIFY existing files
src/qdrant_wrapper.py        # Named vectors config + hybrid search
src/utils.py                 # Dual vector generation (dense + sparse)
src/crawl4ai_mcp.py          # Sparse vector generation for queries

# MODIFY dependencies
pyproject.toml               # Add qdrant-client[fast-bm25] dependency

# ADD new test cases
tests/test_qdrant_wrapper.py # Hybrid search and sparse vector tests
tests/test_hybrid_search.py  # Integration tests for hybrid functionality

# MODIFY scripts
scripts/fix_qdrant_dimensions.py # Handle named vector migration
```

### Known Gotchas & Library Quirks

```python
# CRITICAL: BM25 sparse vectors require IDF modifier in Qdrant
sparse_vectors_config={
    "text-sparse": SparseVectorParams(
        index=SparseIndexParams(
            on_disk=False  # REQUIRED for optimal performance
        ),
        modifier=models.Modifier.IDF  # REQUIRED for BM25
    )
}

# CRITICAL: Named vectors completely change collection schema
# Current: single unnamed vector per collection
# Target: multiple named vectors per collection
# This requires FULL collection recreation - all data will be lost

# CRITICAL: FastBM25 encoder must be "trained" on corpus
from fastembed import SparseTextEmbedding
encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
# Must call encoder.embed() to initialize internal BM25 statistics

# CRITICAL: Sparse vector encoding returns specific format
sparse_embedding = list(encoder.embed(["text"]))[0]
# Returns SparseEmbedding with .indices and .values attributes
# Must convert to Qdrant SparseVector format

# CRITICAL: Collection creation pattern change
# Current: vectors_config (single VectorParams)
# Target: vectors_config dict + sparse_vectors_config dict

# CRITICAL: Point structure changes completely
# Current: PointStruct(id=id, vector=dense_vector, payload=payload)
# Target: PointStruct(id=id, vector={"dense": dense_vec, "sparse": sparse_vec}, payload=payload)

# CRITICAL: Search API changes
# Current: client.search(query_vector=vector)
# Target: client.search_batch([SearchRequest(vector=NamedVector(...))])
```

## Implementation Blueprint

### Data Models and Structure

```python
# NEW: Sparse vector configuration model
@dataclass
class SparseVectorConfig:
    indices: List[int]
    values: List[float]
    
    def to_qdrant_sparse_vector(self) -> SparseVector:
        return SparseVector(indices=self.indices, values=self.values)

# EXTENDED: Collection configuration for named vectors
def get_hybrid_collections_config():
    embedding_dims = get_embedding_dimensions()
    
    return {
        "crawled_pages": {
            # Named dense vectors
            "vectors_config": {
                "text-dense": VectorParams(
                    size=embedding_dims,
                    distance=Distance.COSINE
                )
            },
            # Named sparse vectors  
            "sparse_vectors_config": {
                "text-sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                    modifier=models.Modifier.IDF
                )
            },
            "payload_schema": {...}  # Unchanged
        }
    }
```

### List of Tasks to Complete

```yaml
Task 1 - Dependency Setup:
MODIFY pyproject.toml:
  - FIND pattern: 'qdrant-client = '
  - REPLACE with: 'qdrant-client[fast-bm25] = '
  - RUN: uv sync
  - VERIFY: from fastembed import SparseTextEmbedding works

Task 2 - Collection Configuration Update:
MODIFY src/qdrant_wrapper.py:
  - FIND function: get_collections_config()
  - REPLACE entire function with named vectors configuration
  - ADD sparse_vectors_config to each collection
  - PRESERVE existing payload_schema patterns
  - ADD feature flag check: USE_HYBRID_SEARCH controls new vs old config

Task 3 - Sparse Vector Encoder Integration:
MODIFY src/utils.py:
  - CREATE function: create_sparse_embedding(text: str) -> SparseVectorConfig
  - INITIALIZE global FastBM25 encoder with lazy loading
  - HANDLE encoder training on first batch of documents
  - ADD error handling for encoding failures

Task 4 - Dual Vector Generation:
MODIFY src/utils.py functions:
  - UPDATE create_embeddings_batch() to return (dense_vectors, sparse_vectors)
  - MODIFY add_documents_to_qdrant() to create PointStruct with both vectors
  - PRESERVE existing batch processing patterns
  - MAINTAIN backward compatibility with feature flag

Task 5 - Collection Migration Strategy:
MODIFY src/qdrant_wrapper.py:
  - UPDATE _ensure_collections_exist() to detect schema changes
  - ADD hybrid_schema_migration() method for safe collection recreation
  - IMPLEMENT backup/restore pattern from scripts/fix_qdrant_dimensions.py
  - ADD user confirmation for data loss operations

Task 6 - Native Hybrid Search Implementation:
MODIFY src/qdrant_wrapper.py:
  - CREATE search_documents_hybrid() method using search_batch()
  - IMPLEMENT query sparse vector generation  
  - ADD result fusion with configurable weighting
  - MAINTAIN exact same interface as current search_documents()
  - PRESERVE all existing filter and pagination logic

Task 7 - MCP Tool Integration:
MODIFY src/crawl4ai_mcp.py:
  - UPDATE perform_rag_query to generate query sparse vectors
  - ROUTE to hybrid search when USE_HYBRID_SEARCH=true and collections support it
  - PRESERVE existing response format and error handling
  - ADD hybrid search metadata to response

Task 8 - Testing Implementation:
CREATE tests/test_hybrid_search.py:
  - FOLLOW patterns from tests/test_qdrant_wrapper.py
  - TEST sparse vector generation and encoding
  - TEST collection creation with named vectors
  - TEST hybrid search result fusion
  - TEST migration from old to new schema
  - MOCK FastBM25 encoder for unit tests

Task 9 - Performance Benchmarking:
MODIFY tests/performance_benchmark.py:
  - ADD benchmark_hybrid_search() function
  - COMPARE old client-side vs new native hybrid performance
  - MEASURE search latency and throughput
  - VALIDATE accuracy improvements with test queries

Task 10 - Documentation and Migration Guide:
UPDATE README.md and CLAUDE.md:
  - ADD USE_HYBRID_SEARCH configuration documentation
  - DOCUMENT collection migration process and data loss warning
  - ADD performance benchmarking results
  - UPDATE MCP tool documentation
```

### Task Implementation Pseudocode

```python
# Task 3: Sparse Vector Encoder
class SparseVectorEncoder:
    def __init__(self):
        self._encoder = None
        self._trained = False
    
    def _ensure_encoder(self):
        if self._encoder is None:
            # PATTERN: Lazy loading like get_qdrant_client()
            from fastembed import SparseTextEmbedding
            self._encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
    
    def encode(self, text: str) -> SparseVectorConfig:
        self._ensure_encoder()
        
        # CRITICAL: Handle encoder training on first use
        if not self._trained:
            # GOTCHA: Must call embed once to initialize BM25 stats
            _ = list(self._encoder.embed([text]))
            self._trained = True
        
        # Generate sparse embedding
        sparse_embedding = list(self._encoder.embed([text]))[0]
        
        return SparseVectorConfig(
            indices=sparse_embedding.indices.tolist(),
            values=sparse_embedding.values.tolist()
        )

# Task 6: Native Hybrid Search
async def search_documents_hybrid(
    self, 
    query_text: str,
    match_count: int = 10,
    **kwargs
) -> List[Dict[str, Any]]:
    # PATTERN: Generate both vector types like create_embeddings_batch()
    dense_vector = await create_embedding(query_text)
    sparse_vector = sparse_encoder.encode(query_text)
    
    # PATTERN: Use search_batch for multi-vector search
    requests = [
        SearchRequest(
            vector=NamedVector(name="text-dense", vector=dense_vector),
            limit=match_count,
            query_filter=self._build_filter(**kwargs)  # Preserve existing filter logic
        ),
        SearchRequest(
            vector=NamedSparseVector(
                name="text-sparse", 
                vector=sparse_vector.to_qdrant_sparse_vector()
            ),
            limit=match_count,
            query_filter=self._build_filter(**kwargs)
        )
    ]
    
    # CRITICAL: Native Qdrant fusion
    batch_results = self.client.search_batch(
        collection_name="crawled_pages",
        requests=requests
    )
    
    # PATTERN: Use existing normalize_search_results()
    dense_results = self.normalize_search_results(batch_results[0])
    sparse_results = self.normalize_search_results(batch_results[1])
    
    # IMPLEMENT: Reciprocal Rank Fusion (RRF)
    return self._fuse_results(dense_results, sparse_results, match_count)
```

### Integration Points

```yaml
CONFIGURATION:
  - add to: environment variables
  - pattern: USE_HYBRID_SEARCH=true enables new native implementation
  - fallback: When false, uses existing client-side hybrid search

COLLECTIONS:
  - migration: Automatic recreation when schema mismatch detected
  - warning: "Collection recreation will delete all existing data"
  - script: scripts/fix_qdrant_dimensions.py handles safe migration

API_COMPATIBILITY:
  - preserve: All existing MCP tool interfaces unchanged
  - enhance: Response includes hybrid_search_used: true when active
  - extend: search_documents() auto-detects collection capability

ERROR_HANDLING:
  - fallback: If sparse encoding fails, use dense-only search
  - retry: FastBM25 encoder failures retry with exponential backoff
  - logging: Clear warnings about collection recreation requirements
```

## Validation Loop

### Level 1: Syntax & Style

```bash
# Run these FIRST - fix any errors before proceeding
ruff check src/qdrant_wrapper.py src/utils.py src/crawl4ai_mcp.py --fix
mypy src/qdrant_wrapper.py src/utils.py src/crawl4ai_mcp.py

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests

```python
# NEW test cases following existing patterns from test_qdrant_wrapper.py

def test_sparse_vector_encoding():
    """Test FastBM25 sparse vector generation"""
    encoder = SparseVectorEncoder()
    result = encoder.encode("test document content")
    
    assert isinstance(result.indices, list)
    assert isinstance(result.values, list)
    assert len(result.indices) == len(result.values)
    assert all(isinstance(i, int) for i in result.indices)
    assert all(isinstance(v, float) for v in result.values)

@patch("qdrant_wrapper.QdrantClient")
def test_hybrid_collections_config(self, mock_client):
    """Test named vectors collection configuration"""
    config = get_hybrid_collections_config()
    
    # Verify named vectors structure
    assert "text-dense" in config["crawled_pages"]["vectors_config"]
    assert "text-sparse" in config["crawled_pages"]["sparse_vectors_config"]
    
    # Verify sparse vector has IDF modifier
    sparse_config = config["crawled_pages"]["sparse_vectors_config"]["text-sparse"]
    assert sparse_config.modifier == models.Modifier.IDF

@patch("qdrant_wrapper.QdrantClient")
def test_hybrid_search_result_fusion(self, mock_client):
    """Test native hybrid search with result fusion"""
    # Setup mock search_batch results
    mock_dense_results = [Mock(id="doc1", score=0.9)]
    mock_sparse_results = [Mock(id="doc2", score=0.8)]
    mock_client.return_value.search_batch.return_value = [
        mock_dense_results, mock_sparse_results
    ]
    
    wrapper = QdrantClientWrapper()
    results = wrapper.search_documents_hybrid("test query", match_count=5)
    
    # Verify fusion combines results appropriately
    assert len(results) <= 5
    mock_client.return_value.search_batch.assert_called_once()
```

```bash
# Run and iterate until passing:
uv run pytest tests/test_qdrant_wrapper.py::test_sparse_vector_encoding -v
uv run pytest tests/test_qdrant_wrapper.py::test_hybrid_collections_config -v
uv run pytest tests/test_hybrid_search.py -v

# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test

```bash
# Start Docker services (Qdrant)
docker-compose up -d

# Run integration tests
uv run pytest tests/integration_test.py -v

# Test MCP server functionality
uv run python -m src --dev

# Test hybrid search via MCP tool
echo '{"tool": "perform_rag_query", "arguments": {"query": "QDRANT__SERVICE__ENABLE_CORS", "match_count": 3}}' | \
  uv run python test_mcp_client.py

# Expected: Results show both semantic and exact keyword matching
# If error: Check logs and Qdrant container status
```

### Level 4: Performance & Creative Validation

```bash
# Performance benchmarking
uv run pytest tests/performance_benchmark.py::test_hybrid_search_performance -v

# Expected results:
# - Hybrid search <200ms (vs current >500ms)
# - Accuracy improvement measurable with test queries
# - Memory usage stable with sparse vectors

# Collection migration test
uv run python scripts/fix_qdrant_dimensions.py --dry-run

# Manual validation queries
curl -X POST http://localhost:8000/mcp/tools/perform_rag_query \
  -H "Content-Type: application/json" \
  -d '{"query": "QDRANT__SERVICE__ENABLE_CORS", "match_count": 5}'

# Verify exact term appears in top results
```

## Final Validation Checklist

- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check src/`  
- [ ] No type errors: `uv run mypy src/`
- [ ] Hybrid search performance: <200ms average response time
- [ ] Exact term matching: Technical terms return exact matches first
- [ ] Semantic search preserved: Conceptual queries still work
- [ ] Collection migration: Safe recreation process works
- [ ] Backward compatibility: USE_HYBRID_SEARCH=false still works
- [ ] Error handling: Graceful fallbacks when sparse encoding fails
- [ ] Documentation updated: CLAUDE.md reflects new configuration

---

## Anti-Patterns to Avoid

- ❌ Don't modify existing search interfaces - maintain exact compatibility
- ❌ Don't skip collection recreation warning - users must understand data loss
- ❌ Don't ignore FastBM25 encoder training - affects result quality
- ❌ Don't hardcode vector names - use consistent naming scheme
- ❌ Don't mix old and new search methods - use feature flag properly
- ❌ Don't skip sparse vector validation - empty vectors break search
- ❌ Don't assume collection exists - always validate schema compatibility

---

## Confidence Score: 9/10

This PRP provides comprehensive context for one-pass implementation:

**Strengths:**
- Complete understanding of current architecture and patterns
- Real-world implementation examples with working code
- Detailed collection migration strategy with data loss warnings
- Comprehensive testing strategy following existing patterns
- Performance benchmarking and validation criteria
- Extensive documentation of gotchas and common pitfalls

**Potential Risks:**
- Collection recreation is a breaking change requiring user coordination
- FastBM25 encoder behavior may vary across different document types
- Performance improvement assumptions need validation in real environment

The implementation follows established patterns in the codebase and provides sufficient context for successful execution.