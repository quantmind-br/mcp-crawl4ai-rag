name: "Supabase to Qdrant Migration with Local Infrastructure"
description: |
  Complete migration from Supabase vector database to local Qdrant instance while maintaining all RAG functionality, plus Docker Compose infrastructure setup for local development.

## Goal

Replace Supabase with Qdrant as the vector database while maintaining 100% compatibility with all existing RAG features. Create local development infrastructure with Docker Compose for Qdrant and Neo4j services, plus automation scripts for streamlined development workflow.

**End State:** 
- All existing MCP tools work identically with Qdrant backend
- Local Qdrant + Neo4j running via Docker Compose
- Main server runs via `uv` (not containerized)
- Automated setup/start scripts for development workflow
- Updated environment configuration for local services

## Why

- **Local Development**: Eliminate external Supabase dependency for local development
- **Cost Reduction**: Remove ongoing Supabase costs for development/testing
- **Performance**: Better control over vector database performance and configuration
- **Data Sovereignty**: Keep all data local during development
- **Consistency**: Unified local infrastructure for both Qdrant and Neo4j
- **Developer Experience**: Simplified setup with automation scripts

## What

**User-Visible Behavior:**
- All existing MCP tools continue to work identically
- Same search quality and performance characteristics
- Identical API responses and error handling
- All RAG strategies continue to function (contextual embeddings, hybrid search, agentic RAG, reranking)

**Technical Implementation:**
- Replace `supabase` dependency with `qdrant-client`
- Migrate all database operations from PostgreSQL/pgvector to Qdrant collections
- Create docker-compose.yaml for local Qdrant + Neo4j services
- Build setup.bat and start.bat automation scripts
- Update .env.example with local service configurations

### Success Criteria

- [ ] All existing MCP tools pass functional tests with Qdrant backend
- [ ] Vector search performance matches or exceeds Supabase performance
- [ ] Hybrid search functionality preserved with identical results
- [ ] All RAG strategies work without degradation
- [ ] Docker Compose stack starts reliably with health checks
- [ ] Setup/start scripts work on Windows development environment
- [ ] Migration from existing Supabase data works correctly
- [ ] Local development workflow requires no external dependencies

## All Needed Context

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- docfile: PRPs/ai_docs/qdrant_migration_patterns.md
  why: Complete migration patterns, error handling, and implementation details

- url: https://python-client.qdrant.tech/
  why: Official Qdrant Python client documentation and API reference

- url: https://qdrant.tech/documentation/quickstart/
  why: Collection setup, vector configuration, and search operations

- file: src/utils.py
  why: All current Supabase operations that need migration (lines 17-738)

- file: src/crawl4ai_mcp.py  
  why: MCP tool implementations and search logic (lines 806-868 hybrid search)

- file: crawled_pages.sql
  why: Current database schema and RPC functions to replicate

- url: https://neo4j.com/docs/operations-manual/current/docker/
  why: Neo4j Docker configuration and security setup
```

### Current Codebase Structure

```bash
E:\mcp-crawl4ai-rag\
├── src/
│   ├── crawl4ai_mcp.py          # Main MCP server - modify search functions
│   └── utils.py                 # Database operations - replace entirely  
├── knowledge_graphs/            # Neo4j integration - update connection strings
├── crawled_pages.sql           # PostgreSQL schema - analyze for migration
├── Dockerfile                  # Current container setup - reference patterns
├── pyproject.toml              # Dependencies - add qdrant-client, remove supabase  
├── .env.example                # Environment template - update for local services
└── README.md                   # Documentation - update for new setup
```

### Desired Codebase Structure with New Files

```bash
E:\mcp-crawl4ai-rag\
├── src/
│   ├── crawl4ai_mcp.py          # Updated with Qdrant integration
│   ├── utils.py                 # Rewritten for Qdrant operations
│   └── qdrant_client.py         # NEW: Qdrant client wrapper and utilities
├── docker-compose.yaml         # NEW: Local infrastructure definition
├── setup.bat                   # NEW: Initialize Docker stack
├── start.bat                   # NEW: Start MCP server  
├── scripts/
│   └── migrate_supabase.py     # NEW: Migration utility for existing data
├── .env.example                # Updated for local services
└── README.md                   # Updated setup instructions
```

### Known Gotchas & Library Quirks

```python
# CRITICAL: Qdrant uses different ID systems than Supabase
# Supabase: bigserial auto-increment IDs
# Qdrant: String or integer IDs (no auto-increment)
# SOLUTION: Generate UUIDs or composite keys from url+chunk_number

# CRITICAL: No foreign key relationships in Qdrant
# Current: sources table with foreign key references
# SOLUTION: Embed source metadata in document payloads (denormalization)

# CRITICAL: Different search result formats
# Supabase RPC returns: {id, similarity, content, metadata, source_id, ...}
# Qdrant returns: ScoredPoint{id, score, payload}
# SOLUTION: Normalize results in wrapper functions

# CRITICAL: Qdrant collections must exist before operations
# Unlike Supabase tables, collections don't auto-create
# SOLUTION: Initialize collections in lifespan startup

# CRITICAL: Environment variable changes
# OLD: SUPABASE_URL, SUPABASE_SERVICE_KEY  
# NEW: QDRANT_HOST, QDRANT_PORT (local Docker services)
# NEO4J_URI changes from localhost to service name in Docker
```

## Implementation Blueprint

### Data Models and Structure

Core data structure mapping preserves all functionality:

```python
# Collection Configuration
COLLECTIONS = {
    "crawled_pages": {
        "vectors_config": VectorParams(size=1536, distance=Distance.COSINE),
        "payload_schema": {
            "url": str,
            "content": str, 
            "chunk_number": int,
            "source_id": str,
            "metadata": dict,  # All existing JSONB metadata
            "created_at": str
        }
    },
    "code_examples": {
        "vectors_config": VectorParams(size=1536, distance=Distance.COSINE),
        "payload_schema": {
            "url": str,
            "content": str,
            "summary": str,
            "chunk_number": int, 
            "source_id": str,
            "metadata": dict,
            "created_at": str
        }
    }
}

# Sources handled as in-memory dict or separate collection (no vectors needed)
```

### List of Tasks (In Execution Order)

```yaml
Task 1:
MODIFY pyproject.toml:
  - REMOVE: "supabase==2.15.1" 
  - ADD: "qdrant-client>=1.12.0"
  - PRESERVE: All other dependencies exactly

CREATE docker-compose.yaml:
  - MIRROR pattern from: Research Neo4j configurations
  - ADD services: qdrant, neo4j
  - EXPOSE ports: 6333 (Qdrant), 7687/7474 (Neo4j)
  - INCLUDE: Health checks, volume persistence, environment variables

Task 2:
CREATE src/qdrant_client.py:
  - MIRROR pattern from: src/utils.py structure
  - IMPLEMENT: QdrantClient wrapper class
  - PRESERVE: Same function signatures as Supabase functions
  - INCLUDE: Connection management, error handling, retry logic

Task 3:  
MODIFY src/utils.py:
  - REPLACE: All Supabase client functions
  - MAINTAIN: Identical function signatures and return formats
  - PRESERVE: All error handling patterns
  - CONVERT: PostgreSQL RPC calls to Qdrant search operations

Task 4:
MODIFY src/crawl4ai_mcp.py:
  - UPDATE: Lifespan context initialization
  - REPLACE: Supabase client with Qdrant client
  - PRESERVE: All MCP tool implementations
  - MAINTAIN: Identical response formats

Task 5:
CREATE setup.bat:
  - IMPLEMENT: Docker Compose stack initialization
  - INCLUDE: Container build, network setup, health checks
  - ADD: Qdrant collection initialization
  - VERIFY: All services healthy before exit

CREATE start.bat:
  - IMPLEMENT: MCP server startup sequence
  - VERIFY: Docker services running
  - START: uv run src/crawl4ai_mcp.py
  - INCLUDE: Error handling and service checks

Task 6:
MODIFY .env.example:
  - REMOVE: SUPABASE_URL, SUPABASE_SERVICE_KEY
  - ADD: QDRANT_HOST=localhost, QDRANT_PORT=6333
  - UPDATE: NEO4J_URI=bolt://localhost:7687 (for local Docker)
  - PRESERVE: All other environment variables

Task 7:
CREATE scripts/migrate_supabase.py:
  - IMPLEMENT: Data migration utility
  - SUPPORT: Export from Supabase, import to Qdrant
  - PRESERVE: All metadata and relationships
  - VALIDATE: Data integrity after migration

Task 8:
MODIFY README.md:
  - UPDATE: Setup instructions for local Docker stack
  - REPLACE: Supabase setup with Qdrant setup
  - PRESERVE: All other documentation
  - ADD: Migration instructions for existing data
```

### Per Task Pseudocode

```python
# Task 2: Qdrant Client Implementation
class QdrantClientWrapper:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collections_exist()
    
    def _ensure_collections_exist(self):
        """Initialize collections if they don't exist"""
        for name, config in COLLECTIONS.items():
            if not self._collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=config["vectors_config"]
                )
    
    def add_documents_to_qdrant(self, urls, chunk_numbers, contents, metadatas, url_to_full_document):
        """PRESERVE exact signature from Supabase version"""
        # PATTERN: Convert to PointStruct objects
        points = []
        for url, chunk_num, content, metadata in zip(urls, chunk_numbers, contents, metadatas):
            # CRITICAL: Generate consistent IDs  
            point_id = f"{hashlib.md5(url.encode()).hexdigest()}_{chunk_num}"
            
            # PATTERN: Create embedding same as before
            embedding = create_embedding(content)
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "url": url,
                    "content": content,
                    "chunk_number": chunk_num,
                    "source_id": metadata.get("source"),
                    **metadata  # Preserve all metadata
                }
            ))
        
        # PATTERN: Batch upsert with error handling
        self._batch_upsert("crawled_pages", points)

# Task 3: Search Function Migration  
def search_documents(client, query, match_count=10, filter_metadata=None):
    """PRESERVE exact signature and behavior"""
    query_embedding = create_embedding(query)
    
    # PATTERN: Convert JSONB filter to Qdrant filter
    qdrant_filter = None
    if filter_metadata:
        conditions = []
        for key, value in filter_metadata.items():
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        qdrant_filter = Filter(must=conditions)
    
    # CRITICAL: Maintain same result format as Supabase RPC
    results = client.search(
        collection_name="crawled_pages",
        query_vector=query_embedding,
        query_filter=qdrant_filter,
        limit=match_count,
        with_payload=True
    )
    
    # PATTERN: Normalize to Supabase format
    return normalize_search_results(results)
```

### Integration Points

```yaml
ENVIRONMENT:
  - remove from: .env.example
  - variables: ["SUPABASE_URL", "SUPABASE_SERVICE_KEY"] 
  - add to: .env.example
  - variables: ["QDRANT_HOST=localhost", "QDRANT_PORT=6333"]

DOCKER:
  - create: docker-compose.yaml
  - services: ["qdrant:6333", "neo4j:7687/7474"]
  - volumes: ["qdrant_data", "neo4j_data", "neo4j_logs"]
  - networks: "bridge mode for service communication"

SCRIPTS:
  - create: setup.bat, start.bat
  - pattern: "Windows batch scripts with error handling"
  - functions: ["docker-compose up", "health checks", "uv run server"]
```

## Validation Loop

### Level 1: Syntax & Style

```bash
# Install new dependencies
uv pip install qdrant-client>=1.12.0

# Remove old dependency
uv pip uninstall supabase

# Type checking
mypy src/qdrant_client.py
mypy src/utils.py
mypy src/crawl4ai_mcp.py

# Expected: No type errors, clean imports
```

### Level 2: Unit Tests

```python
# CREATE test_qdrant_integration.py
import pytest
from unittest.mock import Mock, patch
from src.qdrant_client import QdrantClientWrapper

def test_collection_initialization():
    """Test that collections are created on startup"""
    with patch('qdrant_client.QdrantClient') as mock_client:
        wrapper = QdrantClientWrapper()
        # Verify create_collection called for each collection
        assert mock_client.return_value.create_collection.call_count == 2

def test_search_documents_format():
    """Test search results match Supabase format"""
    mock_results = [Mock(id="test1", score=0.95, payload={"content": "test"})]
    
    with patch('src.utils.create_embedding', return_value=[0.1]*1536):
        with patch.object(QdrantClient, 'search', return_value=mock_results):
            results = search_documents(None, "test query")
            
    assert len(results) == 1
    assert "similarity" in results[0]  # Qdrant score mapped to similarity
    assert results[0]["content"] == "test"

def test_hybrid_search_compatibility():
    """Test hybrid search maintains same interface"""
    # Test vector + keyword search combination
    # Verify result merging and boosting logic
    pass

def test_batch_operations():
    """Test batch insert performance and reliability"""
    # Test large document batches
    # Verify transaction-like behavior
    pass
```

```bash
# Run unit tests
uv run pytest test_qdrant_integration.py -v

# Expected: All tests pass, no Supabase dependencies
```

### Level 3: Integration Tests

```bash
# Start Docker infrastructure
docker-compose up -d qdrant neo4j

# Wait for services to be healthy
timeout 60 bash -c 'until docker-compose ps | grep -q "healthy"; do sleep 2; done'

# Start MCP server
uv run src/crawl4ai_mcp.py &
SERVER_PID=$!

# Test MCP tools functionality
python -c "
import json
from qdrant_client import QdrantClient

# Test basic connectivity
client = QdrantClient('localhost', port=6333)
collections = client.get_collections()
assert len(collections.collections) >= 2
print('✅ Qdrant collections initialized')

# Test embedding search
# (Add comprehensive MCP tool tests here)
"

# Cleanup
kill $SERVER_PID
docker-compose down

# Expected: All MCP tools work identically to Supabase version
```

### Level 4: Migration & Performance Validation

```bash
# Test data migration (if existing Supabase data)
python scripts/migrate_supabase.py --validate-only

# Performance benchmarking  
python -c "
import time
from src.utils import search_documents
from src.qdrant_client import QdrantClientWrapper

client = QdrantClientWrapper()

# Benchmark search performance
start = time.time()
results = search_documents(client, 'test query', match_count=50)
duration = time.time() - start

print(f'Search completed in {duration:.3f}s')
assert duration < 2.0  # Should be faster than 2s
assert len(results) > 0
print('✅ Performance meets requirements')
"

# Test automation scripts
setup.bat && start.bat
# Expected: Full stack starts without errors
```

## Final Validation Checklist

- [ ] All unit tests pass: `uv run pytest test_qdrant_integration.py -v`
- [ ] No import errors: `python -c "import src.crawl4ai_mcp"`
- [ ] Docker stack starts: `docker-compose up -d && docker-compose ps`
- [ ] MCP server starts: `uv run src/crawl4ai_mcp.py` (no errors in first 30s)
- [ ] All MCP tools functional: Test each tool via MCP client
- [ ] Search performance acceptable: <2s for typical queries
- [ ] Hybrid search results identical to previous implementation
- [ ] Migration script works: Successfully transfers existing data
- [ ] Automation scripts work: `setup.bat` and `start.bat` complete successfully
- [ ] Documentation updated: README reflects new setup process

---

## Anti-Patterns to Avoid

- ❌ Don't change MCP tool APIs - maintain exact compatibility
- ❌ Don't skip collection initialization - Qdrant requires explicit setup
- ❌ Don't ignore connection errors - implement robust retry logic  
- ❌ Don't change search result formats - preserve Supabase compatibility
- ❌ Don't mix Docker and local services randomly - use consistent patterns
- ❌ Don't hardcode connection strings - use environment variables
- ❌ Don't skip data validation during migration
- ❌ Don't ignore Windows batch script conventions in automation

**Confidence Score: 9/10** - Comprehensive research, detailed migration patterns, proven Qdrant integration examples, and thorough validation gates provide high confidence for successful one-pass implementation.