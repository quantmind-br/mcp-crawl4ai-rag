# Qdrant Migration Patterns and Best Practices

## Core Migration Strategy: Supabase to Qdrant

### Data Model Mapping

| Supabase Component | Qdrant Equivalent | Migration Notes |
|-------------------|-------------------|-----------------|
| `sources` table | `sources` collection | Metadata-only collection |
| `crawled_pages` table | `crawled_pages` collection | Main vector collection |
| `code_examples` table | `code_examples` collection | Code-specific vectors |
| PostgreSQL RPC functions | Client-side search logic | Replicate filtering/sorting |
| JSONB metadata | Payload with JSON values | Use Qdrant filtering |
| Foreign key relationships | Application-level logic | Handle in client code |

### Vector Configuration Mapping

```python
# Supabase: vector(1536) with cosine similarity
# Qdrant equivalent:
from qdrant_client.models import VectorParams, Distance

vectors_config = VectorParams(
    size=1536,  # OpenAI embedding dimensions
    distance=Distance.COSINE  # Same as Supabase
)
```

### Search Function Migration

#### Supabase RPC: `match_crawled_pages`
```sql
-- Current PostgreSQL function
create or replace function match_crawled_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL
) returns table (...)
```

#### Qdrant Equivalent
```python
def search_documents(client, query, match_count=10, filter_metadata=None, source_id=None):
    query_embedding = create_embedding(query)
    
    # Build Qdrant filter
    filter_conditions = []
    if filter_metadata:
        for key, value in filter_metadata.items():
            filter_conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )
    
    if source_id:
        filter_conditions.append(
            FieldCondition(key="source_id", match=MatchValue(value=source_id))
        )
    
    query_filter = Filter(must=filter_conditions) if filter_conditions else None
    
    return client.search(
        collection_name="crawled_pages",
        query_vector=query_embedding,
        query_filter=query_filter,
        limit=match_count,
        with_payload=True,
        score_threshold=0.0  # Include all results like Supabase
    )
```

### Hybrid Search Migration

#### Current Supabase Pattern
```python
# Vector search
vector_results = search_documents(client, query, match_count * 2, filter_metadata)

# Keyword search  
keyword_query = client.from_('crawled_pages').select('*').ilike('content', f'%{query}%')
keyword_results = keyword_query.execute()

# Merge results with boosting
```

#### Qdrant Pattern
```python
def hybrid_search(client, query, match_count=10, filter_metadata=None):
    # Vector search
    vector_results = client.search(
        collection_name="crawled_pages", 
        query_vector=create_embedding(query),
        query_filter=build_filter(filter_metadata),
        limit=match_count * 2
    )
    
    # Keyword search via payload filtering
    keyword_filter = Filter(
        must=[
            FieldCondition(
                key="content",
                match=MatchText(text=query)  # Full-text search
            )
        ]
    )
    
    if filter_metadata:
        keyword_filter.must.extend(build_filter_conditions(filter_metadata))
    
    keyword_results = client.scroll(
        collection_name="crawled_pages",
        scroll_filter=keyword_filter,
        limit=match_count
    )[0]  # Get points from scroll result
    
    # Merge and deduplicate results
    return merge_results(vector_results, keyword_results, match_count)
```

### Batch Operations Migration

#### Current Supabase Pattern
```python
# Delete existing records
client.table("crawled_pages").delete().in_("url", unique_urls).execute()

# Batch insert
for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]
    client.table("crawled_pages").insert(batch).execute()
```

#### Qdrant Pattern
```python
def batch_upsert_documents(client, urls, contents, metadatas, embeddings, batch_size=100):
    # Convert to Qdrant points
    points = []
    for i, (url, content, metadata, embedding) in enumerate(zip(urls, contents, metadatas, embeddings)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),  # Generate unique IDs
            vector=embedding,
            payload={
                "url": url,
                "content": content,
                "chunk_number": metadata.get("chunk_index", 0),
                "source_id": metadata.get("source"),
                **metadata  # Include all metadata
            }
        ))
    
    # Batch upload
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name="crawled_pages",
            points=batch,
            wait=True  # Ensure consistency
        )
```

### Collection Management Patterns

#### Collection Setup
```python
def setup_qdrant_collections(client):
    """Initialize all required collections"""
    
    # Main documents collection
    client.create_collection(
        collection_name="crawled_pages",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        optimizers_config=OptimizersConfig(
            deleted_threshold=0.2,
            vacuum_min_vector_number=1000
        )
    )
    
    # Code examples collection 
    client.create_collection(
        collection_name="code_examples",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    
    # Sources metadata collection (optional - can use regular dict)
    # Sources don't need vectors, just metadata storage
```

### Error Handling Patterns

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def robust_qdrant_operation(client, operation_func, *args, **kwargs):
    """Wrapper for Qdrant operations with retry logic"""
    try:
        return operation_func(client, *args, **kwargs)
    except Exception as e:
        logging.error(f"Qdrant operation failed: {e}")
        raise

def safe_collection_exists(client, collection_name):
    """Check if collection exists without throwing"""
    try:
        client.get_collection(collection_name)
        return True
    except:
        return False
```

### Performance Optimization

#### Connection Configuration
```python
client = QdrantClient(
    host="localhost",
    port=6333,
    prefer_grpc=True,  # Better performance
    timeout=30,        # Longer timeout for large operations
)
```

#### Batch Size Optimization
```python
# Optimal batch sizes based on testing
BATCH_SIZES = {
    "small_docs": 200,   # < 1KB per document
    "medium_docs": 100,  # 1-10KB per document  
    "large_docs": 50,    # > 10KB per document
    "code_examples": 150 # Code snippets
}
```

### Memory Management
```python
def memory_efficient_migration(supabase_client, qdrant_client, table_name, collection_name):
    """Migrate large datasets without memory issues"""
    
    batch_size = 1000
    offset = 0
    
    while True:
        # Fetch batch from Supabase
        response = supabase_client.table(table_name)\
            .select("*")\
            .range(offset, offset + batch_size - 1)\
            .execute()
        
        if not response.data:
            break
            
        # Convert and upload to Qdrant
        points = convert_to_qdrant_points(response.data)
        qdrant_client.upsert(collection_name=collection_name, points=points)
        
        offset += batch_size
        
        # Clear memory
        del response, points
        gc.collect()
```

## Critical Migration Gotchas

### 1. ID Management
```python
# ISSUE: Supabase uses bigserial IDs, Qdrant uses string/int IDs
# SOLUTION: Generate UUIDs or use compound keys

def generate_qdrant_id(url, chunk_number):
    """Generate consistent IDs for Qdrant"""
    return f"{hashlib.md5(url.encode()).hexdigest()}_{chunk_number}"
```

### 2. Relationship Handling
```python
# ISSUE: No foreign keys in Qdrant
# SOLUTION: Embed relationships in payload

payload = {
    "content": content,
    "source_id": source_id,  # Embedded relationship
    "source_summary": source_summary,  # Denormalized data
    **metadata
}
```

### 3. Transaction Consistency
```python
# ISSUE: Qdrant doesn't have ACID transactions
# SOLUTION: Use application-level consistency

def atomic_update_source(qdrant_client, source_id, documents):
    """Update source and related documents atomically"""
    try:
        # 1. Update all documents first
        for doc in documents:
            qdrant_client.upsert(collection_name="crawled_pages", points=[doc])
        
        # 2. Update source metadata last
        update_source_metadata(source_id)
        
    except Exception as e:
        # Rollback logic here
        logging.error(f"Atomic update failed: {e}")
        raise
```

### 4. Search Result Format
```python
# ISSUE: Different result formats between Supabase and Qdrant
# SOLUTION: Normalize result format

def normalize_search_results(qdrant_results):
    """Convert Qdrant results to Supabase-compatible format"""
    normalized = []
    for hit in qdrant_results:
        normalized.append({
            "id": hit.id,
            "similarity": hit.score,
            "content": hit.payload.get("content"),
            "metadata": {k: v for k, v in hit.payload.items() if k != "content"},
            **hit.payload  # Include all payload fields
        })
    return normalized
```

## Testing Patterns

### Unit Tests
```python
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_qdrant_client():
    client = Mock()
    client.search.return_value = [
        Mock(id="test1", score=0.95, payload={"content": "test content"})
    ]
    return client

def test_search_documents(mock_qdrant_client):
    """Test basic search functionality"""
    results = search_documents(mock_qdrant_client, "test query")
    assert len(results) > 0
    assert results[0]["similarity"] == 0.95
```

### Integration Tests
```python
def test_full_migration_pipeline():
    """Test complete migration from Supabase to Qdrant"""
    # Setup test data in Supabase
    # Run migration
    # Verify data in Qdrant
    # Compare search results
    pass
```

This comprehensive guide covers all critical aspects of migrating from Supabase to Qdrant while maintaining functionality and performance.