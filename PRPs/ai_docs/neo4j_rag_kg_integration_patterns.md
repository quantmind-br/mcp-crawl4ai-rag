# Neo4j + Vector Database Integration Patterns for RAG-KG Systems

## Overview

This document consolidates best practices for integrating Neo4j knowledge graphs with vector databases to create RAG-KG (Retrieval Augmented Generation with Knowledge Graphs) systems. The patterns are designed specifically for code repository processing and unified data pipelines.

## Core Integration Architecture

### 1. Bidirectional Data Flow Pattern

**Pattern**: Synchronize data between graph and vector stores with consistent identifiers

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import hashlib

@dataclass
class UnifiedDataRecord:
    """Unified record structure for both Neo4j and vector DB."""
    id: str                    # Consistent ID across both systems
    content: str              # Text content for vector embedding
    metadata: Dict[str, Any]  # Structured metadata for Neo4j
    embedding_vector: Optional[List[float]] = None
    relationships: List[Dict] = None  # Graph relationships
    
    def generate_id(self) -> str:
        """Generate consistent ID from content hash."""
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
        return f"{self.metadata.get('type', 'unknown')}_{content_hash}"

class RAGKGIntegrator:
    """Manages bidirectional sync between Neo4j and vector DB."""
    
    def __init__(self, neo4j_driver, vector_db_client):
        self.neo4j = neo4j_driver
        self.vector_db = vector_db_client
        self.embedding_model = None  # Initialize embedding model
    
    async def store_unified_record(self, record: UnifiedDataRecord):
        """Store record in both systems with consistent linking."""
        
        # 1. Generate embedding if not provided
        if not record.embedding_vector:
            record.embedding_vector = await self._generate_embedding(record.content)
        
        # 2. Store in vector database with metadata
        await self._store_in_vector_db(record)
        
        # 3. Store in Neo4j with vector reference
        await self._store_in_neo4j(record)
        
        # 4. Create bidirectional links
        await self._create_cross_references(record)
    
    async def _create_cross_references(self, record: UnifiedDataRecord):
        """Create links between graph nodes and vector embeddings."""
        
        # Store vector DB reference in Neo4j
        cypher = """
        MATCH (n {unified_id: $id})
        SET n.vector_db_id = $vector_id,
            n.has_embedding = true,
            n.embedding_dimension = $dimension
        """
        await self.neo4j.execute_query(
            cypher,
            id=record.id,
            vector_id=record.id,  # Use same ID in both systems
            dimension=len(record.embedding_vector)
        )
        
        # Store Neo4j reference in vector metadata
        vector_metadata = {
            **record.metadata,
            "neo4j_node_id": record.id,
            "has_graph_context": True
        }
        await self.vector_db.upsert_metadata(record.id, vector_metadata)
```

### 2. Schema Design for Code Repositories

**Neo4j Schema for Code Analysis:**

```cypher
// Core node types for repository processing
CREATE CONSTRAINT repository_name IF NOT EXISTS FOR (r:Repository) REQUIRE r.name IS UNIQUE;
CREATE CONSTRAINT file_path IF NOT EXISTS FOR (f:File) REQUIRE (f.repository_name, f.path) IS UNIQUE;
CREATE CONSTRAINT class_full_name IF NOT EXISTS FOR (c:Class) REQUIRE (c.repository_name, c.full_name) IS UNIQUE;
CREATE CONSTRAINT function_signature IF NOT EXISTS FOR (f:Function) REQUIRE (f.repository_name, f.signature) IS UNIQUE;

// Schema with vector integration fields
CREATE (r:Repository {
  name: $repo_name,
  url: $repo_url,
  unified_id: $unified_id,
  vector_db_collection: $collection_name,
  embedding_model: $model_name,
  created_at: datetime()
})

CREATE (f:File {
  path: $file_path,
  repository_name: $repo_name,
  unified_id: $unified_id,
  vector_db_id: $vector_id,
  has_embedding: true,
  content_hash: $hash,
  language: $detected_language,
  chunk_ids: $chunk_references  // References to chunked content in vector DB
})

CREATE (c:Class {
  name: $class_name,
  full_name: $full_qualified_name,
  repository_name: $repo_name,
  unified_id: $unified_id,
  vector_db_id: $vector_id,
  has_embedding: true,
  semantic_similarity_group: $group_id  // For clustering similar classes
})

// Relationships with vector context
CREATE (r)-[:CONTAINS]->(f)
CREATE (f)-[:DEFINES]->(c)
CREATE (c)-[:SIMILAR_TO {similarity_score: $score}]->(c2)  // From vector similarity
```

### 3. Hybrid Retrieval Pattern

**Pattern**: Combine graph traversal with vector similarity search

```python
class HybridRetriever:
    """Combines graph traversal and vector search for comprehensive retrieval."""
    
    def __init__(self, rag_kg_integrator: RAGKGIntegrator):
        self.integrator = rag_kg_integrator
        self.similarity_threshold = 0.75
    
    async def hybrid_search(
        self, 
        query: str, 
        search_depth: int = 2,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and graph methods."""
        
        # Phase 1: Vector similarity search
        vector_results = await self._vector_similarity_search(query, max_results * 2)
        
        # Phase 2: Graph expansion from vector results
        expanded_results = []
        for result in vector_results:
            # Get Neo4j node using unified ID
            graph_context = await self._get_graph_context(
                result['unified_id'], 
                search_depth
            )
            
            # Combine vector result with graph context
            enhanced_result = {
                **result,
                'graph_context': graph_context,
                'related_entities': await self._get_related_entities(result['unified_id']),
                'path_to_root': await self._get_path_to_repository_root(result['unified_id'])
            }
            expanded_results.append(enhanced_result)
        
        # Phase 3: Semantic deduplication and ranking
        final_results = await self._deduplicate_and_rank(expanded_results)
        
        return final_results[:max_results]
    
    async def _get_graph_context(self, unified_id: str, depth: int) -> Dict[str, Any]:
        """Get graph context around a node."""
        cypher = """
        MATCH path = (n {unified_id: $id})-[*0..$depth]-(related)
        RETURN 
            n as center_node,
            collect(DISTINCT related) as related_nodes,
            collect(DISTINCT relationships(path)) as relationships,
            collect(DISTINCT labels(related)) as related_types
        """
        
        result = await self.integrator.neo4j.execute_query(
            cypher, 
            id=unified_id, 
            depth=depth
        )
        
        return {
            'center_node': result[0]['center_node'],
            'related_nodes': result[0]['related_nodes'],
            'relationships': result[0]['relationships'],
            'entity_types': set(result[0]['related_types'])
        }
```

### 4. Multi-Hop Reasoning Pattern

**Pattern**: Enable complex queries that traverse graph relationships

```python
class GraphReasoningEngine:
    """Enables multi-hop reasoning over code repositories."""
    
    def __init__(self, rag_kg_integrator: RAGKGIntegrator):
        self.integrator = rag_kg_integrator
    
    async def find_implementation_path(
        self, 
        interface_name: str, 
        repository_name: str
    ) -> List[Dict[str, Any]]:
        """Find implementation chain from interface to concrete classes."""
        
        cypher = """
        MATCH path = (interface:Class {name: $interface_name, repository_name: $repo})
                    -[:INHERITED_BY*1..5]->(implementation:Class)
        WHERE interface.type = 'interface' OR interface.is_abstract = true
        
        WITH path, implementation
        MATCH (implementation)-[:DEFINES]->(method:Method)
        
        RETURN 
            path,
            implementation,
            collect(method) as methods,
            length(path) as depth,
            implementation.vector_db_id as vector_ref
        ORDER BY depth
        """
        
        results = await self.integrator.neo4j.execute_query(
            cypher,
            interface_name=interface_name,
            repo=repository_name
        )
        
        # Enhance with vector context for each implementation
        enhanced_results = []
        for record in results:
            implementation = record['implementation']
            
            # Get semantic context from vector DB
            if implementation.get('vector_db_id'):
                vector_context = await self.integrator.vector_db.get_similar(
                    implementation['vector_db_id'], 
                    limit=5
                )
                
                enhanced_results.append({
                    'path': record['path'],
                    'implementation': implementation,
                    'methods': record['methods'],
                    'depth': record['depth'],
                    'semantic_context': vector_context
                })
        
        return enhanced_results
    
    async def analyze_dependency_impact(
        self, 
        component_id: str,
        impact_depth: int = 3
    ) -> Dict[str, Any]:
        """Analyze the impact of changes to a component."""
        
        # Multi-directional graph traversal
        cypher = """
        MATCH (component {unified_id: $id})
        
        // Find upstream dependencies (what this depends on)
        OPTIONAL MATCH upstream_path = (component)-[:DEPENDS_ON|IMPORTS*1..$depth]->(upstream)
        
        // Find downstream dependencies (what depends on this)  
        OPTIONAL MATCH downstream_path = (component)<-[:DEPENDS_ON|IMPORTS*1..$depth]-(downstream)
        
        // Find related components through similar functionality
        OPTIONAL MATCH (component)-[:SIMILAR_TO]-(similar)
        
        RETURN 
            component,
            collect(DISTINCT upstream) as upstream_deps,
            collect(DISTINCT downstream) as downstream_deps,
            collect(DISTINCT similar) as similar_components,
            count(DISTINCT downstream) as impact_score
        """
        
        result = await self.integrator.neo4j.execute_query(
            cypher,
            id=component_id,
            depth=impact_depth
        )
        
        # Enhance with semantic analysis
        impact_analysis = result[0]
        
        # Analyze semantic similarity of impacted components
        if impact_analysis['downstream_deps']:
            semantic_clusters = await self._cluster_by_semantic_similarity(
                impact_analysis['downstream_deps']
            )
            impact_analysis['semantic_clusters'] = semantic_clusters
        
        return impact_analysis
```

## Performance Optimization Patterns

### 5. Efficient Batch Processing

```python
class BatchProcessor:
    """Optimized batch processing for large repository updates."""
    
    def __init__(self, rag_kg_integrator: RAGKGIntegrator):
        self.integrator = rag_kg_integrator
        self.batch_size = 100
        self.concurrency_limit = 10
    
    async def process_repository_batch(
        self, 
        records: List[UnifiedDataRecord]
    ) -> Dict[str, Any]:
        """Process large batches of repository data efficiently."""
        
        # Group by processing type
        files = [r for r in records if r.metadata.get('type') == 'file']
        classes = [r for r in records if r.metadata.get('type') == 'class']
        functions = [r for r in records if r.metadata.get('type') == 'function']
        
        # Process in parallel batches
        tasks = [
            self._process_file_batch(files),
            self._process_class_batch(classes),
            self._process_function_batch(functions)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Post-process: Create relationships and similarity links
        await self._create_batch_relationships(records)
        await self._compute_semantic_similarities(records)
        
        return {
            'files_processed': len(files),
            'classes_processed': len(classes),
            'functions_processed': len(functions),
            'total_processing_time': results[0]['processing_time']
        }
    
    async def _compute_semantic_similarities(self, records: List[UnifiedDataRecord]):
        """Compute and store semantic similarities between code elements."""
        
        # Extract embeddings for similarity computation
        embeddings_map = {
            record.id: record.embedding_vector 
            for record in records 
            if record.embedding_vector
        }
        
        # Compute similarity matrix (optimized for large batches)
        similarities = await self._compute_similarity_matrix(embeddings_map)
        
        # Store high-similarity relationships in Neo4j
        similarity_relationships = []
        threshold = 0.8
        
        for (id1, id2), score in similarities.items():
            if score >= threshold:
                similarity_relationships.append({
                    'from_id': id1,
                    'to_id': id2,
                    'similarity_score': score,
                    'relationship_type': 'SEMANTICALLY_SIMILAR'
                })
        
        # Batch insert relationships
        if similarity_relationships:
            await self._batch_create_relationships(similarity_relationships)
```

### 6. Caching and Optimization

```python
class CacheOptimizedRAGKG:
    """RAG-KG system with intelligent caching."""
    
    def __init__(self, rag_kg_integrator: RAGKGIntegrator):
        self.integrator = rag_kg_integrator
        self.query_cache = {}  # In-memory query cache
        self.embedding_cache = {}  # Embedding cache
        self.graph_cache = {}  # Graph traversal cache
    
    async def cached_hybrid_search(
        self, 
        query: str, 
        cache_ttl: int = 3600
    ) -> List[Dict[str, Any]]:
        """Hybrid search with intelligent caching."""
        
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        
        # Check cache first
        if query_hash in self.query_cache:
            cache_entry = self.query_cache[query_hash]
            if (datetime.now() - cache_entry['timestamp']).seconds < cache_ttl:
                return cache_entry['results']
        
        # Perform fresh search
        results = await self.integrator.hybrid_search(query)
        
        # Cache results
        self.query_cache[query_hash] = {
            'results': results,
            'timestamp': datetime.now()
        }
        
        return results
```

## Integration Patterns for Repository Processing

### 7. Repository Analysis Pipeline

```python
class UnifiedRepositoryProcessor:
    """Unified pipeline for processing code repositories into RAG-KG systems."""
    
    def __init__(self):
        self.rag_kg = RAGKGIntegrator(neo4j_driver, vector_client)
        self.batch_processor = BatchProcessor(self.rag_kg)
        self.reasoning_engine = GraphReasoningEngine(self.rag_kg)
    
    async def process_repository(
        self, 
        repo_url: str, 
        clone_path: str
    ) -> Dict[str, Any]:
        """Complete repository processing pipeline."""
        
        # Phase 1: Repository discovery and cloning
        repo_info = await self._clone_and_analyze_repo(repo_url, clone_path)
        
        # Phase 2: File discovery and language detection  
        file_records = await self._discover_and_classify_files(clone_path)
        
        # Phase 3: Code parsing and analysis (Tree-sitter + AST)
        code_records = await self._parse_code_elements(file_records)
        
        # Phase 4: Content chunking for vector storage
        chunk_records = await self._create_semantic_chunks(code_records)
        
        # Phase 5: Unified storage in both systems
        await self.batch_processor.process_repository_batch(
            code_records + chunk_records
        )
        
        # Phase 6: Relationship extraction and graph building
        await self._extract_code_relationships(code_records)
        
        # Phase 7: Semantic similarity computation
        await self._compute_cross_language_similarities(code_records)
        
        return {
            'repository': repo_info,
            'files_processed': len(file_records),
            'code_elements': len(code_records),
            'chunks_created': len(chunk_records),
            'processing_complete': True
        }
```

## Common Patterns and Best Practices

### 8. Error Handling and Resilience

```python
class ResilientRAGKG:
    """RAG-KG with comprehensive error handling."""
    
    async def safe_unified_operation(self, operation_func, *args, **kwargs):
        """Execute operations with automatic retry and fallback."""
        
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                return await operation_func(*args, **kwargs)
                
            except Neo4jError as e:
                if attempt == max_retries - 1:
                    # Final attempt: log error and use vector-only mode
                    logger.error(f"Neo4j operation failed: {e}")
                    return await self._fallback_to_vector_only(*args, **kwargs)
                await asyncio.sleep(retry_delay * (2 ** attempt))
                
            except VectorDBError as e:
                if attempt == max_retries - 1:
                    # Final attempt: log error and use graph-only mode
                    logger.error(f"Vector DB operation failed: {e}")
                    return await self._fallback_to_graph_only(*args, **kwargs)
                await asyncio.sleep(retry_delay * (2 ** attempt))
```

This integration guide provides the essential patterns for building robust RAG-KG systems that combine the strengths of both graph databases and vector stores for comprehensive code repository analysis and retrieval.