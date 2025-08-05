# RAG Strategies and Configuration

## Available RAG Strategies

### Hybrid Search (`USE_HYBRID_SEARCH=true`)
- **Purpose**: Combines semantic (dense) + keyword (sparse) search using FastBM25
- **Implementation**: Native Qdrant with Reciprocal Rank Fusion (RRF)
- **Performance**: <200ms search, +20-40% accuracy improvement
- **Cost**: None (computational only)
- **Migration**: Automatic collection migration available

### Reranking (`USE_RERANKING=true`)
- **Purpose**: Re-order search results using cross-encoder models
- **Performance**: +100-200ms per query, +10-15% accuracy improvement
- **Models**: cross-encoder/ms-marco-MiniLM-L-6-v2 (default), customizable
- **GPU Support**: Optional GPU acceleration with automatic fallback
- **Cost**: None (computational only)

### Contextual Embeddings (`USE_CONTEXTUAL_EMBEDDINGS=true`)
- **Purpose**: Enhance chunks with document context for better accuracy
- **Performance**: +30% token usage, +15-25% accuracy improvement
- **Implementation**: LLM-enhanced chunk context before embedding
- **Cost**: Additional API calls for context generation

### Agentic RAG (`USE_AGENTIC_RAG=true`)
- **Purpose**: Extract and index code examples separately
- **Use Case**: Programming/technical documentation with dedicated code search
- **Performance**: Much slower crawling due to additional processing
- **Cost**: Additional API calls for code analysis

### Knowledge Graph (`USE_KNOWLEDGE_GRAPH=true`)
- **Purpose**: AI hallucination detection using Neo4j knowledge graphs
- **Implementation**: Code structure analysis and validation
- **Requirements**: Neo4j database and repository parsing
- **Use Case**: Validate AI-generated code against indexed repositories

## Recommended Configurations

### General Documentation RAG
```bash
USE_HYBRID_SEARCH=true
USE_RERANKING=true
USE_CONTEXTUAL_EMBEDDINGS=false
USE_AGENTIC_RAG=false
USE_KNOWLEDGE_GRAPH=false
```

### AI Coding Assistant
```bash
USE_HYBRID_SEARCH=true
USE_RERANKING=true
USE_CONTEXTUAL_EMBEDDINGS=true
USE_AGENTIC_RAG=true
USE_KNOWLEDGE_GRAPH=false
```

### Full AI Assistant with Validation
```bash
USE_HYBRID_SEARCH=true
USE_RERANKING=true
USE_CONTEXTUAL_EMBEDDINGS=true
USE_AGENTIC_RAG=true
USE_KNOWLEDGE_GRAPH=true
```

## Advanced Configuration

### Hybrid Search Settings
- **RRF_K=60**: Reciprocal Rank Fusion parameter (higher = less rank impact)
- **DENSE_WEIGHT=0.5**: Semantic vs keyword weight (0.7 = 70% semantic)
- **AUTO_MIGRATE_COLLECTIONS=false**: Enable automatic collection migration

### Reranking Settings
- **RERANKING_MODEL_NAME**: CrossEncoder model selection
- **RERANKING_WARMUP_SAMPLES=5**: Startup warmup predictions
- **USE_GPU_ACCELERATION=auto**: GPU usage for reranking

### GPU Configuration
- **GPU_PRECISION=float32**: Model precision (float32/float16/bfloat16)
- **GPU_DEVICE_INDEX=0**: GPU device selection
- **GPU_MEMORY_FRACTION=0.8**: Memory usage fraction

## Performance Impact Summary
| Strategy | Setup Time | Query Time | Accuracy Gain | API Cost |
|----------|------------|------------|---------------|----------|
| Hybrid Search | +5min | +50ms | +20-40% | None |
| Reranking | +30s | +100-200ms | +10-15% | None |
| Contextual | None | Same | +15-25% | +30% tokens |
| Agentic | None | Same | Variable | +50% tokens |
| Knowledge Graph | +10min | +200ms | Variable | None |