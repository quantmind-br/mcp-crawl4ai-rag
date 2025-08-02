# RAG Strategies and Configuration

## Available RAG Strategies
The project supports five advanced RAG strategies that can be enabled independently via environment variables.

### 1. Contextual Embeddings (`USE_CONTEXTUAL_EMBEDDINGS`)
- **Purpose**: Enhance chunk embeddings with document context
- **Process**: Passes full document + chunk to LLM for enriched context
- **When to use**: High-precision retrieval where context matters
- **Trade-offs**: Slower indexing, additional LLM API calls, better accuracy

### 2. Hybrid Search (`USE_HYBRID_SEARCH`)
- **Purpose**: Combine keyword search with semantic vector search
- **Process**: Parallel searches with intelligent result merging
- **When to use**: Technical content with specific terms and function names
- **Trade-offs**: Slightly slower queries, more robust results

### 3. Agentic RAG (`USE_AGENTIC_RAG`)
- **Purpose**: Specialized code example extraction and storage
- **Process**: Identifies code blocks, extracts with context, generates summaries
- **When to use**: AI coding assistants needing specific code examples
- **Trade-offs**: Much slower crawling, more storage, dedicated search tool
- **Provides**: `search_code_examples` tool for targeted code retrieval

### 4. Reranking (`USE_RERANKING`)
- **Purpose**: Improve result relevance using cross-encoder models
- **Process**: Re-scores search results against original query
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **When to use**: When search precision is critical
- **Trade-offs**: +100-200ms per query, significantly better ranking

### 5. Knowledge Graph (`USE_KNOWLEDGE_GRAPH`)
- **Purpose**: AI hallucination detection via Neo4j code analysis
- **Process**: Parse repositories into graph, validate AI code against structure
- **When to use**: Validate AI-generated code against real implementations
- **Trade-offs**: Requires Neo4j setup, slow for large codebases
- **Provides**: Repository parsing, hallucination detection, graph querying

## GPU Acceleration
When reranking is enabled, GPU acceleration can significantly improve performance:

```env
USE_GPU_ACCELERATION=auto      # auto, true/cuda, mps, false/cpu
GPU_PRECISION=float32          # float32, float16, bfloat16
GPU_DEVICE_INDEX=0            # GPU index for multi-GPU systems
GPU_MEMORY_FRACTION=0.8       # Fraction of GPU memory to use
```

**Performance Benefits**:
- 5-10x speedup on GPU-enabled systems
- Automatic fallback to CPU when GPU unavailable
- Configurable memory management

## Recommended Configurations

### General Documentation RAG
```env
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=true
USE_KNOWLEDGE_GRAPH=false
```

### AI Coding Assistant (with code examples)
```env
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
USE_KNOWLEDGE_GRAPH=false
```

### AI Coding Assistant (with hallucination detection)
```env
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
USE_KNOWLEDGE_GRAPH=true
```

### Fast Basic RAG
```env
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=false
USE_KNOWLEDGE_GRAPH=false
```

## API Configuration
The project supports flexible API configuration for different providers:

### OpenAI (Default)
```env
CHAT_MODEL=gpt-4o-mini
CHAT_API_KEY=sk-your-openai-key
EMBEDDINGS_MODEL=text-embedding-3-small
EMBEDDINGS_API_KEY=sk-your-openai-key
```

### Azure OpenAI
```env
CHAT_MODEL=gpt-35-turbo
CHAT_API_KEY=your-azure-api-key
CHAT_API_BASE=https://your-resource.openai.azure.com/
EMBEDDINGS_MODEL=text-embedding-ada-002
EMBEDDINGS_API_KEY=your-azure-api-key
EMBEDDINGS_API_BASE=https://your-resource.openai.azure.com/
```

### Mixed Providers
```env
# Chat via Azure OpenAI
CHAT_MODEL=gpt-4
CHAT_API_KEY=your-azure-key
CHAT_API_BASE=https://your-resource.openai.azure.com/

# Embeddings via regular OpenAI
EMBEDDINGS_MODEL=text-embedding-3-small
EMBEDDINGS_API_KEY=sk-your-openai-key
```

## Database Configuration
```env
# Qdrant Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Neo4j Knowledge Graph (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
```