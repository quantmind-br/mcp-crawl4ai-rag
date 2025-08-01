# RAG Strategies and Configuration

## Available RAG Enhancement Strategies

The project supports 5 configurable RAG strategies via environment variables:

### 1. USE_CONTEXTUAL_EMBEDDINGS
- **Purpose**: Enhances chunk embeddings with document context via LLM
- **When to Enable**: High-precision retrieval where context matters
- **Trade-offs**: Slower indexing, higher API costs, better accuracy
- **Configuration**: Set `USE_CONTEXTUAL_EMBEDDINGS=true`

### 2. USE_HYBRID_SEARCH  
- **Purpose**: Combines vector similarity with keyword search
- **When to Enable**: Technical content with specific terms/function names
- **Trade-offs**: Slightly slower queries, more robust results
- **Configuration**: Set `USE_HYBRID_SEARCH=true`

### 3. USE_AGENTIC_RAG
- **Purpose**: Specialized code example extraction and search
- **When to Enable**: AI coding assistants needing code snippets
- **Trade-offs**: Much slower crawling, requires more storage
- **Configuration**: Set `USE_AGENTIC_RAG=true`
- **Additional Tool**: Enables `search_code_examples` MCP tool

### 4. USE_RERANKING
- **Purpose**: Cross-encoder reranking for improved result relevance
- **When to Enable**: When search precision is critical
- **Trade-offs**: +100-200ms per query, significantly better ordering
- **Configuration**: Set `USE_RERANKING=true`
- **Model**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2`

### 5. USE_KNOWLEDGE_GRAPH
- **Purpose**: AI hallucination detection via Neo4j graph analysis
- **When to Enable**: Code validation and hallucination detection needed
- **Trade-offs**: Requires Neo4j, not fully Docker-compatible yet
- **Configuration**: Set `USE_KNOWLEDGE_GRAPH=true` + Neo4j credentials
- **Additional Tools**: 
  - `parse_github_repository`
  - `check_ai_script_hallucinations`
  - `query_knowledge_graph`

## Recommended Configuration Presets

### General Documentation RAG
```env
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=true
USE_KNOWLEDGE_GRAPH=false
```

### AI Coding Assistant (Full Features)
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

## LLM Configuration
- **MODEL_CHOICE**: LLM for summaries and contextual embeddings (e.g., `gpt-4o-mini`)
- **Embedding Model**: Fixed to OpenAI `text-embedding-3-small`
- **Reranking**: Local cross-encoder model (no API calls)