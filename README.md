# Crawl4AI RAG MCP Server

**Advanced Web Crawling and RAG Capabilities for AI Agents**

A powerful Model Context Protocol (MCP) server that integrates [Crawl4AI](https://crawl4ai.com), [Qdrant](https://qdrant.tech/), and [Neo4j](https://neo4j.com/) to provide AI agents and coding assistants with intelligent web crawling, GitHub repository indexing, vector search, and AI hallucination detection capabilities.

![MCP](https://img.shields.io/badge/MCP-Compatible-blue)
![Python](https://img.shields.io/badge/Python-3.12+-green)
![Docker](https://img.shields.io/badge/Docker-Required-blue)

---

## 🚀 Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) 
- [Python 3.12+](https://www.python.org/downloads/)
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key (or compatible provider)

### Installation

1. **Clone and setup**:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   uv sync
   ```

2. **Start services**:
   ```bash
   # Windows
   setup.bat
   
   # Linux/Mac
   docker-compose up -d
   ```

3. **Configure environment**:
   ```bash
   # Create .env file with your API keys
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start the server**:
   ```bash
   # Windows
   start.bat
   
   # Linux/Mac
   uv run -m src
   ```

---

## 🛠️ MCP Tools

### 🌐 Web Tools
- **`crawl_single_page`** - Crawl and index individual webpages
  - **Parameters**: `url` (required)
  - **Timeout**: 30 minutes (LONG_TIMEOUT)
  
- **`smart_crawl_url`** - Intelligent crawling (auto-detects sitemaps, recursive crawling)
  - **Parameters**: `url` (required), `max_depth` (optional), `max_concurrent` (optional), `chunk_size` (optional)
  - **Timeout**: 1 hour (VERY_LONG_TIMEOUT)

### 🔍 RAG Tools
- **`get_available_sources`** - List all indexed sources
  - **Parameters**: None
  - **Timeout**: 1 minute (QUICK_TIMEOUT)
  
- **`perform_rag_query`** - Semantic search with source filtering
  - **Parameters**: `query` (required), `source` (optional), `match_count` (optional), `file_id` (optional)
  - **Timeout**: 5 minutes (MEDIUM_TIMEOUT)

### 🐙 GitHub Tools
- **`index_github_repository`** - Unified GitHub repository indexing (replaces `smart_crawl_github`)
  - **Parameters**: `repo_url` (required), `destination` (optional), `file_types` (optional), `max_files` (optional), `chunk_size` (optional), `max_size_mb` (optional), `enable_intelligent_routing` (optional), `force_rag_patterns` (optional), `force_kg_patterns` (optional)
  - **Timeout**: 1 hour (VERY_LONG_TIMEOUT)
  - **Features**: Dual-system processing (Qdrant + Neo4j), multi-language support, enterprise-scale (10,000 files default)

### 🧠 Knowledge Graph Tools (Optional)
- **`search_code_examples`** - Specialized code search (requires `USE_AGENTIC_RAG=true`)
  - **Parameters**: `query` (required), `source_id` (optional), `match_count` (optional), `file_id` (optional)
  - **Timeout**: 5 minutes (MEDIUM_TIMEOUT)
  
- **`check_ai_script_hallucinations`** - Validate AI-generated code
  - **Parameters**: `script_path` (required)
  - **Timeout**: 5 minutes (MEDIUM_TIMEOUT)
  
- **`query_knowledge_graph`** - Explore repository structure
  - **Parameters**: `command` (required)
  - **Timeout**: 1 minute (QUICK_TIMEOUT)
  - **Commands**: `repos`, `explore <repo>`, `classes [repo]`, `class <name>`, `method <name> [class]`, `query <cypher>`

### 🔄 Tool Migration Guide

#### GitHub Repository Indexing

The `parse_github_repository` tool has been deprecated in favor of the more comprehensive `index_github_repository` tool. For equivalent functionality:

```python
# Old approach (deprecated)
# parse_github_repository("https://github.com/user/repo")

# New approach - Neo4j only indexing
index_github_repository(
    repo_url="https://github.com/user/repo",
    destination="neo4j",        # Equivalent to parse_github_repository
    file_types=[".py"],         # Optional: specify file types
    max_files=1000             # Optional: control scale
)

# Enhanced approach - Dual system indexing
index_github_repository(
    repo_url="https://github.com/user/repo", 
    destination="both",                    # Both Qdrant and Neo4j
    file_types=[".py", ".md", ".js"],     # Multiple languages
    max_files=5000,                       # Enterprise scale
    enable_intelligent_routing=True       # Smart file classification
)
```

The unified tool provides all the same Neo4j functionality plus additional features like intelligent routing, cross-system linking, and enterprise-scale processing.

### 📋 MCP Tools JSON Specification

Complete JSON specification sent to LLM clients:

```json
{
  "tools": [
    {
      "name": "crawl_single_page",
      "description": "Crawl and index individual webpages into Qdrant vector database",
      "inputSchema": {
        "type": "object",
        "properties": {
          "url": {
            "type": "string",
            "description": "URL of the webpage to crawl"
          }
        },
        "required": ["url"]
      },
      "timeout": "LONG_TIMEOUT (1800s)"
    },
    {
      "name": "smart_crawl_url",
      "description": "Intelligent crawling with auto-detection of sitemaps and recursive crawling",
      "inputSchema": {
        "type": "object",
        "properties": {
          "url": {
            "type": "string",
            "description": "URL to crawl (can be sitemap, webpage, or repository)"
          },
          "max_depth": {
            "type": "integer",
            "description": "Maximum depth for recursive crawling (default: 3)"
          },
          "max_concurrent": {
            "type": "integer", 
            "description": "Maximum concurrent requests (default: 5)"
          },
          "chunk_size": {
            "type": "integer",
            "description": "Chunk size for text processing (default: 1000)"
          }
        },
        "required": ["url"]
      },
      "timeout": "VERY_LONG_TIMEOUT (3600s)"
    },
    {
      "name": "get_available_sources",
      "description": "List all indexed sources in the vector database",
      "inputSchema": {
        "type": "object",
        "properties": {}
      },
      "timeout": "QUICK_TIMEOUT (60s)"
    },
    {
      "name": "perform_rag_query",
      "description": "Perform semantic search with optional source filtering",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "Search query text"
          },
          "source": {
            "type": "string",
            "description": "Optional source filter (domain or identifier)"
          },
          "match_count": {
            "type": "integer",
            "description": "Number of results to return (default: 10)"
          },
          "file_id": {
            "type": "string",
            "description": "Optional specific file ID to search within"
          }
        },
        "required": ["query"]
      },
      "timeout": "MEDIUM_TIMEOUT (300s)"
    },
    {
      "name": "search_code_examples",
      "description": "Specialized code search (requires USE_AGENTIC_RAG=true)",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "Code search query"
          },
          "source_id": {
            "type": "string",
            "description": "Optional source identifier"
          },
          "match_count": {
            "type": "integer",
            "description": "Number of results to return (default: 10)"
          },
          "file_id": {
            "type": "string",
            "description": "Optional specific file ID to search within"
          }
        },
        "required": ["query"]
      },
      "timeout": "MEDIUM_TIMEOUT (300s)"
    },
    {
      "name": "index_github_repository",
      "description": "Unified GitHub repository indexing for both Qdrant and Neo4j",
      "inputSchema": {
        "type": "object",
        "properties": {
          "repo_url": {
            "type": "string",
            "description": "GitHub repository URL (https://github.com/user/repo.git)"
          },
          "destination": {
            "type": "string",
            "enum": ["both", "qdrant", "neo4j"],
            "description": "Indexing destination (default: both)"
          },
          "file_types": {
            "type": "array",
            "items": {"type": "string"},
            "description": "File extensions to process (default: ['.py', '.js', '.ts', '.md', '.txt'])"
          },
          "max_files": {
            "type": "integer",
            "description": "Maximum files to process (default: 10000)"
          },
          "chunk_size": {
            "type": "integer",
            "description": "Text chunk size (default: 1000)"
          },
          "max_size_mb": {
            "type": "integer",
            "description": "Maximum repository size in MB (default: 500)"
          },
          "enable_intelligent_routing": {
            "type": "boolean",
            "description": "Enable intelligent content routing (default: true)"
          },
          "force_rag_patterns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Force specific patterns to RAG processing"
          },
          "force_kg_patterns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Force specific patterns to Knowledge Graph processing"
          }
        },
        "required": ["repo_url"]
      },
      "timeout": "VERY_LONG_TIMEOUT (3600s)"
    },
    {
      "name": "check_ai_script_hallucinations",
      "description": "Validate AI-generated code against indexed repositories",
      "inputSchema": {
        "type": "object",
        "properties": {
          "script_path": {
            "type": "string",
            "description": "Path to the script file to validate"
          }
        },
        "required": ["script_path"]
      },
      "timeout": "MEDIUM_TIMEOUT (300s)"
    },
    {
      "name": "query_knowledge_graph",
      "description": "Explore repository structure using natural language or Cypher queries",
      "inputSchema": {
        "type": "object",
        "properties": {
          "command": {
            "type": "string",
            "description": "Command: 'repos', 'explore <repo>', 'classes [repo]', 'class <name>', 'method <name> [class]', 'query <cypher>'"
          }
        },
        "required": ["command"]
      },
      "timeout": "QUICK_TIMEOUT (60s)"
    }
  ]
}
```

---

## ⚙️ Configuration

### Complete Environment Configuration (.env)
```bash
# === MCP SERVER CONFIGURATION ===
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# === PRIMARY API CONFIGURATION ===
# Chat Model Configuration (for summaries, contextual embeddings, code analysis)
CHAT_MODEL=gpt-4o-mini
CHAT_API_KEY=your_chat_api_key
CHAT_API_BASE=https://api.openai.com/v1

# Embeddings Model Configuration (for vector search and semantic similarity)
EMBEDDINGS_MODEL=text-embedding-3-small
EMBEDDINGS_API_KEY=your_embeddings_api_key
EMBEDDINGS_API_BASE=https://api.openai.com/v1
EMBEDDINGS_DIMENSIONS=1536  # Optional: override auto-detection

# === FALLBACK API CONFIGURATION ===
# Fallback configuration for high availability
CHAT_FALLBACK_MODEL=gpt-4o-mini
CHAT_FALLBACK_API_KEY=your_fallback_chat_key
CHAT_FALLBACK_API_BASE=https://api.openai.com/v1

EMBEDDINGS_FALLBACK_MODEL=text-embedding-3-small
EMBEDDINGS_FALLBACK_API_KEY=your_fallback_embeddings_key
EMBEDDINGS_FALLBACK_API_BASE=https://api.openai.com/v1

# === VECTOR DATABASE ===
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# === KNOWLEDGE GRAPH ===
# Neo4j Configuration (required for knowledge graph functionality)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# === RAG STRATEGIES ===
# Set to "true" or "false", all default to "false"
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=true
USE_KNOWLEDGE_GRAPH=false

# === GPU ACCELERATION ===
# GPU Configuration (optional - requires USE_RERANKING=true)
USE_GPU_ACCELERATION=auto      # auto, true/cuda, mps, false/cpu
GPU_PRECISION=float32          # float32, float16, bfloat16
GPU_DEVICE_INDEX=0            # GPU index for multi-GPU systems
GPU_MEMORY_FRACTION=0.8       # Fraction of GPU memory to use

```

### Multi-Provider Examples

**OpenAI (Default)**:
```bash
CHAT_MODEL=gpt-4o-mini
CHAT_API_KEY=sk-your-openai-key

EMBEDDINGS_MODEL=text-embedding-3-small
EMBEDDINGS_API_KEY=sk-your-openai-key
```

**Azure OpenAI**:
```bash
CHAT_MODEL=gpt-35-turbo
CHAT_API_KEY=your-azure-api-key
CHAT_API_BASE=https://your-resource.openai.azure.com/

EMBEDDINGS_MODEL=text-embedding-ada-002
EMBEDDINGS_API_KEY=your-azure-api-key
EMBEDDINGS_API_BASE=https://your-resource.openai.azure.com/
```

**DeepInfra (Cost-Effective)**:
```bash
CHAT_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
CHAT_API_KEY=your-deepinfra-key
CHAT_API_BASE=https://api.deepinfra.com/v1/openai

EMBEDDINGS_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDINGS_API_KEY=your-deepinfra-key
EMBEDDINGS_API_BASE=https://api.deepinfra.com/v1/openai
```

**Mixed Providers with Fallback**:
```bash
# Primary: DeepInfra for cost efficiency
CHAT_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
CHAT_API_KEY=your-deepinfra-key
CHAT_API_BASE=https://api.deepinfra.com/v1/openai

# Fallback: OpenAI for reliability
CHAT_FALLBACK_MODEL=gpt-4o-mini
CHAT_FALLBACK_API_KEY=sk-your-openai-key

EMBEDDINGS_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDINGS_API_KEY=your-deepinfra-key
EMBEDDINGS_API_BASE=https://api.deepinfra.com/v1/openai

EMBEDDINGS_FALLBACK_MODEL=text-embedding-3-small
EMBEDDINGS_FALLBACK_API_KEY=sk-your-openai-key

# === MCP TOOLS TIMEOUT CONFIGURATION ===
# MCP Tools Timeout Configuration (seconds)
MCP_QUICK_TIMEOUT=60        # Quick operations: Simple queries, data retrieval, status checks
MCP_MEDIUM_TIMEOUT=300      # Medium operations: RAG queries, analysis tasks, script validation
MCP_LONG_TIMEOUT=1800       # Long operations: Single page crawls, repository parsing, complex analysis
MCP_VERY_LONG_TIMEOUT=3600  # Very long operations: Multi-page crawls, full repository indexing, bulk processing

# === PERFORMANCE OPTIMIZATION (Optional) ===
# ProcessPoolExecutor workers for CPU-bound parsing tasks
CPU_WORKERS=4

# ThreadPoolExecutor workers for I/O-bound operations  
IO_WORKERS=10

# Batch size for Qdrant vector insertion operations
BATCH_SIZE_QDRANT=500

# Batch size for Neo4j UNWIND operations
BATCH_SIZE_NEO4J=5000

# Batch size for embeddings API calls
BATCH_SIZE_EMBEDDINGS=1000

# Batch size for concurrent file processing
BATCH_SIZE_FILE_PROCESSING=10

# Maximum concurrent parsing operations
MAX_CONCURRENT_PARSING=8

# Chunk size for embedding generation
EMBEDDING_CHUNK_SIZE=100

# === HTTP OPTIMIZATION (Optional) ===
# HTTP/2 support for better API performance
HTTPX_HTTP2=true

# HTTP connection limits
HTTPCORE_MAX_CONNECTIONS=100
HTTPCORE_KEEPALIVE_EXPIRY=30

# === SYSTEM CONFIGURATION (Optional) ===
# Logging level (DEBUG for development, INFO for production)
LOG_LEVEL=INFO
```

## 🕐 MCP Tools Timeout Configuration

### Timeout Categories

| Category | Default Timeout | Use Cases | Tools Using This Timeout |
|----------|----------------|-----------|--------------------------|
| **QUICK_TIMEOUT** | 60 seconds (1 minute) | Simple queries, data retrieval, status checks | `get_available_sources`, `query_knowledge_graph` |
| **MEDIUM_TIMEOUT** | 300 seconds (5 minutes) | RAG queries, analysis tasks, script validation | `perform_rag_query`, `search_code_examples`, `check_ai_script_hallucinations` |
| **LONG_TIMEOUT** | 1800 seconds (30 minutes) | Single page crawls, repository parsing, complex analysis | `crawl_single_page` |
| **VERY_LONG_TIMEOUT** | 3600 seconds (1 hour) | Multi-page crawls, full repository indexing, bulk processing | `smart_crawl_url`, `index_github_repository` |

### Performance Tuning Guidelines

#### For Large-Scale Operations
```bash
# Increase timeouts for enterprise repositories
MCP_LONG_TIMEOUT=3600      # 1 hour for complex repositories
MCP_VERY_LONG_TIMEOUT=7200 # 2 hours for massive repositories (10,000+ files)
```

#### For Quick Development Cycles
```bash
# Reduce timeouts for faster failure detection
MCP_QUICK_TIMEOUT=30       # 30 seconds for quick operations
MCP_MEDIUM_TIMEOUT=120     # 2 minutes for medium operations
```

#### Troubleshooting Timeout Issues

**Symptoms of Timeout Problems:**
- MCP client disconnections during large crawls
- "Operation timed out" errors
- Incomplete indexing results

**Solutions:**
1. **Increase specific timeout**: Adjust the relevant timeout category
2. **Monitor resource usage**: Check CPU/memory during operations
3. **Optimize input size**: Limit `max_files` or `max_depth` parameters
4. **Use progress tracking**: Enable verbose logging to monitor progress

**Implementation Status:**
- Infrastructure prepared for timeout configuration
- Environment variables active for runtime configuration
- Future FastMCP versions will support tool-level timeout enforcement

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| **Server Configuration** |
| `HOST` | `0.0.0.0` | Server host address |
| `PORT` | `8051` | Server port |
| `TRANSPORT` | `sse` | Transport protocol (sse/stdio) |
| **Primary API Configuration** |
| `CHAT_MODEL` | - | Chat model (gpt-4o-mini, etc.) |
| `CHAT_API_KEY` | - | API key for chat operations |
| `CHAT_API_BASE` | OpenAI default | Base URL for chat API |
| `EMBEDDINGS_MODEL` | - | Embeddings model |
| `EMBEDDINGS_API_KEY` | - | API key for embeddings |
| `EMBEDDINGS_API_BASE` | OpenAI default | Base URL for embeddings API |
| `EMBEDDINGS_DIMENSIONS` | Auto-detect | Override dimension auto-detection |
| **Fallback Configuration** |
| `CHAT_FALLBACK_MODEL` | `gpt-4o-mini` | Fallback chat model |
| `CHAT_FALLBACK_API_KEY` | - | Fallback chat API key |
| `CHAT_FALLBACK_API_BASE` | - | Fallback chat base URL |
| `EMBEDDINGS_FALLBACK_MODEL` | `text-embedding-3-small` | Fallback embeddings model |
| `EMBEDDINGS_FALLBACK_API_KEY` | - | Fallback embeddings API key |  
| `EMBEDDINGS_FALLBACK_API_BASE` | - | Fallback embeddings base URL |
| **Database Configuration** |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |  
| `NEO4J_URI` | - | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | - | Neo4j password |
| **RAG Strategies** |
| `USE_CONTEXTUAL_EMBEDDINGS` | `false` | Enhanced chunk context |
| `USE_HYBRID_SEARCH` | `false` | Semantic + keyword search |
| `USE_AGENTIC_RAG` | `false` | Code example extraction |
| `USE_RERANKING` | `false` | Result reordering |
| `USE_KNOWLEDGE_GRAPH` | `false` | AI hallucination detection |
| **GPU Acceleration** |
| `USE_GPU_ACCELERATION` | `auto` | GPU usage (auto/cuda/mps/cpu) |
| `GPU_PRECISION` | `float32` | Model precision (float32/float16/bfloat16) |
| `GPU_DEVICE_INDEX` | `0` | GPU device index |
| `GPU_MEMORY_FRACTION` | `0.8` | GPU memory fraction to use |
| **MCP Tools Timeout Configuration** |
| `MCP_QUICK_TIMEOUT` | `60` | Quick operations timeout (seconds) |
| `MCP_MEDIUM_TIMEOUT` | `300` | Medium operations timeout (seconds) |
| `MCP_LONG_TIMEOUT` | `1800` | Long operations timeout (seconds) |
| `MCP_VERY_LONG_TIMEOUT` | `3600` | Very long operations timeout (seconds) |
| **Performance Optimization** |
| `CPU_WORKERS` | `4` | ProcessPoolExecutor workers for CPU-bound tasks |
| `IO_WORKERS` | `10` | ThreadPoolExecutor workers for I/O-bound operations |
| `BATCH_SIZE_QDRANT` | `500` | Batch size for Qdrant vector insertion |
| `BATCH_SIZE_NEO4J` | `5000` | Batch size for Neo4j UNWIND operations |
| `BATCH_SIZE_EMBEDDINGS` | `1000` | Batch size for embeddings API calls |
| `BATCH_SIZE_FILE_PROCESSING` | `10` | Batch size for concurrent file processing |
| `MAX_CONCURRENT_PARSING` | `8` | Maximum concurrent parsing operations |
| `EMBEDDING_CHUNK_SIZE` | `100` | Chunk size for embedding generation |
| **HTTP Optimization** |
| `HTTPX_HTTP2` | `true` | Enable HTTP/2 support for better performance |
| `HTTPCORE_MAX_CONNECTIONS` | `100` | HTTP connection limits |
| `HTTPCORE_KEEPALIVE_EXPIRY` | `30` | HTTP keepalive expiry (seconds) |
| **System Configuration** |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |

---

## 🔌 MCP Client Integration

### Claude Desktop / Windsurf (SSE)
```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "sse",
      "url": "http://localhost:8051/sse"
    }
  }
}
```

### Claude Code CLI
```bash
claude mcp add-json crawl4ai-rag '{"type":"http","url":"http://localhost:8051/sse"}' --scope user
```

### Stdio Configuration
```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "uv",
      "args": ["run", "-m", "src"],
      "cwd": "/path/to/mcp-crawl4ai-rag",
      "env": {
        "TRANSPORT": "stdio",
        "CHAT_API_KEY": "your_key",
        "EMBEDDINGS_API_KEY": "your_key"
      }
    }
  }
}
```

---

## 🧠 RAG Strategies

### Recommended Configurations

**General Documentation**:
```bash
USE_HYBRID_SEARCH=true
USE_RERANKING=true
```

**AI Coding Assistant**:
```bash
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
```

**With Hallucination Detection**:
```bash
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
USE_KNOWLEDGE_GRAPH=true
```

### Strategy Details

| Strategy | Purpose | Cost | Performance Impact |
|----------|---------|------|-------------------|
| **Hybrid Search** | Combines semantic + keyword search | None | Minimal |
| **Reranking** | Improves result relevance | None | +100-200ms |
| **Contextual Embeddings** | Enhanced chunk context | API calls | Slower indexing |
| **Agentic RAG** | Code example extraction | API calls | Much slower crawling |
| **Knowledge Graph** | AI hallucination detection | None | Requires Neo4j |

---

## 🔧 Development

### Testing
```bash
# Run all tests
uv run pytest

# Run by category (hierarchical structure)
uv run pytest tests/unit/                      # Unit tests by module
uv run pytest tests/specialized/               # Domain-specific tests  
uv run pytest tests/infrastructure/            # Infrastructure tests
uv run pytest tests/integration/               # End-to-end tests

# Run specific modules
uv run pytest tests/unit/tools/                # MCP tools tests
uv run pytest tests/specialized/embedding/     # Embedding system tests
uv run pytest tests/specialized/knowledge_graphs/ # Knowledge graph tests
uv run pytest tests/infrastructure/storage/    # Database tests
uv run pytest tests/integration/test_integration_basic.py # Basic integration

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/unit/tools/test_web_tools.py -v
```

### Utilities
```bash
# Clean vector database
uv run python scripts/clean_qdrant.py

# Fix dimension mismatches
uv run python scripts/define_qdrant_dimensions.py

# Analyze repository for hallucinations
uv run python -m src.k_graph.analysis.hallucination_detector script.py
```

### Docker Services Management
```bash
# View logs
docker-compose logs qdrant
docker-compose logs neo4j

# Restart services
docker-compose restart

# Stop everything
docker-compose down --volumes
```

---

## 📊 Architecture

### Core Components
- **MCP Server**: FastMCP-based server with async tools
- **Vector Database**: Qdrant for semantic search and storage
- **Web Crawler**: Crawl4AI with smart URL detection and GitHub integration
- **Knowledge Graph**: Neo4j for code structure analysis
- **Device Manager**: Automatic GPU/CPU detection and fallback

### Data Flow
1. **Crawling**: URLs/GitHub repos → Crawl4AI/Git clone → Content extraction
2. **Processing**: Content → Chunking → Embeddings → Qdrant
3. **Search**: Query → Vector search → Reranking → Results
4. **Validation**: Code → AST analysis → Neo4j validation

---

## 🎯 Use Cases

### Documentation RAG
```python
# Crawl documentation site
crawl_single_page("https://docs.python.org/3/tutorial/")

# Index entire GitHub project documentation
index_github_repository("https://github.com/psf/requests")

# Search for specific topics
perform_rag_query("how to handle exceptions in Python")
```

### Code Assistant
```python
# Enable code extraction
# USE_AGENTIC_RAG=true

# Crawl API documentation
smart_crawl_url("https://api.github.com/docs")

# Index GitHub repository documentation  
index_github_repository("https://github.com/user/awesome-project")

# Find code examples
search_code_examples("GitHub API authentication")
```

### Hallucination Detection
```python
# Index a repository for knowledge graph analysis
index_github_repository("https://github.com/psf/requests.git", destination="neo4j")

# Check AI-generated code
check_ai_script_hallucinations("/path/to/script.py")
```

---

## 🤝 Contributing

This is an evolving testbed for advanced RAG capabilities. The project is being actively developed for integration with [Archon](https://github.com/coleam00/Archon).

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Run tests: `uv run pytest`
4. Submit a pull request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Crawl4AI](https://crawl4ai.com) for powerful web crawling
- [Qdrant](https://qdrant.tech/) for vector database capabilities
- [MCP](https://modelcontextprotocol.io) for AI agent integration
- [Neo4j](https://neo4j.com/) for knowledge graph functionality

---

**Built for the future of AI-powered development** 🚀