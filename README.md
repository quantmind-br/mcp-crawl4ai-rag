# Crawl4AI RAG MCP Server

**Advanced Web Crawling and RAG Capabilities for AI Agents**

A powerful Model Context Protocol (MCP) server that integrates [Crawl4AI](https://crawl4ai.com), [Qdrant](https://qdrant.tech/), and [Neo4j](https://neo4j.com/) to provide AI agents and coding assistants with intelligent web crawling, vector search, and AI hallucination detection capabilities.

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

### Core Tools
- **`crawl_single_page`** - Crawl and index individual webpages
- **`smart_crawl_url`** - Intelligent crawling (auto-detects sitemaps, recursive crawling)
- **`get_available_sources`** - List all indexed sources
- **`perform_rag_query`** - Semantic search with source filtering

### Advanced Features (Optional)
- **`search_code_examples`** - Specialized code search (requires `USE_AGENTIC_RAG=true`)
- **`parse_github_repository`** - Index GitHub repos for hallucination detection
- **`check_ai_script_hallucinations`** - Validate AI-generated code
- **`query_knowledge_graph`** - Explore repository structure

---

## ⚙️ Configuration

### Basic Setup (.env)
```bash
# API Configuration
CHAT_MODEL=gpt-4o-mini
CHAT_API_KEY=your_openai_api_key
EMBEDDINGS_MODEL=text-embedding-3-small
EMBEDDINGS_API_KEY=your_openai_api_key

# Services
QDRANT_HOST=localhost
QDRANT_PORT=6333
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# RAG Strategies (all default to false)
USE_HYBRID_SEARCH=true
USE_RERANKING=true
USE_AGENTIC_RAG=false
USE_CONTEXTUAL_EMBEDDINGS=false
USE_KNOWLEDGE_GRAPH=false
```

### Multi-Provider Support
```bash
# Mix providers - chat via Azure, embeddings via OpenAI
CHAT_MODEL=gpt-4
CHAT_API_KEY=your_azure_key
CHAT_API_BASE=https://your-resource.openai.azure.com/

EMBEDDINGS_MODEL=text-embedding-3-small
EMBEDDINGS_API_KEY=sk-your_openai_key
```

### GPU Acceleration (Optional)
```bash
USE_GPU_ACCELERATION=auto    # auto, cuda, mps, cpu
GPU_PRECISION=float16        # Reduce memory usage
```

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

# Specific test suites
uv run pytest tests/test_mcp_basic.py          # MCP functionality
uv run pytest tests/test_qdrant_wrapper.py     # Vector database
uv run pytest tests/integration_test.py        # Full integration
```

### Utilities
```bash
# Clean vector database
uv run python scripts/clean_qdrant.py

# Fix dimension mismatches
uv run python scripts/fix_qdrant_dimensions.py

# Analyze repository for hallucinations
uv run python knowledge_graphs/ai_hallucination_detector.py script.py
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
- **Web Crawler**: Crawl4AI with smart URL detection
- **Knowledge Graph**: Neo4j for code structure analysis
- **Device Manager**: Automatic GPU/CPU detection and fallback

### Data Flow
1. **Crawling**: URLs → Crawl4AI → Content extraction
2. **Processing**: Content → Chunking → Embeddings → Qdrant
3. **Search**: Query → Vector search → Reranking → Results
4. **Validation**: Code → AST analysis → Neo4j validation

---

## 🎯 Use Cases

### Documentation RAG
```python
# Crawl documentation site
crawl_single_page("https://docs.python.org/3/tutorial/")

# Search for specific topics
perform_rag_query("how to handle exceptions in Python")
```

### Code Assistant
```python
# Enable code extraction
# USE_AGENTIC_RAG=true

# Crawl API documentation
smart_crawl_url("https://api.github.com/docs")

# Find code examples
search_code_examples("GitHub API authentication")
```

### Hallucination Detection
```python
# Index a repository
parse_github_repository("https://github.com/psf/requests.git")

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