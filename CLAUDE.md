# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MCP (Model Context Protocol) server that integrates Crawl4AI, Qdrant vector database, and Neo4j knowledge graph to provide AI agents with advanced web crawling, GitHub repository indexing, vector search, and AI hallucination detection capabilities.

## Architecture Overview

### Core Components

- **MCP Server**: FastMCP-based server (`src/core/app.py`) with async context management
- **Tools**: MCP tools organized by functionality (`src/tools/`)
  - `web_tools.py` - Web crawling and URL processing
  - `github_tools.py` - GitHub repository indexing
  - `rag_tools.py` - Vector search and retrieval
  - `kg_tools.py` - Knowledge graph and hallucination detection
- **Services**: Core business logic (`src/services/`)
- **Clients**: Database and API integrations (`src/clients/`)
- **Knowledge Graphs**: Tree-sitter parsers for multi-language code analysis (`knowledge_graphs/`)

## Common Development Commands

### Server Management

```bash
# Start the MCP server
uv run -m src

# Alternative entry point
uv run python run_server.py

# Start with Docker services
setup.bat          # Windows
docker-compose up -d  # Linux/Mac
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test suites
uv run pytest tests/integration/
uv run pytest tests/rag/
uv run pytest tests/knowledge_graphs/

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_web_tools.py -v
```

### Code Quality

```bash
# Check linting
uv run ruff check .

# Fix linting issues automatically  
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Database Management

```bash
# Clean Qdrant vector database
uv run python scripts/clean_qdrant.py

# Fix dimension mismatches
uv run python scripts/define_qdrant_dimensions.py

# View Docker services
docker-compose logs qdrant
docker-compose logs neo4j
```

## MCP Tools Architecture

### MCP Tool Registration

Tools are registered in `src/core/app.py` and organized by functionality:

- **Web Tools** (`src/tools/web_tools.py`)
  - `crawl_single_page` - Crawl individual webpages
  - `smart_crawl_url` - Auto-detect URL types (sitemaps, recursive)
  
- **GitHub Tools** (`src/tools/github_tools.py`)  
  - `smart_crawl_github` - Clone and index GitHub repositories
  - `index_github_repository` - Unified indexing for both Qdrant and Neo4j

- **RAG Tools** (`src/tools/rag_tools.py`)
  - `get_available_sources` - List indexed sources
  - `perform_rag_query` - Semantic vector search
  - `search_code_examples` - Specialized code search

- **Knowledge Graph Tools** (`src/tools/kg_tools.py`)
  - `parse_github_repository` - Index code structure in Neo4j
  - `check_ai_script_hallucinations` - Validate AI-generated code
  - `query_knowledge_graph` - Cypher queries

### Context Management

The server uses a singleton context pattern (`src/core/context.py`) to manage:
- AsyncWebCrawler instances
- Qdrant client connections  
- Embedding models and caching
- Cross-encoder rerankers

## Environment Configuration

The server requires environment variables (`.env` file):

### Required Variables
```bash
# API Configuration
CHAT_API_KEY=your_openai_api_key
EMBEDDINGS_API_KEY=your_openai_api_key

# Database Configuration  
QDRANT_HOST=localhost
QDRANT_PORT=6333
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=password123
```

### RAG Strategy Flags
```bash
# Enable advanced features
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true  
USE_AGENTIC_RAG=true
USE_RERANKING=true
USE_KNOWLEDGE_GRAPH=true
```

### Docker Services

Required services via `docker-compose.yaml`:
- **Qdrant**: Vector database on port 6333
- **Neo4j**: Graph database on port 7474/7687

```bash
docker-compose up -d  # Start services
docker-compose logs   # View logs
```

## Testing Strategy

### Test Organization

Tests are organized by functionality matching the source structure:

```
tests/
├── integration/          # Full workflow tests
├── clients/              # Database client tests
├── tools/                # MCP tool tests
├── knowledge_graphs/     # Tree-sitter parser tests
├── rag/                  # RAG service tests
└── conftest.py          # Shared pytest fixtures
```

### Key Test Commands

```bash
# Full test suite
uv run pytest

# Integration tests (requires Docker services)
uv run pytest tests/integration/

# Unit tests only
uv run pytest tests/clients/ tests/tools/

# Performance benchmarks
uv run pytest tests/performance/
```

### Test Fixtures

Key fixtures in `tests/conftest.py`:
- `mock_crawler` - Mocked AsyncWebCrawler
- `mock_qdrant` - Mocked Qdrant client
- `sample_github_repo` - Test repository data

## Tree-sitter Multi-Language Parsing

### Language Support

The knowledge graph system supports parsing multiple programming languages via Tree-sitter grammars in `knowledge_graphs/grammars/`:

- **Python** - Class/function extraction, imports
- **JavaScript/TypeScript** - Module analysis, function definitions  
- **Java** - Class hierarchies, method signatures
- **Go** - Package structure, function definitions
- **Rust** - Module system, trait implementations
- **C/C++** - Header analysis, function declarations

### Parser Architecture

Key files for language parsing:
- `knowledge_graphs/language_parser.py` - Tree-sitter integration
- `knowledge_graphs/query_patterns/` - Language-specific queries
- `knowledge_graphs/parse_repo_into_neo4j.py` - Repository analysis

### Usage Examples

```bash
# Parse repository for hallucination detection
uv run python knowledge_graphs/parse_repo_into_neo4j.py https://github.com/user/repo

# Check AI-generated Python code
uv run python knowledge_graphs/ai_hallucination_detector.py script.py
```

## Key Dependencies

### Core Dependencies (from pyproject.toml)
- **crawl4ai==0.6.2** - Web crawling engine
- **mcp==1.7.1** - Model Context Protocol framework  
- **qdrant-client>=1.12.0** - Vector database client
- **neo4j>=5.28.1** - Graph database client
- **openai==1.71.0** - LLM and embedding API client
- **sentence-transformers>=5.0.0** - Text embeddings and reranking
- **tree-sitter>=0.23.0** - Multi-language code parsing

### Package Management
Use `uv` for all dependency management:
```bash
# Add new dependency
uv add package-name

# Add development dependency  
uv add --dev pytest-package

# Sync after pyproject.toml changes
uv sync

# Never edit pyproject.toml directly - always use uv commands
```

## Important Notes

- **Entry Points**: Use `uv run -m src` or `uv run python run_server.py`
- **Docker Required**: Qdrant and Neo4j services must be running
- **Environment Variables**: Copy `.env.example` to `.env` with API keys
- **Multi-Language Support**: Tree-sitter grammars are auto-initialized
- **Testing**: Run `uv run pytest` for full test suite
- **GPU Support**: Auto-detected for reranking models

## Unicode Character Guidelines

**CRITICAL**: To avoid `UnicodeEncodeError` in Windows console environments, follow these strict guidelines:

### ❌ NEVER Use These Unicode Characters
- **Checkmarks**: ✅ ✓ ❌ ✗ 
- **Emojis**: 🚀 🔧 📋 🎯 🔍 🕸️ 🐍 etc.
- **Special symbols**: ➖ ⚡ 💾 🔄 📊 🛡️ 🧠 🎉 💥
- **Arrows**: ← → ↑ ↓ ⬅️ ➡️ ⬆️ ⬇️

### ✅ Use ASCII Alternatives Instead
```python
# WRONG - Will cause UnicodeEncodeError
print("✅ SUCCESS: Operation completed")
print("❌ FAILED: Operation failed")

# CORRECT - Use ASCII alternatives
print("SUCCESS: Operation completed")
print("FAILED: Operation failed")
print("* Operation completed successfully")
print("ERROR: Operation failed")
```

### Testing and Debugging Guidelines
```bash
# When testing with print statements, always use ASCII-only text
# WRONG:
uv run python -c "print('✓ Test passed')"

# CORRECT:
uv run python -c "print('SUCCESS: Test passed')"
```

### Code Comments and Documentation
```python
# WRONG - Unicode in comments can cause issues
# ✅ This function works correctly
# 🚀 Performance optimized

# CORRECT - ASCII-only comments
# SUCCESS: This function works correctly  
# PERFORMANCE: Optimized implementation
```

**Remember**: Windows console (cp1252 encoding) cannot display Unicode characters. Always use ASCII alternatives to prevent encoding errors that interrupt development workflow.

---

_This MCP server provides advanced RAG capabilities for AI agents and coding assistants._
