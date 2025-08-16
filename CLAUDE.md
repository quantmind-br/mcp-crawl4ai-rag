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
- **Knowledge Graphs**: Tree-sitter parsers for multi-language code analysis (`src/k_graph/`)

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

# Run by category (new hierarchical structure)
uv run pytest tests/unit/                    # Unit tests by module
uv run pytest tests/specialized/             # Domain-specific tests  
uv run pytest tests/infrastructure/          # Infrastructure tests
uv run pytest tests/integration/             # End-to-end tests

# Run specific modules
uv run pytest tests/unit/tools/              # MCP tools tests
uv run pytest tests/specialized/embedding/   # Embedding system tests
uv run pytest tests/specialized/knowledge_graphs/ # Knowledge graph tests

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/unit/tools/test_web_tools.py -v
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
  - `index_github_repository` - Unified GitHub repository indexing for both Qdrant and Neo4j with intelligent content extraction

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

### MCP Tools Timeout Configuration

**Status**: Environment configuration prepared for future FastMCP timeout support

The timeout configuration infrastructure has been implemented to prevent client disconnections during long-running operations. The timeout constants are available for when FastMCP adds timeout support:

```bash
# MCP Tools Timeout Configuration (seconds)
# Controls maximum execution time for MCP tools to prevent client disconnections

# Quick operations: Simple queries, data retrieval, status checks
MCP_QUICK_TIMEOUT=60

# Medium operations: RAG queries, analysis tasks, script validation  
MCP_MEDIUM_TIMEOUT=300

# Long operations: Single page crawls, repository parsing, complex analysis
MCP_LONG_TIMEOUT=1800

# Very long operations: Multi-page crawls, full repository indexing, bulk processing
MCP_VERY_LONG_TIMEOUT=3600
```

#### Timeout Categories and Tool Mapping

| Timeout Category | Duration | Tools |
|------------------|----------|--------|
| **QUICK (60s)** | 1 minute | `get_available_sources`, `query_knowledge_graph` |
| **MEDIUM (300s)** | 5 minutes | `perform_rag_query`, `search_code_examples`, `check_ai_script_hallucinations` |
| **LONG (1800s)** | 30 minutes | `crawl_single_page`, `parse_github_repository` |
| **VERY_LONG (3600s)** | 1 hour | `smart_crawl_url`, `index_github_repository` |

#### Performance Tuning Guidelines

- **Development**: Use shorter timeouts (30s/120s/600s/1200s) for faster feedback
- **Production**: Use standard timeouts for reliability
- **Enterprise**: Increase timeouts (120s/600s/3600s/7200s) for large-scale operations
- **CI/CD**: Set conservative timeouts to handle resource constraints

#### Troubleshooting Timeout Issues

**Common timeout problems:**
- Client disconnections during repository indexing → Increase `MCP_VERY_LONG_TIMEOUT`
- Web crawling failures on complex sites → Increase `MCP_LONG_TIMEOUT`
- RAG query timeouts under load → Increase `MCP_MEDIUM_TIMEOUT`

**Diagnostic commands:**
```bash
# Check current timeout configuration
grep "MCP_.*_TIMEOUT" .env

# Test server startup with timeout config
uv run -m src --test-mode

# Monitor tool execution times
tail -f logs/mcp-server.log | grep "timeout"
```

**Implementation Note**: Currently, the timeout constants are defined and available in the application, but FastMCP does not yet support timeout parameters at the tool level. The infrastructure is ready for when this feature becomes available in the MCP specification or FastMCP library.

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

Tests are organized in a hierarchical structure for better maintainability:

```
tests/
├── unit/                          # Unit tests by module
│   ├── clients/                   # Database and API clients
│   ├── core/                      # Core application functionality
│   ├── services/                  # Business logic services
│   ├── tools/                     # MCP tools
│   └── utils/                     # Utility functions
├── specialized/                   # Domain-specific functionality
│   ├── embedding/                 # Embedding system tests
│   ├── knowledge_graphs/          # Knowledge graph tests
│   └── device_management/         # GPU/CPU device management
├── infrastructure/                # Infrastructure components
│   ├── storage/                   # Qdrant, Redis storage tests
│   └── validation/                # Data validation tests
├── integration/                   # End-to-end workflow tests
├── fixtures/                      # Test data and samples
└── conftest.py                   # Shared pytest fixtures
```

### Key Test Commands

```bash
# Full test suite
uv run pytest

# By test category
uv run pytest tests/unit/                    # All unit tests
uv run pytest tests/specialized/             # Domain-specific tests
uv run pytest tests/infrastructure/          # Infrastructure tests
uv run pytest tests/integration/             # Integration tests (requires Docker)

# Specific functionality
uv run pytest tests/unit/tools/              # MCP tools only
uv run pytest tests/specialized/embedding/   # Embedding system only
```

### Test Fixtures

Key fixtures in `tests/conftest.py`:
- `mock_crawler` - Mocked AsyncWebCrawler
- `mock_qdrant` - Mocked Qdrant client
- `sample_github_repo` - Test repository data

## Knowledge Graph Architecture (src/k_graph/)

### Modular Structure

The knowledge graph system has been modularized into organized components:

```
src/k_graph/
├── core/                          # Core interfaces and models
│   ├── interfaces.py              # Language parser interfaces
│   └── models.py                  # Data models for parsing
├── parsing/                       # Multi-language parsing engine
│   ├── tree_sitter_parser.py     # Tree-sitter integration
│   ├── simple_fallback_parser.py # Fallback for unsupported languages
│   ├── parser_factory.py         # Parser selection logic
│   └── query_patterns/           # Language-specific AST queries
│       ├── python_queries.py     # Python AST patterns
│       ├── javascript_queries.py # JavaScript/TypeScript patterns
│       ├── java_queries.py       # Java AST patterns
│       └── ...                   # Other language patterns
├── analysis/                      # Code analysis and validation
│   ├── hallucination_detector.py # AI code validation
│   ├── validator.py              # Knowledge graph validation
│   ├── script_analyzer.py        # Script analysis engine
│   └── reporter.py               # Analysis reporting
└── services/                      # High-level services
    └── repository_parser.py      # Repository processing orchestrator
```

### Language Support

Multi-language parsing with Tree-sitter grammars:
- **Python** - Class/function extraction, imports, docstrings
- **JavaScript/TypeScript** - Module analysis, function definitions, exports
- **Java** - Class hierarchies, method signatures, annotations
- **Go** - Package structure, function definitions, interfaces
- **Rust** - Module system, trait implementations, cargo metadata
- **C/C++** - Header analysis, function declarations, includes

### Usage Examples

```bash
# Parse repository for hallucination detection
uv run python scripts/query_knowledge_graph.py

# Check AI-generated Python code (from MCP tools)
# Use kg_tools.check_ai_script_hallucinations MCP tool
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

## Unified Indexing Service

The core service (`src/services/unified_indexing_service.py`) orchestrates repository processing:

### Key Features
- **Dual-destination indexing**: Simultaneous Qdrant (RAG) and Neo4j (KG) processing
- **Cross-system file linking**: Consistent file_id linking between systems
- **Modular processor architecture**: GitHub, web, and custom content processors
- **Resource management**: Automatic cleanup and memory management
- **Comprehensive reporting**: Detailed processing statistics and error handling

### Processing Pipeline
1. **Repository cloning** with size and file type validation
2. **File discovery** using configurable patterns and filters
3. **Parallel processing** for RAG embeddings and KG parsing
4. **Cross-system linking** via unified file_id generation
5. **Storage optimization** with batch operations and caching

## Important Notes

- **Entry Points**: Use `uv run -m src` or `uv run python run_server.py`
- **Docker Required**: Qdrant and Neo4j services must be running
- **Environment Variables**: Copy `.env.example` to `.env` with API keys
- **Multi-Language Support**: Tree-sitter grammars are auto-initialized in `src/utils/grammar_initialization.py`
- **Testing**: Run `uv run pytest` - see hierarchical structure in `tests/README.md`
- **GPU Support**: Auto-detected for reranking models via `src/device_manager.py`
- **Knowledge Graph**: Modularized in `src/k_graph/` with comprehensive language support

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
