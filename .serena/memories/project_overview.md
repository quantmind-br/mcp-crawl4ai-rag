# Project Overview

## Purpose
This is a Model Context Protocol (MCP) server that integrates Crawl4AI web crawling, Qdrant vector database, and Neo4j knowledge graph to provide AI agents and coding assistants with advanced capabilities:

- **Web crawling and indexing** - Intelligent web crawling with auto-detection for sitemaps
- **GitHub repository indexing** - Clone and index GitHub repositories with multi-language parsing
- **Vector search and RAG** - Semantic search with contextual embeddings and reranking  
- **AI hallucination detection** - Validate AI-generated code against knowledge graphs
- **Multi-language code analysis** - Tree-sitter parsers for Python, JavaScript, Java, Go, Rust, C/C++

## Key Features
- **MCP Tool Architecture** - FastMCP-based server with async tools organized by functionality
- **Dual Database Strategy** - Qdrant for vectors, Neo4j for code structure graphs
- **Advanced RAG Strategies** - Hybrid search, contextual embeddings, agentic RAG, reranking
- **Multi-Provider API Support** - OpenAI, Azure, DeepInfra with fallback configuration
- **Docker-based Services** - Containerized Qdrant, Neo4j, and Redis services
- **GPU Acceleration** - Auto-detection for reranking models with CUDA/MPS support

## Architecture Components
- **MCP Server**: `src/core/app.py` - FastMCP application with async context management
- **Tools**: `src/tools/` - MCP tools (web_tools.py, github_tools.py, rag_tools.py, kg_tools.py)
- **Services**: `src/services/` - Core business logic (RAG, embedding, unified indexing)
- **Clients**: `src/clients/` - Database and API integrations
- **Knowledge Graphs**: `src/k_graph/` - Multi-language code analysis with Tree-sitter

## Target Users
AI agents, coding assistants, and developers needing advanced RAG capabilities with web crawling and code analysis features.