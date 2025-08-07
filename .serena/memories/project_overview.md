# Project Overview

## Project Purpose
This is a **Crawl4AI RAG MCP Server** that provides AI agents and coding assistants with advanced web crawling and RAG (Retrieval Augmented Generation) capabilities through the Model Context Protocol (MCP). The system integrates:

- **Crawl4AI** for intelligent web scraping
- **Qdrant** for vector storage and semantic search
- **Neo4j** for knowledge graph-based AI hallucination detection (optional)

## Key Features
- **Smart Web Crawling**: Auto-detects sitemaps, recursive crawling, GitHub repository indexing
- **Vector Search**: Semantic search with source filtering and optional reranking
- **Code Analysis**: Specialized code search and AI hallucination detection
- **Multi-Provider API Support**: Flexible configuration for OpenAI, DeepInfra, Azure, etc.
- **Advanced RAG Strategies**: Contextual embeddings, hybrid search, agentic RAG, reranking
- **GPU Acceleration**: Optional GPU support for reranking models

## Target Use Cases
1. **Documentation RAG**: Index and search documentation websites
2. **AI Coding Assistant**: Extract and search code examples with hallucination detection  
3. **GitHub Repository Analysis**: Index repository structure for code validation
4. **Multi-Source Knowledge Base**: Combine multiple documentation sources with filtering

## Architecture Components
- **MCP Server**: `src/crawl4ai_mcp.py` - Main FastMCP server with async tools
- **Vector Database**: Qdrant client wrapper for document and code storage
- **Web Crawler**: Crawl4AI integration with parallel processing
- **Knowledge Graph**: Neo4j integration for code structure analysis
- **Device Management**: GPU/CPU detection and fallback for reranking
- **Embedding Services**: Multi-provider API support with fallback strategies