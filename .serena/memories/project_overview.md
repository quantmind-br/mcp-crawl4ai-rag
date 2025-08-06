# Crawl4AI RAG MCP Server - Project Overview

## Project Purpose
Advanced web crawling and RAG capabilities for AI agents and coding assistants through Model Context Protocol (MCP). Integrates Crawl4AI for web scraping, Qdrant for vector storage, and Neo4j for knowledge graph-based AI hallucination detection.

## Core Architecture
- **MCP Server**: FastMCP-based server providing 11+ async tools for crawling and RAG operations
- **Vector Database**: Qdrant client with multi-collection support for documents and code
- **Web Crawler**: Crawl4AI integration with sitemap detection and parallel processing
- **Knowledge Graph**: Neo4j integration for codebase analysis and AI hallucination detection
- **Device Management**: Automatic GPU detection, CUDA optimization with Windows compatibility

## Key Capabilities
- Web crawling with intelligent URL detection
- Repository indexing and GitHub integration
- Semantic search with vector+RAG capabilities
- AI-generated code validation via knowledge graphs
- Multi-provider API support with fallback configurations