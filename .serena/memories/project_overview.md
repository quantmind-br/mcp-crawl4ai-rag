# MCP Crawl4AI RAG Server - Project Overview

## Purpose
Advanced web crawling and RAG (Retrieval Augmented Generation) capabilities for AI agents through an MCP (Model Context Protocol) server. Integrates Crawl4AI, Qdrant vector database, and Neo4j knowledge graph to provide intelligent web crawling, GitHub repository indexing, vector search, and AI hallucination detection.

## Key Capabilities
- **Web Crawling**: Single page and intelligent multi-page crawling with sitemap detection
- **GitHub Integration**: Repository cloning and documentation indexing
- **Vector Search**: Semantic search with RAG capabilities using Qdrant
- **Code Analysis**: Tree-sitter based multi-language parsing for knowledge graphs
- **Hallucination Detection**: Neo4j-based validation of AI-generated code
- **Multi-Provider Support**: Flexible API configuration with fallback support

## Target Users
- AI coding assistants
- Claude Code users
- AI agents requiring web crawling and RAG capabilities
- Developers building documentation search systems

## Architecture
- **MCP Server**: FastMCP-based async server
- **Vector DB**: Qdrant for semantic search
- **Knowledge Graph**: Neo4j for code structure analysis  
- **Web Crawler**: Crawl4AI with smart detection
- **Device Manager**: Auto GPU/CPU detection with fallback