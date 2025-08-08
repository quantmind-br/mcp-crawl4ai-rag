# Project Overview - Crawl4AI MCP RAG

## Purpose
**Advanced Web Crawling and RAG Capabilities for AI Agents** - A powerful Model Context Protocol (MCP) server that integrates Crawl4AI, Qdrant, and Neo4j to provide AI agents and coding assistants with intelligent web crawling, GitHub repository indexing, vector search, and AI hallucination detection capabilities.

## Key Features
- **Web Crawling**: Intelligent crawling with auto-detection of sitemaps and recursive crawling
- **GitHub Integration**: Clone and index GitHub repositories with markdown documentation
- **Vector Search**: Semantic search using Qdrant vector database with reranking
- **Knowledge Graph**: Neo4j integration for code structure analysis and AI hallucination detection  
- **RAG Strategies**: Multiple strategies including contextual embeddings, hybrid search, agentic RAG
- **Multi-Provider Support**: OpenAI, Azure OpenAI, DeepInfra, and other OpenAI-compatible providers
- **GPU Acceleration**: Optional GPU support for reranking models
- **Redis Caching**: Optional Redis integration for embedding caching

## Architecture
- **MCP Server**: FastMCP-based server with async tools
- **Vector Database**: Qdrant for semantic search and storage
- **Web Crawler**: Crawl4AI with smart URL detection and GitHub integration
- **Knowledge Graph**: Neo4j for code structure analysis  
- **Device Manager**: Automatic GPU/CPU detection and fallback

## Target Users
- AI coding assistants and agents
- Developers building RAG applications
- Users needing intelligent web crawling and document indexing capabilities